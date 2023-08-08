import datetime
import gc
import math
import multiprocessing
import os
import pickle
import shutil
import time
from multiprocessing import Process, Queue
from operator import itemgetter
import psutil
from tools.lang import newu, constructSMfromDict
import networkx as nx
from tools.rw import saveDictAsJson, readDictFromJson


class Pseudolemmatizer:

    dateobj = datetime.datetime

    def __init__(self, dirpath, name, mode='prepare', loadpath=None, wordforms=None, simcsv=None):
        """
        Initiates a lemmatization model. The model will use dirpath as a directory to store temporary files as well as
        the path for exporting or loading itself. If loadpath is given, files will be load from this path instead.
        Mode defines the initialization behaviour:
            * 'prepare' = Use the given wordforms to prepare the model (default). Export afterwards.
            * 'train' = Use the given wordforms to prepare and train a model. Export afterwards.
            * 'load' = Load a trained model from the given dir- or loadpath.
            * 'loadbase' = Load a prepared but untrained model from the given dir- or loadpath.
            * 'trainbase' = Perform loadbase. Train and export the model afterwards.
            * None or any other value = Do nothing.

        Preparation includes calculating similarity values for all possible pairings of the tokens in the wordforms
        list and constructing a graph representation of words (as nodes) and similarities (as edges). If a path to a
        simcsv with precalculated similarity values is supplied, calculation is skipped and the graph is constructed
        based on the given values.

        Training includes detecting wordgroups by community detection and assigning lemmata based on standard
        parameters. If non-standard parameters are needed, use None as initialization behaviour and apply the different
        functions separately.

        :param dirpath: The path to a directory, where the lemmatizer may store all model files.
        :param mode: Init mode. See the list above for possible values
        :param loadpath: If loadpath is given and mode is 'load' or 'loadbase', the files aren't loaded from dirpath,
        but from loadpath
        :param wordforms: A list of all wordforms that the model should use for training.
        :param simcsv: Optional path to precalculated similarities used for graph construction.
        """

        self.name = name
        self.state = -1
        self.G = nx.Graph()
        self.communities = []
        self.wordgroups = []
        self.dirpath = dirpath
        self.csvpath = os.path.join(dirpath, 'sims-all.csv')
        self.info = {}
        self.forms = None
        self.lemmas = None
        if mode is not None:
            if mode == 'load':
                self.load(loadpath=loadpath)
            elif mode == 'loadbase':
                self.load(full=False, loadpath=loadpath)
            elif mode == 'trainbase':
                self.load(full=False, loadpath=loadpath)
                self.train()
                self.export()
            else:
                self.forms = sorted(wordforms)
                self.lemmas = [None for i in range(len(self.forms))]
                self.info['types'] = len(self.forms)
                if mode == 'prepare':
                    self.prepare(simcsv)
                    self.export()
                else:
                    self.prepare(simcsv)
                    self.train()
                    self.export()
        else:
            self.forms = sorted(wordforms)
            self.lemmas = [None for i in range(len(self.forms))]
            self.info['types'] = len(self.forms)

    def load(self, full=True, loadpath=None):
        """
        Loads the model from model export files.

        :param full: Whether or not a trained or a prepared-only model should be loaded.
        :param loadpath: The path from where to load the files, if not identical to the model's dirpath.
        """

        print(self.date() + ' PL-'+self.name+f': Loading model...')
        if full:
            if loadpath is not None:
                shutil.copy(os.path.join(loadpath, 'wordgroups.json'), os.path.join(self.dirpath, 'wordgroups.json'))
            print(self.date() + ' PL-'+self.name+f': ...importing word groups...')
            self.loadcommunities(os.path.join(self.dirpath, 'wordgroups.json'))
        print('...loading model infos...')
        if loadpath is not None:
            shutil.copy(os.path.join(loadpath, 'info.json'), os.path.join(self.dirpath, 'info.json'))
        self.info = readDictFromJson(os.path.join(self.dirpath, 'info.json'))
        print(self.date() + ' PL-'+self.name+f': ...loading words...')
        if loadpath is not None:
            shutil.copy(os.path.join(loadpath, 'words.json'), os.path.join(self.dirpath, 'words.json'))
        words = readDictFromJson(os.path.join(self.dirpath, 'words.json'))
        self.forms = words['forms']
        if full:
            self.lemmas = words['lemmas']
        else:
            self.lemmas = [None for i in range(len(self.forms))]
        del words
        print(self.date() + ' PL-'+self.name+f': ...loading graph...')
        if loadpath is not None:
            shutil.copy(os.path.join(loadpath, 'graph.pickle'), os.path.join(self.dirpath, 'graph.pickle'))
        with open(os.path.join(self.dirpath, 'graph.pickle'), 'rb') as gp:
            self.G = pickle.load(gp)
        if full:
            self.state = 1
        else:
            self.state = 0
        print(self.date() + ' PL-'+self.name+f': Done.')

    def loadcommunities(self, wgpath):
        """
        Saves the given human-readable json representation of wordgroups in the internal format.
        """

        communities = list(readDictFromJson(wgpath))
        for com in communities:
            self.communities.append((
                com['lemma'],
                com['meansim'],
                com['quality'],
                [self.forms.index(w) for w in com['words']]
            ))

    def readSims(self, csvpath):
        no = 0
        with open(csvpath, 'r') as csv:
            nextline = csv.readline()
            while nextline is not None and nextline != '':
                sim = nextline.strip().split(',')
                if sim[0] != sim[1] and float(sim[3]) > 0:
                    self.G.add_edge(self.forms.index(sim[0]), self.forms.index(sim[1]), weight=float(sim[3]))
                no += 1
                nextline = csv.readline()
        self.info['simpairs'] = no

    def export(self):
        """
        Exports the model in human-readable formats to the model's dirpath:

        * words.json: Containing the wordforms and their corresponding lemmata as lists.
        * wordgroups.json: Containing the found groups of wordforms representing one lemma, as well as their lemma, meansim and quality measures.
        * graph.json: Containing the nodes and edges of the model's graph together with the edge weights.

        Additionally, the model's graph is pickled for fast loading.
        If the model isn't trained yet, but prepared, only the graph will be exported and the words.json will only
        contain the wordforms, not the corresponding lemmata.
        """

        print(self.date() + ' PL-'+self.name+f': Exporting model...')
        print(self.date() + ' PL-'+self.name+f': ...exporting words...')
        if self.state == 1:
            words = {
                'forms': self.forms,
                'lemmas': self.lemmas
            }
        else:
            words = {
                'forms': self.forms
            }
        saveDictAsJson(os.path.join(self.dirpath, 'words.json'), words)
        gc.collect()

        if self.state == -1:
            print(self.date() + ' PL-'+self.name+f': Model is untrained. Nothing else to export.')
            return
        if self.state == 1:
            print(self.date() + ' PL-'+self.name+f': ...exporting word groups...')
            saveDictAsJson(os.path.join(self.dirpath, 'wordgroups.json'), self.formatcommunities())
            gc.collect()
        print('...exporting model infos...')
        saveDictAsJson(os.path.join(self.dirpath, 'info.json'), self.info)
        print(self.date() + ' PL-'+self.name+f': ...exporting human-readable graph...')
        saveDictAsJson(os.path.join(self.dirpath, 'graph.json'),
                       nx.node_link_data(self.G, link='edges', source='w1', target='w2'))
        gc.collect()
        print(self.date() + ' PL-'+self.name+f': ...serializing graph...')
        with open(os.path.join(self.dirpath, 'graph.pickle'), 'wb') as gp:
            pickle.dump(self.G, gp, pickle.HIGHEST_PROTOCOL)
        print(self.date() + ' PL-'+self.name+f': Done.')

    def formatcommunities(self):
        """
        Returns a human-readable form of the model's wordgroups.

        :return: A json representation of the wordgroups.
        """

        communities = []
        for com in self.communities:
            communities.append({
                'lemma': com[0],
                'count': len(com[3]),
                'meansim': com[1],
                'quality': com[2],
                'words': [self.forms[wid] for wid in com[3]]
            })
        return communities

    def prepare(self, smcustomization=None, gapex=1.0, gapstart=0.5, phonetic=False, disfavor=0,
                combilists=None, simcsv=None):
        """
        Prepare the model for training. This includes calculating similarity values for all possible pairings of the
        model's wordforms and constructing a graph representation of words (as nodes) and similarities (as edges). If a
        path to a simcsv with precalculated similarity values is supplied, calculation is skipped and the graph is
        constructed based on the given values.

        :param simcsv: Optional path to precalculated similarities used for graph construction.
        """

        print(self.date() + ' PL-'+self.name+f': Preparing model.')
        i = 0
        self.info['simconf'] = {
            'SM': smcustomization if smcustomization is not None else {},
            'gapex': gapex,
            'gapstart':gapstart,
            'phonetic': phonetic,
            'combilists': combilists if combilists is not None else {},
            'disfavor': disfavor
        }

        print(self.date() + ' PL-'+self.name+f': Adding words as nodes.')
        for word in self.forms:
            self.G.add_node(i, wf=word)
            i += 1

        print(self.date() + ' PL-'+self.name+f': Calculating similarities and adding them as edges.')

        if simcsv is None:
            self.__calculateSimsAllThread(smcustomization=smcustomization, gapex=gapex, gapstart=gapstart,
                                          phonetic=phonetic, disfavor=disfavor, combilists=combilists)
        else:
            self.readSims(os.path.join(self.dirpath, simcsv))

        self.state = 0
        print(self.date() + ' PL-'+self.name+f': Preparation finished:')
        self.info['nodes'] = self.G.number_of_nodes()
        self.info['edges'] = self.G.number_of_edges()
        print('Nodes: ' + str(self.info['nodes']))
        print('Edges: ' + str(self.info['edges']))

    def __calculateSimsAllThread(self, smcustomization=None, gapex=1.0, gapstart=0.5, phonetic=False,
                                 disfavor=0, combilists=None):
        """
        Calculate the similarity between all possible pairings of the words in the lemmatizers wordform list. The
        calculation is done with as many processes as are available on the machine that the programme is executed on.
        The similarities are saved as csv file in the dirpath of the lemmatizer object.
        """

        print(self.date() + ' PL-'+self.name+f': Calculating batches for calculation')
        pno = len(psutil.Process().cpu_affinity())
        batches = [(i, len(self.forms) - i) for i in range(len(self.forms))]
        full = int(((len(self.forms) + 1) * len(self.forms)) / 2)
        self.info['simpairs'] = full
        part = full/pno
        partitions = []
        counts = []

        # Calculate partitions for multithreading
        for p in range(pno):
            nextbatch = max(batches, key=lambda b: b[1])
            batches.remove(nextbatch)
            scope = [nextbatch[0]]
            reach = nextbatch[1]
            while reach <= part and len(batches) > 0:
                possbatches = [batch for batch in batches if reach + batch[1] <= part]
                if len(possbatches) == 0:
                    nextbatch = min(batches, key=lambda b: b[1])
                else:
                    nextbatch = max(possbatches, key=lambda b: b[1])
                batches.remove(nextbatch)
                scope.append(nextbatch[0])
                reach += nextbatch[1]
            partitions.append(scope)
            counts.append(reach)

        partitions[-1].extend([i[0] for i in batches])
        counts[-1] += sum([i[1] for i in batches])
        print(self.date() + ' PL-'+self.name+f': Created '+str(pno)+' batches with quantity:')
        print(counts)

        # Start threads
        print(self.date() + ' PL-'+self.name+f': Starting subprocesses.')
        tmppath = os.path.join(self.dirpath, 'tmp')
        os.mkdir(tmppath)
        processes = []
        manager = multiprocessing.Manager()
        q = manager.Queue()
        for p in range(pno):
            csvpath = os.path.join(tmppath, str(p + 1) + '-' + str(pno) + '.csv')
            p = Process(target=self._calculateSimsBatch, name=str(p+1)+'/'+str(pno),
                        args=(partitions[p], self.forms, counts[p], str(p+1)+'/'+str(pno),
                              csvpath, q, smcustomization, gapex, gapstart, phonetic, disfavor, combilists))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        print(self.date() + ' PL-'+self.name+f': Adding edges.')
        for p in range(pno):
            gc.collect()
            edges = q.get()
            for edge in edges:
                self.G.add_edge(edge[0], edge[1], weight=edge[2])

        print(self.date() + ' PL-'+self.name+f': Joining similarity CSVs.')
        with open(self.csvpath, 'wb') as ff:
            for p in range(1, pno+1):
                f = os.path.join(tmppath, str(p) + '-' + str(pno) + '.csv')
                with open(f, 'rb') as bf:
                    shutil.copyfileobj(bf, ff)
                os.remove(f)
        os.rmdir(tmppath)

    def _calculateSimsBatch(self, idxes, words, full, name: str, csvpath, q: Queue, smcustomization=None, gapex=1.0,
                            gapstart=0.5, phonetic=False, disfavor=0, combilists=None):
        if smcustomization is None:
            SM = None
        else:
            SM = constructSMfromDict(smcustomization)
        print(self.date() + ' PL-'+self.name+f'-' + name + ': Calculating similarities for word pair 1 of ' + str(full))
        step = int((10 ** (int(math.log10(full)))) / 2)
        no = 1
        nextno = 10
        nextwrite = 1000000
        edges = []
        sims = []
        with open(csvpath, 'w') as csv:
            ts = time.time()
            for i in idxes:
                for j in range(i, len(words)):
                    if no == nextno:
                        print(self.date() + ' PL-'+self.name+f'-' + name + ':... for word pair ' + str(nextno) + ' of ' + str(full))
                        if nextno < step:
                            nextno = nextno * 10
                        else:
                            nextno += step
                    if len(sims) == nextwrite:
                        for s in sims:
                            csv.write(s)
                        sims = []
                        print(self.date() + ' PL-'+self.name+f'-' + name + ':... written up to ' + str(no-1) + ' to csv.')
                        gc.collect()
                    sim = newu(words[i], words[j], SM=SM, gapex=gapex, gapstart=gapstart,
                               phonetic=phonetic, disfavor=disfavor, combilists=combilists)
                    sims.append(
                        f'{words[i]},{words[j]},{sim["sim"]},{sim["normsim"]},{sim["alignx"]},{sim["aligny"]}\n')
                    if i != j and sim['normsim'] > 0:
                        edges.append([i, j, sim['normsim']])
                    no += 1
            for s in sims:
                csv.write(s)
            te = time.time()
        print(self.date() + ' PL-'+self.name+f'-' + name + ': Done in '+time.strftime('%H:%M:%S', time.gmtime(te-ts))+'.')
        q.put_nowait(edges)

    def train(self, algo='louvain', resolution=1, enforcequality=True):
        """
        Trains the model. To apply this function, the model needs to have been prepared first by the prepare() function.
        The training works by detecting communities in the model's graph using the community detection algorithm given
        in algo. Possible algorithms are:

        * "louvain" = Louvain algorithm, default (docs: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.louvain.louvain_communities.html)
        * "mod" = greedy modularity maximization (https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.modularity_max.greedy_modularity_communities.html)
        * "lpa" = asynchronous labelpropagation (docs: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.label_propagation.asyn_lpa_communities.html)
        * "girvan_newman" = Girvan-Newman algorithm (docs: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.centrality.girvan_newman.html)

        If louvain or mod is used, the resolution parameter will determine the resolution value used for the community
        detection.

        If enforcequality is True (default), then the community detection will run iteratively until either all found
        communities are fully connected, or until no new communities are found within the exiting ones.

        :param algo: The algorithm to use for the community detection.
        :param resolution: The resolution value to use when detecting communities with louvain or mod.
        :param enforcequality: Whether or not to train iteratively, favouring fully connected communities.
        """

        print(self.date() + ' PL-'+self.name+f': Training started.')
        print(self.date() + ' PL-'+self.name+f': Calculating communities using ' + algo + ' algorithm...')

        communities = self.detectCommunities(
            self.G,
            algo=algo,
            resolution=resolution
        )
        print(self.date() + ' PL-'+self.name+f': Done.')
        if enforcequality:
            print(self.date() + ' PL-'+self.name+f': Checking quality of the detected communities.')
            coverage = 0
            oldcov = 0
            it = 1
            weakcomms = []
            while coverage != len(self.forms):
                oldcov = coverage
                for comi in reversed(range(len(communities))):
                    if communities[comi][1] < 1:
                        weakcomms.append(communities.pop(comi))
                    else:
                        coverage += len(communities[comi][2])
                self.communities.extend(self.assignRepresentative(communities))
                if coverage == oldcov:
                    print(self.date() + ' PL-'+self.name+f': Community detection converges.')
                    self.info['comno'] = len(self.communities) + len(weakcomms)
                    self.info['weakcomno'] = len(weakcomms)
                    self.communities.extend(self.assignRepresentative(weakcomms))
                    break
                communities = []
                gc.collect()
                print(self.date() +
                      f' PL-'+self.name+f': Covered {coverage} of {len(self.forms)} wordforms. Starting iteration no. {it}.')
                for wcom in weakcomms:
                    communities.extend(self.detectCommunities(
                        nx.Graph(nx.subgraph(self.G, wcom[2])),
                        algo=algo,
                        resolution=resolution
                    ))
                weakcomms = []
                it += 1
        else:
            self.communities.extend(self.assignRepresentative(communities))
            self.info['comno'] = len(self.communities)

        print(self.date() + ' PL-'+self.name+f': Assigning lemmata.')
        self.assignLemmata()

        self.info['lemmacount'] = len(set(self.lemmas))

        print(self.date() + ' PL-'+self.name+f': Training finished: '+ str(self.info['lemmacount']) + ' unique lemmata.')

        self.state = 1

    def qualitizecommunities(self, communities):
        # Todo: Maximize similarity instead of density
        # or Todo: find the two nodes that are'nt connected to each other, remove the one with lowest similarity

        coverage = 0
        newcommunities=[]
        outcommunities = []
        for com in communities:
            SG = nx.Graph(nx.subgraph(self.G, com[2]))
            deg = sorted(
                dict(nx.degree_centrality(SG)).items(),
                key=lambda a: a[1])
            highest = deg[0]
            deg.reverse()
            highestidx = deg.index(highest)
            outliers = [n[0] for n in deg[0:highestidx]]
            reals = [n[0] for n in deg[highestidx:]]
            coverage += len(reals)
            omeansim, oquality = self.evaluatecommunity(outliers, SG)
            rmeansim, rquality = self.evaluatecommunity(reals, SG)

            newcommunities.append((
                rmeansim,  # mean similarity
                rquality,  # quality of the group
                reals  # list of wordform ids
            ))

            if oquality == 1:
                newcommunities.append((
                    omeansim,  # mean similarity
                    oquality,  # quality of the group
                    outliers  # list of wordform ids
                ))
            else:
                outcommunities.append((
                    omeansim,  # mean similarity
                    oquality,  # quality of the group
                    outliers  # list of wordform ids
                ))

        return newcommunities, outcommunities, coverage

        # Source: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.centrality.girvan_newman.html

    def detectCommunities(self, G, algo='louvain', resolution=1):
        """
        Detects communities in the given graph G using the selected algorithm. Possible algorithms are:

        * "louvain" = Louvain algorithm, default (docs: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.louvain.louvain_communities.html)
        * "mod" = greedy modularity maximization (https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.modularity_max.greedy_modularity_communities.html)
        * "lpa" = asynchronous labelpropagation (docs: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.label_propagation.asyn_lpa_communities.html)
        * "girvan_newman" = Girvan-Newman algorithm (docs: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.centrality.girvan_newman.html)

        If louvain or mod is used, the resolution parameter will determine the resolution value used for the community
        detection.

        :param algo: The algorithm to use for the community detection.
        :param resolution: The resolution value to use when detecting communities with louvain or mod.
        :return: A list of the found communities, each community being a tuple of the form: (meansim, quality, list of
        word indices)
        """
        communities = []
        if algo == 'girvan_newman':
            communities_generator = nx.community.girvan_newman(G, most_valuable_edge=self.heaviestedge)
            foundcomms = next(communities_generator)
        elif algo == 'lpa':
            foundcomms = nx.community.asyn_lpa_communities(G, weight='weight')
        elif algo == 'mod':
            foundcomms = nx.community.greedy_modularity_communities(G, weight='weight', resolution=resolution)
        else:
            foundcomms = [list(c) for c in nx.community.louvain_communities(
                G, weight='weight', resolution=resolution)]

        comms = sorted(map(sorted, foundcomms))
        gc.collect()
        for com in comms:
            gc.collect()

            meansim, quality = self.evaluatecommunity(com, G)

            communities.append((
                meansim,  # mean similarity
                quality,  # quality of the group
                com  # list of wordform ids
            ))

        return communities

    def heaviestedge(self, G: nx.Graph):
        u, v, w = max(G.edges(data="weight"), key=itemgetter(2))
        return (u, v)

    def evaluatecommunity(self, com, G):
        """
        Calculates the average similarity between all words in a community com of the graph G, as well as the quality
        of the community. The quality is defined as the number of edges in the community divided by the number of
        possible edges between the nodes of this community.

        :param com: The community to be evaluated.
        :param G: The graph that the community is part of.
        :return: The average similarity and the quality as tuple.
        """
        SG = nx.Graph(G.subgraph(com))
        if len(com) > 1:
            possible = (len(com) * (len(com) - 1)) / 2
            actual = SG.size()
            quality = actual / possible
            meansim = SG.size(weight='weight') / possible
        else:
            meansim = 1.0
            quality = 1.0

        return meansim, quality

    def assignRepresentative(self, communities):
        """
        Assigns a lemma to each of the given communities. The lemma is determined by choosing the word with the highest
        similarity to all other words in the community. If more than one word fulfills this condition, the one with a
        character length closest to the mean character length of the community is taken. If, again, more than one
        candidate exists, one is chosen at random.

        :param communities: A list of communities, each community being a tuple with the following elements: (meansim,
        quality, list of the indices of the words in the community).
        :return: The community tuples, supplemented by a fourth element (at index 0), containing the lemma.
        """

        for ci in range(len(communities)):
            wordsidxs = communities[ci][2]
            if len(wordsidxs) == 1:
                communities[ci] = (self.forms[wordsidxs[0]], communities[ci][0], communities[ci][1], wordsidxs)
                continue
            SG = nx.Graph(nx.subgraph(self.G, wordsidxs))
            wordsims = [(SG.degree(n, weight='weight') / SG.degree(n)) * (len(wordsidxs) -1) for n in wordsidxs]
            maxsim = max(wordsims)
            maxnodes = [i for i in range(len(wordsims)) if wordsims[i] == maxsim]
            if len(maxnodes) == 1:
                lemma = self.forms[wordsidxs[maxnodes[0]]]
            else:
                words = [self.forms[w] for w in wordsidxs]
                meanlength = sum(map(len, words)) / len(words)

                # https://stackoverflow.com/a/9706105/14393183
                lemma = words[min(range(len(maxnodes)),
                                     key=lambda i: abs(len(self.forms[wordsidxs[maxnodes[i]]]) - meanlength))]
            communities[ci] = (lemma, communities[ci][0], communities[ci][1], wordsidxs)
        return communities

    # Todo
    def assignLemmata(self):
        for com in range(len(self.communities)):
            for word in self.communities[com][3]:
                self.lemmas[word] = self.communities[com][0]

    # Todo
    def finetuneWordgroups(self):
        for com in self.communities:
            if com[1] >= 0.3:
                self.wordgroups.append((len(self.communities) - 1, com[3]))

    def date(self):
        return self.dateobj.now().strftime('%Y-%m-%d %X')

    def printstats(self, export=False):
        pass

    def lemma(self, word):
        pass

    def lemmagroup(self, word1, word2):
        pass
