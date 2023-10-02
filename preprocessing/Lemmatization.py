import datetime
import time
import gc
import os
import pickle
import shutil
import psutil
import copy
import math
import re
import networkx as nx
from multiprocessing import Process, Queue, Manager
from operator import itemgetter
from tools.lang import newu, constructSMfromDict, isNumber
from tools.rw import saveDictAsJson, readDictFromJson, readFromCSV, writeToCSV


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

        self.lonewords = []
        self.dirpath = dirpath
        self.csvpath = os.path.join(dirpath, 'sims-all.csv')
        self.info = {}
        self.name = name
        self.state = -1
        self.forms = None
        self.lemmas = None
        self.G = nx.Graph()
        self.communities = []
        self.wordgroups = []
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

        print(self.date() + ' PL-' + self.name + f': Loading model...')
        print(self.date() + ' PL-' + self.name + f': ...loading words...')
        if loadpath is not None:
            shutil.copy(os.path.join(loadpath, 'words.json'), os.path.join(self.dirpath, 'words.json'))
        words = readDictFromJson(os.path.join(self.dirpath, 'words.json'))
        self.forms = words['forms']
        if full:
            self.lemmas = words['lemmas']
        else:
            self.lemmas = [None for i in range(len(self.forms))]
        del words

        print(self.date() + ' PL-' + self.name + f': ...loading model infos...')
        if loadpath is not None:
            shutil.copy(os.path.join(loadpath, 'info.json'), os.path.join(self.dirpath, 'info.json'))
        self.info = readDictFromJson(os.path.join(self.dirpath, 'info.json'))

        if full:
            if loadpath is not None:
                shutil.copy(os.path.join(loadpath, 'wordgroups.json'), os.path.join(self.dirpath, 'wordgroups.json'))
                shutil.copy(os.path.join(loadpath, 'communities.json'), os.path.join(self.dirpath, 'communities.json'))
            print(self.date() + ' PL-' + self.name + f': ...importing communities...')
            self.loadcommunities(os.path.join(self.dirpath, 'communities.json'), 'com')
            print(self.date() + ' PL-' + self.name + f': ...importing word groups...')
            self.loadcommunities(os.path.join(self.dirpath, 'wordgroups.json'), 'wg')
        print(self.date() + ' PL-' + self.name + f': ...loading graph...')
        if loadpath is not None:
            shutil.copy(os.path.join(loadpath, 'graph.pickle'), os.path.join(self.dirpath, 'graph.pickle'))
        with open(os.path.join(self.dirpath, 'graph.pickle'), 'rb') as gp:
            self.G = pickle.load(gp)
        if full:
            self.state = 1
        else:
            self.state = 0
        print(self.date() + ' PL-' + self.name + f': Done.')

    def loadcommunities(self, wgpath, comtype):
        """
        Saves the given human-readable json representation of wordgroups in the internal format.
        """

        if comtype == 'wg':
            comlist = self.wordgroups
        elif comtype == 'com':
            comlist = self.communities
        else:
            raise ValueError
        communities = list(readDictFromJson(wgpath))
        for com in communities:
            words = [self.forms.index(w) for w in com['words']]
            com['words'] = words
            comlist.append(com)

    def readSims(self, csvpath, G: nx.Graph = None):
        if G is None:
            G = self.G
        no = 0
        with open(csvpath, 'r') as csv:
            nextline = csv.readline()
            while nextline is not None and nextline != '':
                sim = nextline.strip().split(',')
                if sim[0] != sim[1] and float(sim[3]) > 0:
                    G.add_edge(self.forms.index(sim[0]), self.forms.index(sim[1]), weight=float(sim[3]))
                no += 1
                nextline = csv.readline()
        return G, no

    def export(self, reexportgraph=False):
        """
        Exports the model in human-readable formats to the model's dirpath:

        * words.json: Containing the wordforms and their corresponding lemmata as lists.
        * wordgroups.json: Containing the found groups of wordforms representing one lemma, as well as their lemma, meansim and density measures.
        * graph.json: Containing the nodes and edges of the model's graph together with the edge weights.

        Additionally, the model's graph is pickled for fast loading.
        If the model isn't trained yet, but prepared, only the graph will be exported and the words.json will only
        contain the wordforms, not the corresponding lemmata.
        """

        print(self.date() + ' PL-' + self.name + f': Exporting model...')
        print(self.date() + ' PL-' + self.name + f': ...exporting words...')
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
            print(self.date() + ' PL-' + self.name + f': Model is untrained. Nothing else to export.')
            return
        if self.state == 1:
            print(self.date() + ' PL-' + self.name + f': ...exporting word groups...')
            saveDictAsJson(os.path.join(self.dirpath, 'wordgroups.json'), self.formatcommunities(self.wordgroups))
            gc.collect()
            print(self.date() + ' PL-' + self.name + f': ...exporting communities...')
            saveDictAsJson(os.path.join(self.dirpath, 'communities.json'), self.formatcommunities(self.communities))
            gc.collect()

        print(self.date() + ' PL-' + self.name + f' ...exporting model infos...')
        saveDictAsJson(os.path.join(self.dirpath, 'info.json'), self.info)
        if not os.path.exists(os.path.join(self.dirpath, 'graph.json')) or \
                not os.path.exists(os.path.join(self.dirpath, 'graph.pickle')) or \
                reexportgraph:
            print(self.date() + ' PL-' + self.name + f': ...exporting human-readable graph...')
            saveDictAsJson(os.path.join(self.dirpath, 'graph.json'),
                           nx.node_link_data(self.G, link='edges', source='w1', target='w2'))
            gc.collect()
            print(self.date() + ' PL-' + self.name + f': ...serializing graph...')
            with open(os.path.join(self.dirpath, 'graph.pickle'), 'wb') as gp:
                pickle.dump(self.G, gp, pickle.HIGHEST_PROTOCOL)
        print(self.date() + ' PL-' + self.name + f': Done.')

    def formatcommunities(self, toformat):
        """
        Returns a human-readable form of the model's wordgroups.

        :return: A json representation of the wordgroups.
        """

        communities = []
        for com in toformat:
            com = copy.copy(com)
            com['words'] = [self.forms[wid] for wid in com['words']]
            communities.append(com)
        return communities

    def prepare(self, simcsv=None, minsim=0.0, smcustomization=None, gapex=1.0, gapstart=0.5, phonetic=False, disfavor=0,
                combilists=None):
        """
        Prepare the model for training. This includes calculating similarity values for all possible pairings of the
        model's wordforms and constructing a graph representation of words (as nodes) and similarities (as edges). If a
        path to a simcsv with precalculated similarity values is supplied, calculation is skipped and the graph is
        constructed based on the given values.

        :param simcsv: Optional path to precalculated similarities used for graph construction.
        """

        print(self.date() + ' PL-' + self.name + f': Preparing model.')
        i = 0
        self.info['simconf'] = {
            'SM': smcustomization if smcustomization is not None else {},
            'gapex': gapex,
            'gapstart': gapstart,
            'phonetic': phonetic,
            'combilists': combilists if combilists is not None else {},
            'disfavor': disfavor
        }

        print(self.date() + ' PL-' + self.name + f': Adding words as nodes.')
        for word in self.forms:
            self.G.add_node(i, wf=word)
            i += 1

        print(self.date() + ' PL-' + self.name + f': Calculating similarities and adding them as edges.')

        if simcsv is None:
            self.G, simno = self.__calculateSimsAllThread(minsim=minsim,
                                                          smcustomization=smcustomization, gapex=gapex,
                                                          gapstart=gapstart,
                                                          phonetic=phonetic, disfavor=disfavor, combilists=combilists)
        else:
            self.G, simno = self.readSims(os.path.join(self.dirpath, simcsv))

        self.state = 0
        self.info['simno'] = simno
        self.info['nodes'] = self.G.number_of_nodes()
        self.info['edges'] = self.G.number_of_edges()
        print(self.date() + ' PL-' + self.name + f': Preparation finished: {self.info["nodes"]} nodes,'
                                                 f' {self.info["edges"]} edges.')

    def __calculateSimsAllThread(self, G: nx.Graph=None, forms=None, csvallpath=None, minsim=0.0,
                                 smcustomization=None, gapex=1.0, gapstart=0.5,
                                 phonetic=False,
                                 disfavor=0, combilists=None):
        """
        Calculate the similarity between all possible pairings of the words in the lemmatizers wordform list. The
        calculation is done with as many processes as are available on the machine that the programme is executed on.
        The similarities are saved as csv file in the dirpath of the lemmatizer object.
        """
        if G is None:
            G = self.G
        if forms is None:
            forms = self.forms
        if csvallpath is None:
            csvallpath = self.csvpath

        print(self.date() + ' PL-' + self.name + f': Calculating batches for calculation')
        pno = len(psutil.Process().cpu_affinity())
        batches = [(i, len(forms) - i) for i in range(len(forms))]
        full = int(((len(forms) + 1) * len(forms)) / 2)
        part = full / pno
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
        print(self.date() + ' PL-' + self.name + f': Created {pno} batches with quantity: {counts}')

        # Start threads
        print(self.date() + ' PL-' + self.name + f': Starting subprocesses.')
        tmppath = os.path.join(self.dirpath, 'tmp')
        if not os.path.exists(tmppath):
            os.mkdir(tmppath)
        processes = []
        manager = Manager()
        q = manager.Queue()
        for p in range(pno):
            csvpath = os.path.join(tmppath, str(p + 1) + '-' + str(pno) + '.csv')
            p = Process(target=self._calculateSimsBatch, name=str(p + 1) + '/' + str(pno),
                        args=(partitions[p], forms, counts[p], str(p + 1) + '/' + str(pno),
                              csvpath, q, minsim, smcustomization, gapex, gapstart, phonetic, disfavor, combilists))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        print(self.date() + ' PL-' + self.name + f': Adding edges.')
        for p in range(pno):
            gc.collect()
            edges = q.get()
            for edge in edges:
                G.add_edge(edge[0], edge[1], weight=edge[2])

        print(self.date() + ' PL-' + self.name + f': Joining similarity CSVs.')
        with open(csvallpath, 'wb') as ff:
            for p in range(1, pno + 1):
                f = os.path.join(tmppath, str(p) + '-' + str(pno) + '.csv')
                with open(f, 'rb') as bf:
                    shutil.copyfileobj(bf, ff)
                os.remove(f)
        os.rmdir(tmppath)
        return G, full

    def _calculateSimsBatch(self, idxes, words, full, name: str, csvpath, q: Queue, minsim=0.0,
                            smcustomization=None, gapex=1.0,
                            gapstart=0.5, phonetic=False, disfavor=0, combilists=None):
        if smcustomization is None:
            SM = None
        else:
            SM = constructSMfromDict(smcustomization)
        print(self.date() + ' PL-' + self.name + f'-' + name + ': Calculating similarities for word pair 1 of ' + str(
            full))
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
                        print(self.date() + ' PL-' + self.name + f'-' + name + ':... for word pair ' + str(
                            nextno) + ' of ' + str(full))
                        if nextno < step:
                            nextno = nextno * 10
                        else:
                            nextno += step
                    if len(sims) == nextwrite:
                        for s in sims:
                            csv.write(s)
                        sims = []
                        print(self.date() + ' PL-' + self.name + f'-' + name + ':... written up to ' + str(
                            no - 1) + ' to csv.')
                        gc.collect()
                    sim = newu(words[i], words[j], SM=SM, gapex=gapex, gapstart=gapstart,
                               phonetic=phonetic, disfavor=disfavor, combilists=combilists)
                    sims.append(
                        f'{words[i]},{words[j]},{sim["sim"]},{sim["normsim"]},{sim["alignx"]},{sim["aligny"]}\n')
                    if i != j and sim['normsim'] > minsim:
                        edges.append([i, j, sim['normsim']])
                    no += 1
            for s in sims:
                csv.write(s)
            te = time.time()
        print(self.date() + ' PL-' + self.name + f'-' + name + ': Done in ' + time.strftime('%H:%M:%S',
                                                                                            time.gmtime(te - ts)) + '.')
        q.put_nowait(edges)

    def train(self, algo='louvain', resolution=1, enforcedensity=True, resumecomdir=None, communities=None):
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

        If enforcedensity is True (default), then the community detection will run iteratively until either all found
        communities are fully connected, or until no new communities are found within the exiting ones.

        :param algo: The algorithm to use for the community detection.
        :param resolution: The resolution value to use when detecting communities with louvain or mod.
        :param enforcedensity: Whether or not to train iteratively, favouring fully connected communities.
        :param resumecomdir: A path to a directory containing intermediate data from a previous iterative training. If
            resumecomdir is not None, the training will be initialized with the newest data in that directory.
        """

        if resumecomdir is not None:
            communities = self._resumeIterativeTraining(resumecomdir, algo=algo, resolution=resolution)
        else:
            print(self.date() + ' PL-' + self.name + f': Training started.')
            self.communities = []
            self.wordgroups = []
            self.info['cdalgo'] = algo

            compath = os.path.join(self.dirpath, 'comms')
            if not os.path.exists(compath):
                os.mkdir(compath)

            if communities is None:
                print(self.date() + ' PL-' + self.name + f': Calculating communities using ' + algo + ' algorithm...')
                communities = self.detectCommunities(
                    self.G,
                    algo=algo,
                    resolution=resolution
                )
                saveDictAsJson(os.path.join(compath, 'basecomms.json'), {'comms': communities})
                print(self.date() + ' PL-' + self.name + f': Done.')
            if enforcedensity:
                communities = self._iterdetectCommunities(compath, algo=algo, resolution=resolution,
                                                      comcandidates=communities)

        communities = self.calculateMeansim(communities)
        self.communities.extend(self.assignRepresentative(communities))
        self.communities = self.calculateAverageWordLength()
        self.info['comno'] = len(self.communities)

        self.wordgroups = self.communities

        print(self.date() + ' PL-' + self.name + f': Assigning lemmata.')
        self.assignLemmata()

        print(self.date() + ' PL-' + self.name + f': Training finished: ' + str(
            self.info['lemmacount']) + ' unique lemmata.')

        self.state = 1

    def _resumeIterativeTraining(self, comdir, G: nx.Graph = None, algo='louvain', resolution=1):
        """
        Resume training with iterative community detection based on the intermediate data saved during iterative
        community detection in comdir. The function looks for the last iteration file, reads the data and calls
        the function _iterdetectCommunities with the corresponiding initial data.

        :param comdir: The path to the directory containing the intermediate training data.
        :param G: The graph, that all communities are subgraphs of.
        :param algo: The algorithm to use for the community detection.
        :param resolution: The resolution value to use when detecting communities with louvain or mod.
        :return: A list of found communities.
        """

        comdirname = os.path.split(comdir)[1]
        files = os.listdir(comdir)
        newcomdir = os.path.join(self.dirpath, comdirname)
        if os.path.realpath(comdir) != os.path.realpath(newcomdir):
            if not os.path.exists(newcomdir):
                os.mkdir(newcomdir)
            for f in files:
                shutil.copy(os.path.join(comdir, f), os.path.join(self.dirpath, comdirname, f))

        if len(files) == 0:
            return FileNotFoundError

        files.sort()
        comfile = files[-1]
        coms = readDictFromJson(os.path.join(comdir, comfile))

        if len(files) == 1 and comfile == 'basecomms.json':
            comcandidates = coms['comms']
            print(self.date() + ' PL-' + self.name + f': Resuming iterative training using'
                                                     f' {len(comcandidates)} base communities.')
            return self._iterdetectCommunities(newcomdir, algo=algo, resolution=resolution, comcandidates=comcandidates)

        communities = coms['previous']
        currcov = sum([len(c['words']) for c in communities])
        currit = int(os.path.splitext(comfile)[0].split('-')[-1])
        comcandidates = list(coms['new']) + list(coms['weak'])
        if G is None:
            G = self.G

        print(self.date() + ' PL-' + self.name + f': Resuming iterative training with data from iteration {currit-1}.')

        return self._iterdetectCommunities(newcomdir, G, algo, resolution, comcandidates, communities, currcov, currit)

    def detectCommunities(self, BG, algo='louvain', resolution=1) -> list[dict]:
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
        :return: A list of the found communities, each community being a dictionary with the following keys:
        meansim = mean similarity, density = density, words = list of word indices
        """
        communities = []
        isolates = list(nx.isolates(BG))
        for isol in isolates:
            communities.append({
                'meansim': 1.0,
                'density': 1.0,
                'words': [isol]
            })
        G = nx.Graph(nx.subgraph(BG, [n for n in BG.nodes if not n in isolates]))
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

            communities.append({
                'density': self.evaluatecommunity(com, G),
                'words': com
            })

        return communities

    def _iterdetectCommunities(self, comdir, G: nx.Graph=None, algo='louvain', resolution=1,
                               comcandidates: list[dict] = None, communities=None, coverage=0, it=1) -> list[dict]:
        """
        Calculates communities iteratively, starting with the comcandidates, until either all found communities are
        fully connected, or until no new communities are found within the exiting ones. To apply this function, the
        model needs to have been prepared first by the prepare() function. The training works by detecting communities
        in the given graph G using the community detection algorithm given in algo. Possible algorithms are:

        * "louvain" = Louvain algorithm, default (docs: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.louvain.louvain_communities.html)
        * "mod" = greedy modularity maximization (https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.modularity_max.greedy_modularity_communities.html)
        * "lpa" = asynchronous labelpropagation (docs: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.label_propagation.asyn_lpa_communities.html)
        * "girvan_newman" = Girvan-Newman algorithm (docs: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.centrality.girvan_newman.html)

        If louvain or mod is used, the resolution parameter will determine the resolution value used for the community
        detection.

        Use the parameters comdir, comcandidates, communities, coverage and it if the detection shall be initiated with
        data from a previously interrupted iterative community detection.

        :param comdir: Path to a directory where intermediate community data will be stored.
        :param algo: The algorithm to use for the community detection.
        :param resolution: The resolution value to use when detecting communities with louvain or mod.
        :param comcandidates: The base communities to be iteratively refined. If None, the comcandidates will be defined
            by running detectCommunities() first.
        :param communities: The communities that have already been found and should no be reiterated.
        :param coverage: The current coverage of words in the already found communities.
        :param it: The number of the next iteration.
        :return: A list of found communities.
        """
        if G is None:
            G = self.G
        if communities is None:
            communities = []
        if not os.path.exists(comdir):
            os.mkdir(comdir)

        if comcandidates is None:
            print(self.date() + ' PL-' + self.name + f': Calculating communities using ' + algo + ' algorithm...')
            comcandidates = self.detectCommunities(
                G,
                algo=algo,
                resolution=resolution
            )
            saveDictAsJson(os.path.join(comdir, 'basecomms.json'), {'comms': comcandidates})
            print(self.date() + ' PL-' + self.name + f': Done.')

        print(self.date() + ' PL-' + self.name + f': Checking density of the detected communities.')
        #coverage = 0
        #it = 1
        #coverage = currcov
        #it = currit
        weakcomms = []
        formno = G.number_of_nodes()
        while coverage != formno:
            for comi in reversed(range(len(comcandidates))):
                if comcandidates[comi]['density'] < 1:
                    weakcomms.append(comcandidates.pop(comi))
                else:
                    coverage += len(comcandidates[comi]['words'])
            saveDictAsJson(os.path.join(comdir, f'comms-{it}.json'), {
                'previous': communities,
                'new': comcandidates,
                'weak': weakcomms
            })
            communities.extend(comcandidates)
            comcandidates = []
            gc.collect()
            if coverage != formno:
                print(self.date() +
                      f' PL-' + self.name
                      + f': Covered {coverage} of {formno} wordforms in'
                      f' {len(communities)} communities. {len(weakcomms)} weak communities left.'
                      f' Starting iteration no. {str(it)}.')
            else:
                print(self.date() +
                      f' PL-' + self.name
                      + f': Finished. Covered {coverage} of {formno} wordforms in'
                      f' {len(communities)} communities. {len(weakcomms)} weak communities left.')
                break
            for wcom in weakcomms:
                newcoms = self.detectCommunities(
                    nx.Graph(nx.subgraph(G, wcom['words'])),
                    algo=algo,
                    resolution=resolution
                )
                gc.collect()
                if len(newcoms) == 1:
                    coverage += len(newcoms[0]['words'])
                    communities.extend(self.deisolateCommunity(G, newcoms[0]))
                else:
                    comcandidates.extend(newcoms)
            weakcomms = []
            it += 1
            if coverage == formno:
                print(self.date() +
                      f' PL-' + self.name + f': Finished. Covered {coverage} of {formno} wordforms in'
                      f' {len(communities)} communities. {len(weakcomms)} weak communities left.')
                break

        return communities

    def deisolateCommunity(self, BG: nx.Graph, com):
        SG = nx.Graph(nx.subgraph(BG, com['words']))
        isolates = list(nx.isolates(SG))
        if len(isolates) == 0:
            return [com]
        communities = []
        for isol in isolates:
            communities.append({
                'meansim': 1.0,
                'density': 1.0,
                'words': [isol]
            })
        if len(isolates) < SG.number_of_nodes():
            SG.remove_nodes_from(isolates)
            nodes = list(SG.nodes)
            communities.append({
                'density': self.evaluatecommunity(nodes, SG),
                'words': nodes
            })
        return communities

    # Source: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.centrality.girvan_newman.html
    def heaviestedge(self, G: nx.Graph):
        u, v, w = max(G.edges(data="weight"), key=itemgetter(2))
        return (u, v)

    def evaluatecommunity(self, com: list, G:nx.Graph = None):
        """
        Calculates the connectednes of the graph G, as well as the density of the community. The density is defined as
        the number of edges in the community divided by the number of possible edges between the nodes of the community.

        :param com: The community to be evaluated.
        :param G: The graph that the community is part of. Defaults to the model's Graph G if None is given.
        :return: The density.
        """
        if G is None:
            G = self.G
        if len(com) > 1:
            SG = nx.Graph(G.subgraph(com))
            possible = (len(com) * (len(com) - 1)) / 2
            actual = SG.size()
            density = actual / possible
        else:
            density = 1.0

        return density

    def calculateMeansim(self, communities: list[dict]=None, G: nx.Graph = None, forms=None, gapex=1.0, gapstart=0.5,
                         phonetic=False, disfavor=0, smcustomization=None, combilists=None) -> list[dict]:
        """
        Calculates the average similarity between all words in a community.
        """
        updatecoms = False
        if communities is None:
            updatecoms = True
            communities = self.communities
        if G is None:
            G = self.G
        if forms is None:
            forms = self.forms
        if smcustomization is not None:
            smcustomization= constructSMfromDict(smcustomization)

        for comdict in communities:
            com = comdict['words']
            if len(com) > 1:
                SG = nx.Graph(G.subgraph(com))
                totalsim = SG.size(weight='weight')
                possible = (len(com) * (len(com) - 1)) / 2
                actual = SG.size()
                if actual != possible:
                    degrees = nx.degree(SG)
                    for n in com:
                        if degrees[n] < len(com) - 1:
                            for n2 in com:
                                if n == n2 or SG.has_edge(n, n2):
                                    continue
                                totalsim += newu(forms[n], forms[n2], gapstart=gapstart, gapex=gapex, phonetic=phonetic,
                                                 SM=smcustomization, combilists=combilists,
                                                 disfavor=disfavor)['normsim']
                meansim = totalsim / possible
            else:
                meansim = 1.0

            comdict['meansim'] = meansim

        if updatecoms:
            self.communities = communities
        return communities

    def calculateAverageWordLength(self, communities: list[dict]=None, forms=None) -> list[dict]:
        if forms is None:
            forms = self.forms
        updatecoms = False
        if communities is None:
            communities = self.communities
            updatecoms = True
        for ci in range(len(communities)):
            wordsidxs = communities[ci]['words']
            meanlen = sum([len(forms[wi]) for wi in wordsidxs]) / len(wordsidxs)
            communities[ci]['meanlen'] = meanlen
        if updatecoms:
            self.communities = communities
        return communities

    def assignRepresentative(self, communities: list[dict], G: nx.Graph = None, forms=None) -> list[dict]:
        """
        Assigns a lemma to each of the given communities. The lemma is determined by choosing the word with the highest
        similarity to all other words in the community. If more than one word fulfills this condition, the one with a
        character length closest to the mean character length of the community is taken. If, again, more than one
        candidate exists, one is chosen at random.

        :param communities: A list of communities as dictionaries.
        :return: The community tuples, supplemented by a fourth element (at index 0), containing the lemma.
        """
        if forms is None:
            forms = self.forms
        if G is None:
            G = self.G

        for ci in range(len(communities)):
            wordsidxs = communities[ci]['words']
            if len(wordsidxs) == 1:
                communities[ci]['lemma'] = forms[wordsidxs[0]]
                continue
            SG = nx.Graph(nx.subgraph(G, wordsidxs))
            nonzerowidxs = [n for n in wordsidxs if SG.degree(n) > 0]
            if len(nonzerowidxs) == 0:
                words = [forms[w] for w in wordsidxs]
                meanlength = communities[ci].get('meanlen',  sum(map(len, words)) / len(words))
                lemma = words[min(range(len(words)),
                                  key=lambda i: abs(len(words[i]) - meanlength))]
            else:
                wordsims = [(SG.degree(n, weight='weight') / SG.degree(n)) * (len(wordsidxs) - 1) for n in nonzerowidxs]
                maxsim = max(wordsims)
                maxnodes = [i for i in range(len(wordsims)) if wordsims[i] == maxsim]
                if len(maxnodes) == 1:
                    lemma = forms[nonzerowidxs[maxnodes[0]]]
                else:
                    words = [forms[w] for w in wordsidxs]
                    meanlength = communities[ci].get('meanlen',  sum(map(len, words)) / len(words))

                    # https://stackoverflow.com/a/9706105/14393183
                    lemma = words[min(range(len(maxnodes)),
                                      key=lambda i: abs(len(forms[nonzerowidxs[maxnodes[i]]]) - meanlength))]
            communities[ci]['lemma'] = lemma
        return communities

    def assignLemmata(self):
        self.lemmas = [None for i in range(len(self.forms))]
        for com in range(len(self.wordgroups)):
            for word in self.wordgroups[com]['words']:
                self.lemmas[word] = self.wordgroups[com]['lemma']
        for idx in range(len(self.lemmas)):
            if isNumber(self.forms[idx]):
                self.lemmas[idx] = '123'
            elif self.lemmas[idx] is None:
                self.lemmas[idx] = self.forms[idx]
        self.info['lemmacount'] = len(set(self.lemmas))

    def finetuneShortWordCommunities(self, smcustomization=None, gapex=1.0, gapstart=1.0, disfavor=1, combilists=None,
                                     algo='louvain', resolution=1, simcsv=None):

        print(self.date() + ' PL-' + self.name + ': Starting finetuning of communities with short words.')
        shortnodes = []
        todel = []
        print(self.date() + ' PL-' + self.name + ': Identifying communities with short words.')
        for i in range(len(self.communities)):
            wg = self.communities[i]
            if (wg['meansim'] < 0.8 and wg['meanlen'] <= 6.0) or wg['meanlen'] <= 3.0:
                shortnodes.extend(wg['words'])
                todel.append(i)
        for j in reversed(todel):
            del self.communities[j]
        print(self.date() + ' PL-' + self.name + f': Found {len(todel)} communities with {len(shortnodes)} short words.')
        SG = nx.Graph()
        SG.add_nodes_from([i for i in range(len(shortnodes))])
        forms = [self.forms[w] for w in shortnodes]
        csvpath = os.path.join(self.dirpath, 'swsims.csv')
        print(self.date() + ' PL-' + self.name + ': Calculating phonetic similarity for short words.')
        if simcsv is None:
            SG, simno = self.__calculateSimsAllThread(G=SG, forms=forms, csvallpath=csvpath,
                                                      minsim=0.4, smcustomization=smcustomization, gapex=gapex,
                                                      gapstart=gapstart,
                                                      phonetic=True, disfavor=disfavor, combilists=combilists)
        else:
            SG, simno = self.readSims(os.path.join(self.dirpath, simcsv), G=SG)
        SG = self.refilterEdges(0.5, 1, G=SG, forms=forms)
        print(self.date() + ' PL-' + self.name + ': Starting iterative community detection on short words.')
        communities = self._iterdetectCommunities(os.path.join(self.dirpath, 'swcomms'), G=SG, algo=algo,
                                                  resolution=resolution)
        print(self.date() + ' PL-' + self.name + ': Calculating average word length of the new communities.')
        communities = self.calculateAverageWordLength(communities, forms=forms)
        print(self.date() + ' PL-' + self.name + ': Assigning representatives of the new communities.')
        communities = self.assignRepresentative(communities, G=SG, forms=forms)
        print(self.date() + ' PL-' + self.name + ': Calculating mean similarity of the new communities.')
        communities = self.calculateMeansim(communities, G=SG, forms=forms, smcustomization=smcustomization,
                                            phonetic=True, gapex=gapex, gapstart=gapstart, disfavor=disfavor,
                                            combilists=combilists)
        print(self.date() + ' PL-' + self.name + ': Fitting short word indexes to the overall indexes.')
        communities = self.translateComIdxes(communities, shortnodes)
        self.communities.extend(communities)

        print(self.date() + ' PL-' + self.name + ': Finished fine-tuning.')

    def translateComIdxes(self, communities:list[dict], origidxes) -> list[dict]:
        for com in communities:
            gc.collect()
            words = []
            for widx in com['words']:
                words.append(origidxes[widx])
            com['words'] = words
        return communities

    def finetuneWordgroups(self):
        self.wordgroups = []
        self.lonewords = []
        for com in self.communities:
            if len(com['words']) == 1:
                self.lonewords.extend(com['words'])
                continue
            if com['meansim'] >= 0.7 and com['meanlen'] > 3.0:
                self.wordgroups.append(com)
            elif com['meansim'] == 1.0 and com['meanlen'] <= 3.0:
                self.wordgroups.append(com)
        self.info['wgno'] = len(self.wordgroups)
        self.info['lwno'] = len(self.lonewords)

    def joinWordgroups(self, lemmamap:dict, lemmata:list, exportdocu=True):
        """
        Joins wordgroups according to an external reference containing a mapping for wordforms and their corresponding
        lemmata. Wordgroups are joined, if they contain wordforms of the same lemma. If a wordgroup contains forms of
        more than one lemma, it is not joined.

        :param lemmamap: A dictionary mapping wordforms as keys to the corresponding lemmata as string-values.
        :param lemmata: A list containing all lemmata.
        :param exportdocu: Whether or not to export a documentation about the old and newly joined wordgroups.
        """

        wgmap = [set() for i in range(len(lemmata))]
        forms = [[] for i in range(len(lemmata))]

        # Map wordforms and wordgroups to lemmata
        for i in range(len(self.wordgroups)):
            wg = self.wordgroups[i]
            for wi in wg['words']:
                w = self.forms[wi]
                if w in lemmamap.keys():
                    lemidx = lemmata.index(lemmamap[w])
                    wgmap[lemidx].add(i)
                    forms[lemidx].append(w)
        # Find lemmata present in more than one wordgroup
        joincandidates = [(wgmap[i], forms[i], lemmata[i]) for i in range(len(wgmap)) if len(wgmap[i]) > 1]

        # Check for ambigue wordgroups containing forms of more than one lemma. Remove all ambigue wordgroups from the
        # candidates
        impactedwgs = []
        for jc in joincandidates:
            impactedwgs.extend(jc[0])
        counts = {}
        for i in impactedwgs:
            counts[i] = counts.get(i, 0) + 1
        ambiguewgs = [k for k in counts.keys() if counts[k] > 1]
        discardcandidates = []
        for i in range(len(joincandidates)):
            discardwgs = []
            for wg in joincandidates[i][0]:
                if wg in ambiguewgs:
                    discardwgs.append(wg)
            if len(joincandidates[i][0]) - len(discardwgs) < 2:
                discardcandidates.append(i)
                continue
            newwgs = joincandidates[i][0]
            for wg in discardwgs:
                newwgs.remove(wg)
            joincandidates[i] = (newwgs, joincandidates[i][1], joincandidates[i][2])
        for dis in sorted(discardcandidates, reverse=True):
            del joincandidates[dis]
        for jc in joincandidates:
            impactedwgs.extend(jc[0])

        # Join wordgroups
        wgno = 0
        nwgs = []
        discard = []
        docu = []
        for d in joincandidates:
            impactedwgs.extend(d[0])
            wgno += len(d[0])
            words = []
            rep = ''
            wglen = 0
            simtotal = 0
            wcount = 0
            for wg in d[0]:
                discard.append(wg)
                wlen = len(self.wordgroups[wg]['words'])
                wcount += wlen
                simtotal += self.wordgroups[wg]['meansim']
                words.extend(self.wordgroups[wg]['words'])
                if wlen > wglen:
                    wglen = wlen
                    rep = self.wordgroups[wg]['lemma']
            joinedwg = {
                'meansim': simtotal/len(d[0]),
                'density': self.evaluatecommunity(words),
                'words': words,
                'lemma': rep
            }
            nwgs.append(joinedwg)
            docu.append({
                'lemma': d[2],
                'old': self.formatcommunities([self.wordgroups[wgi] for wgi in d[0]]),
                'new': self.formatcommunities([joinedwg])[0]
            })
        nwgs = self.calculateAverageWordLength(nwgs)

        # Remove old wordgroups and add new, joined wordgroups
        for dis in sorted(discard, reverse=True):
            del self.wordgroups[dis]
        self.wordgroups.extend(nwgs)
        self.info['wgno'] = len(self.wordgroups)
        print(self.date() + ' PL-' + self.name +
              f': Joined {len(impactedwgs)} wordgroups into {len(nwgs)} new wordgroups.')

        if exportdocu is not None:
            print(self.date() + ' PL-' + self.name + f': Exporting documentation of the performed joins.')
            saveDictAsJson(os.path.join(self.dirpath, 'joinedWGs.json'), docu)

    def joinLoneWords(self, lemmamap:dict, lemmata:list, exportdocu=True):
        """
        Joins wordgroups according to an external reference containing a mapping for wordforms and their corresponding
        lemmata. Wordgroups are joined, if they contain wordforms of the same lemma. If a wordgroup contains forms of
        more than one lemma, it is not joined.

        :param lemmamap: A dictionary mapping wordforms as keys to the corresponding lemmata as string-values.
        :param lemmata: A list containing all lemmata.
        :param exportdocu: Whether or not to export a documentation about the old and newly joined wordgroups.
        """

        wmap = [set() for i in range(len(lemmata))]

        # Map word forms to lemmata
        for wi in self.lonewords:
            w = self.forms[wi]
            if w in lemmamap.keys():
                lemidx = lemmata.index(lemmamap[w])
                wmap[lemidx].add(wi)

        # Find lemmata with two or more word forms in the model
        joincandidates = [(list(wmap[i]), lemmata[i]) for i in range(len(wmap)) if len(wmap[i]) > 1]

        # Join words to new wordgroup
        wordforms = []
        nwgs = []
        for d in joincandidates:
            wordforms.extend(d[0])
            joinedwg = {
                'density': self.evaluatecommunity(d[0]),
                'words': d[0]
            }
            nwgs.append(joinedwg)
        nwgs = self.calculateMeansim(nwgs)
        nwgs = self.calculateAverageWordLength(nwgs)
        nwgs = self.assignRepresentative(nwgs)

        # Remove words from lonewords list and add new, joined wordgroups
        for dis in sorted(wordforms, reverse=True):
            self.lonewords.remove(dis)
        self.wordgroups.extend(nwgs)
        self.info['wgno'] = len(self.wordgroups)
        self.info['lwno'] = len(self.lonewords)
        print(self.date() + ' PL-' + self.name +
              f': Joined {len(wordforms)} words into {len(nwgs)} new wordgroups.')

        if exportdocu is not None:
            print(self.date() + ' PL-' + self.name + f': Exporting documentation of the performed joins.')
            saveDictAsJson(os.path.join(self.dirpath, 'joinedLWs.json'), self.formatcommunities(nwgs))

    def refilterEdges(self, minsim, minlength, G:nx.Graph=None, forms=None, updateinfo = True):
        if G is None:
            G = self.G
        if forms is None:
            forms = self.forms

        toolow = 0
        tooshort = 0
        discard = []
        for (u, v, c) in G.edges.data('weight'):
            if c < minsim:
                discard.append((u, v))
                toolow += 1
            elif minlength > 1 and (len(forms[u]) < minlength or len(forms[v]) < minlength) and c != 1.0:
                discard.append((u, v))
                tooshort += 1
        G.remove_edges_from(discard)
        if updateinfo:
            self.info['edges'] = G.size()
            self.info['minsim'] = minsim
            self.info['minlength'] = minlength
            self.info['length_discarded_edges'] = tooshort
        print(self.date() + ' PL-' + self.name +
              f': Removed {toolow} edges with a similarity lower than {minsim} and'
              f' {tooshort} edges leading from or to a word too short for pseudolemmatization.'
              f' {G.size()} edges left in graph.')
        return G

    def date(self):
        return self.dateobj.now().strftime('%Y-%m-%d %X')

    def lemmatizeTextFromCSV(self, innput, output):
        csv = readFromCSV(innput)
        for i in range(len(csv['lines'])):
            l = csv['lines'][i][0]
            tokens = l.split()
            leml = []
            for token in tokens:
                try:
                    match = re.fullmatch(r'((#SEND#|#CSTART#|#lb#|#INSTART#|#INEND#|\$seg[^$]+\$|\s)*)'
                                         r'([a-z]+)((#SEND#|#CSTART#|#lb#|#INSTART#|#INEND#|\$seg[^$]+\$|\s)*)', token)
                    if match is None:
                        lem = token
                    else:
                        lem = str(self.lemmas[self.forms.index(match[3])])
                        lem = match[1] + lem + match[4]
                except ValueError:
                    lem = token
                leml.append(lem)
            csv['lines'][i][0] = ' '.join(leml)
        writeToCSV(output, csv['lines'], header=csv['header'])
