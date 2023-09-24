import os
import shutil
import tempfile
from multiprocessing import Queue, Manager, Process

import psutil

from tools.rw import saveDictAsJson, readDictFromJson, readFromTxt, readJsonFromJsonlines
from trd.wrapping import runblastFull, runpassimFull, runtextpairFull


def extractReusePairs(indir, outdir):
    segments = {}
    infiles = os.listdir(indir)

    opensegments = {}

    for f in infiles:
        if not f.endswith('.txt'):
            continue
        fid = f[:-4]
        with open(os.path.join(indir, f), 'r', encoding='utf-8') as infile:
            with open(os.path.join(outdir, f), 'w', encoding='utf-8') as outfile:
                towrite = ''
                nextchar = infile.read(1)
                offset = 0
                while nextchar != '':
                    if nextchar == '$':
                        outfile.write(towrite)
                        towrite = ''
                        segmeta = ''
                        nextchar = infile.read(1)
                        while nextchar != '$':
                            segmeta += nextchar
                            nextchar = infile.read(1)
                        segmeta = segmeta.split('_')
                        if segmeta[0] == 'segstart':
                            segments[segmeta[1]] = segmeta[1:]
                            opensegments[segmeta[1]] = {
                                'start': offset,
                                'text': ''
                            }
                        else:
                            segments[segmeta[1]].append(fid)
                            segments[segmeta[1]].append(opensegments[segmeta[1]]['start'])
                            segments[segmeta[1]].append(offset - 1)
                            segments[segmeta[1]].append(opensegments[segmeta[1]]['text'])
                            del opensegments[segmeta[1]]
                        nextchar = infile.read(1)
                    else:
                        offset += 1
                        towrite += nextchar
                        if len(opensegments) > 0:
                            for k in opensegments:
                                opensegments[k]['text'] = opensegments[k]['text'] + nextchar
                        nextchar = infile.read(1)
                outfile.write(towrite)

    matches = []
    for segid in segments.keys():
        seg = segments[segid]
        if len(seg) > 5:
            for sourceid in seg[2].split('&'):
                source = segments[sourceid]
                match = {
                    'type': seg[1],
                    'source': source[-4],
                    'target': seg[-4],
                    'start_s': source[-3],
                    'end_s': source[-2],
                    'start_t': seg[-3],
                    'end_t': seg[-2],
                    'text_s': source[-1],
                    'text_t': seg[-1]
                }
                matches.append(match)

    saveDictAsJson(os.path.join(outdir, 'goldmatches.json'), {'matches': matches})


def parseGoldmatches(filepath):
    goldmatchdict = readDictFromJson(filepath)
    literal = []
    weak_literal = []
    non_literal = []
    for match in goldmatchdict['matches']:
        parsedmatch = [
            (match['source'], match['start_s'], match['end_s'],),
            (match['target'], match['start_t'], match['end_t'],)
        ]
        if match['type'] == 'literal':
            literal.append(parsedmatch)
        elif match['type'] == 'weak-literal':
            weak_literal.append(parsedmatch)
        else:
            non_literal.append(parsedmatch)
    return literal, weak_literal, non_literal


def identicalPassages(match1: list[tuple[str, int, int]], match2: list[tuple[str, int, int]]) -> bool:
    """
    Match needs to be a list containing triples, each triple indicating an identifier for the text, the start index and
    end index of the match. The matches are seen as identical, if at least two pairs of triplets can be found, such that
    the pair's triplets belong to a different match respectively and each pair belongs to a different text, but the
    triplets in each pair still belong to the same text and their indices overlap.

    :param match1: A list of matching passages in triple.
    :param match2: A list of matching passages in triple.
    :return: True, if both matches denote the same passage. False otherwise.
    """

    matchingpairs = dict()
    for m1 in match1:
        docid = m1[0]
        for m2 in match2:
            if m2[0] == docid and (
                    (m1[1] <= m2[1] <= m1[2])
                    or
                    (m1[1] <= m2[2] <= m1[2])
                    or
                    (m2[1] <= m1[1] <= m2[2])
                    or
                    (m2[1] <= m1[2] <= m2[2])):
                mplist = matchingpairs.get(docid, [])
                mplist.append((m1, m2))
                matchingpairs[docid] = mplist

    if len(matchingpairs.keys()) > 1:
        return True

    return False


def evaluateRecall(testmatches, goldmatches):
    foundgms = []
    # Evaluate the recall of goldmatches
    for gmi in range(len(goldmatches)):
        gm = goldmatches[gmi]
        for tmi in range(len(testmatches)):
            tm = testmatches[tmi]
            if identicalPassages(gm, tm):
                foundgms.append((gmi, tm))
                break
    return foundgms


def findBestRun(runinfos):
    """
    Find the best run out of the runs with the given runinfos. The best run is the run with the highest 'goldmatch_no',
    the lowest overall 'testmatch_no' and the lowest 'time' (prioritized in that order).

    :param runinfos: A list of dictionaries. Each dict can contain arbitrary information about the test run, but must at
    least have the keys 'goldmatch_no', 'testmatch_no' and 'time'.
    :return: The dictionary of the best run.
    """

    highmatchruns = []
    for ri in range(len(runinfos)):
        r = runinfos[ri]
        recall = r['goldmatch_no']
        if len(highmatchruns) == 0 or recall > runinfos[highmatchruns[0]]['goldmatch_no']:
            highmatchruns = [ri]
        elif recall == runinfos[highmatchruns[0]]['goldmatch_no']:
            highmatchruns.append(ri)

    if len(highmatchruns) == 1:
        bestrun = runinfos[highmatchruns[0]]
    else:
        lownoiseruns = []
        mcount = runinfos[highmatchruns[0]]['testmatch_no']
        for r in highmatchruns:
            if runinfos[r]['testmatch_no'] < mcount:
                lownoiseruns = [r]
            elif runinfos[r]['testmatch_no'] == mcount:
                lownoiseruns.append(r)

        if len(lownoiseruns) == 1:
            bestrun = runinfos[lownoiseruns[0]]
        else:
            fastruns = []
            tme = runinfos[lownoiseruns[0]]['time']
            for r in lownoiseruns:
                if runinfos[r]['time'] < tme:
                    fastruns = [r]
                elif runinfos[r]['time'] == tme:
                    fastruns.append(r)
            bestrun = runinfos[fastruns[0]]

    return dict(bestrun)


def humanReadableMatchinfo(matchpairs, textdir, goldmatches):
    """
    Returns a list of human-readable matchpairs, each containing the text passages of the goldmatches as well as the
    corresponding passages of the testmatches.

    :param matchpairs: A list of tuples, each tuple containing the index of a goldmatch and a corresponding testmatch.
    :param textdir: The path to the directory containing the test texts.
    :param goldmatches: A list of all goldmatches.
    :return: A list of pretty-formatted matchpairs.
    """

    texts = dict()
    for f in os.listdir(textdir):
        txt = readFromTxt(os.path.join(textdir, f))
        title = f.split('.')[0]
        texts[title] = txt

    matchinfo = []
    for mp in matchpairs:
        gmlist = []
        for m in goldmatches[mp[0]]:
            gmlist.append({
                'title': m[0],
                'text': texts[m[0]][m[1]:m[2] + 1]
            })

        tmlist = []
        for m in mp[1]:
            tmlist.append({
                'title': m[0],
                'text': texts[m[0]][m[1]:m[2] + 1]
            })

        minfo = {
            'gm': gmlist,
            'tm': tmlist
        }
        matchinfo.append(minfo)

    return matchinfo


def _multiprocessEvalRuns(golddir, outdir, goldmatches, target, runparams, paramnames, cpudivisor=1):
    runinfos = []
    runnototal = len(runparams)

    runno = 0
    cpuno = int(len(psutil.Process().cpu_affinity())/cpudivisor)
    if cpuno ==0:
        cpuno = 1
    while runno < runnototal:
        pno = 0
        manager = Manager()
        q = manager.Queue()
        processes = []
        while pno < cpuno and runno < runnototal:
            prms = runparams[runno]
            baseargs = [runno, runnototal, golddir, outdir, goldmatches, q]
            paramargs = [prms[name] for name in paramnames]
            args = tuple(baseargs + paramargs)
            p = Process(target=target, name=str(runno),
                        args=args)
            processes.append(p)
            p.start()
            pno += 1
            runno += 1
        for proc in processes:
            proc.join()
        while q.qsize() != 0:
            info = q.get()
            runinfos.append(info)
    return runinfos


def evaluateBlast(outdir: str, golddir: str, goldmatches: list[tuple[str, int, int]], e_value: list[float],
                  word_size: list[int], evalinfo: dict = None):
    """
    Runs BLAST with each possible combination of the given parameters and returns the results of evaluating each run.
    Each parameter must be given as a list of parameter values.

    :param outdir: The path to the directory in which the results are to be stored.
    :param golddir: The path in which the golddata as .txt files can be found.
    :param goldmatches: A list of matches in the golddata, each match being a tuple containing the document identifier
        (i.e. the document's without the file extension), the start index and finally the end index of the match.
    :param e_value: The e_value as float. Influences, how long or short parallels must be to be recognized as match.
    :param word_size: The number of characters constituting a 'word' for BLAST. Must be between 2 and 7.
    :param evalinfo: Optional. A dictionary containing any kind of additional information about the evaluation setting.
    :return: The configuration of the best run.
    """

    print()
    print('### Evaluating BLAST')
    print()

    runparams = []
    for e in e_value:
        for ws in word_size:
            runparams.append({
                'e_value': e,
                'word_size': ws
            })
    paramnames = ['e_value', 'word_size']
    target = _blastEvalRun

    runinfos = _multiprocessEvalRuns(golddir, outdir, goldmatches, target, runparams, paramnames)
    runinfos.sort(key=lambda r: r['run_no'])

    overallinfo = {
        'tool': 'BLAST',
        'runno': len(runparams),
        'test_e_value': e_value,
        'test_word_size': word_size,
        'all_goldmatches': len(goldmatches),
        'note': evalinfo if evalinfo is not None else dict()
    }

    print('Finding best run.')
    overallinfo['run_details'] = runinfos
    bestrun = findBestRun(runinfos)
    bestrun['goldmatches'] = humanReadableMatchinfo(bestrun['goldmatches'], golddir, goldmatches)
    overallinfo['bestrun'] = bestrun
    saveDictAsJson(os.path.join(outdir, 'evaluation.json'), overallinfo)

    print('Evaluation finished.')
    print(f'Best run uses an e_value of {bestrun["config"]["e_value"]} and a word_size '
          f'of {bestrun["config"]["word_size"]}.')
    print(f'The run found {bestrun["goldmatch_no"]} of {len(goldmatches)} goldmatches '
          f'and {bestrun["testmatch_no"]} matches in total.')
    print()

    return bestrun["config"]


def _blastEvalRun(runno, runnototal, golddir, outdir, goldmatches, q: Queue, e_value, word_size):
    print()
    print(f'### Performing run {runno + 1} of {runnototal}')

    resultdir = os.path.join(outdir, 'run_' + str(runno))
    if os.path.exists(resultdir):
        shutil.rmtree(resultdir)
    os.mkdir(resultdir)

    with tempfile.TemporaryDirectory() as tmpdir:
        info = runblastFull(golddir, tmpdir, resultdir, e_value, word_size)
        info['resultdir'] = resultdir
        info['run_no'] = runno

        testmatches = []

        # Extract matches found by the tool
        clusterfiles = os.listdir(resultdir)
        for cf in clusterfiles:
            cname = os.path.basename(cf)
            clusters = readDictFromJson(os.path.join(resultdir, cname))
            for ck in clusters.keys():
                c = clusters[ck]
                match = []
                for m in c['hits']:
                    match.append((
                        m['doc_id'],
                        m['original_indices'][0],
                        m['original_indices'][1]
                    ))
                testmatches.append(match)

    # Evaluate the recall of goldmatches
    literalgms = evaluateRecall(testmatches, goldmatches)

    info['goldmatch_no'] = len(literalgms)
    info['goldmatches'] = literalgms
    info['testmatch_no'] = len(testmatches)

    saveDictAsJson(os.path.join(resultdir, 'info.json'), info)
    q.put(info)


def evaluatePassim(outdir: str, golddir: str, goldmatches: list[tuple[str, int, int]], n: list[int],
                   min_align: list[int], beam: list[int], pcopy: list[float], all_pairs: list[bool],
                   maxDF: list[int] = None, min_match: list[int] = None, evalinfo: dict = None):
    """
    Runs Passim with each possible combination of the given parameters and returns the results of evaluating each run.
    Each parameter must be given as a list of parameter values.

    :param outdir: The path to the directory in which the results are to be stored.
    :param golddir: The path in which the golddata as .txt files can be found.
    :param goldmatches: A list of matches in the golddata, each match being a tuple containing the document identifier
        (i.e. the document's without the file extension), the start index and finally the end index of the match.
    :param n: Number of characters (n) in each character n-gram used for TRD.
    :param min_align: Minimum number of characters in an aligned passage to be considered a match.
    :param beam: Number of optimal alignments kept in memory during beam search.
    :param pcopy: Probability of a character being copied from the source to the target passage.
    :param all_pairs: Whether to compute alignments for all pairs.
    :param maxDF: Upper limit on ngram document frequency to be considered during detection. Default: 100.
    :param min_match: Minimum number of n-gram matches between documents for the documents to be compared. Default: 5.
    :param evalinfo: Optional. A dictionary containing any kind of additional information about the evaluation setting.
    :return: The configuration of the best run.
    """

    print()
    print('### Evaluating Passim')
    print()

    if min_match is None:
        min_match = [5]
    if maxDF is None:
        maxDF = [100]

    runparams = []
    for nn in n:
        for ma in min_align:
            for b in beam:
                for p in pcopy:
                    for ap in all_pairs:
                        for mm in min_match:
                            for mdf in maxDF:
                                runparams.append({
                                    'n': nn,
                                    'min_align': ma,
                                    'beam': b,
                                    'pcopy': p,
                                    'all_pairs': ap,
                                    'min_match': mm,
                                    'maxDF': mdf
                                })

    paramnames = ['n', 'min_align', 'beam', 'pcopy', 'all_pairs', 'min_match', 'maxDF']
    target = _passimEvalRun

    runinfos = _multiprocessEvalRuns(golddir, outdir, goldmatches, target, runparams, paramnames, cpudivisor=2)
    runinfos.sort(key=lambda r: r['run_no'])

    overallinfo = {
        'tool': 'Passim',
        'runno': len(runparams),
        'test_n': n,
        'test_min_align': min_align,
        'test_beam': beam,
        'test_pcopy': pcopy,
        'test_all_pairs': all_pairs,
        'test_min_match': min_match,
        'test_maxDF': maxDF,
        'all_goldmatches': len(goldmatches),
        'note': evalinfo if evalinfo is not None else dict()
    }

    print('Finding best run.')
    overallinfo['run_details'] = runinfos
    bestrun = findBestRun(runinfos)
    bestrun['goldmatches'] = humanReadableMatchinfo(bestrun['goldmatches'], golddir, goldmatches)
    overallinfo['bestrun'] = bestrun
    saveDictAsJson(os.path.join(outdir, 'evaluation.json'), overallinfo)

    print('Evaluation finished.')
    print(f'Best run uses an n of {bestrun["config"]["n"]}, a min_align of {bestrun["config"]["min_align"]}, a beam '
          f'size of {bestrun["config"]["beam"]}, a pcopy of {bestrun["config"]["pcopy"]}, a maxDF of'
          f' {bestrun["config"]["maxDF"]}, a min_match of {bestrun["config"]["min_match"]} and'
          f' {"no " if not bestrun["config"]["all_pairs"] else ""}all_pairs.')
    print(f'The run found {bestrun["goldmatch_no"]} of {len(goldmatches)} goldmatches '
          f'and {bestrun["testmatch_no"]} matches in total.')
    print()

    return bestrun["config"]


def _passimEvalRun(runno, runnototal, golddir, outdir, goldmatches, q: Queue, n, min_align, beam,
                   pcopy, all_pairs, min_match, maxDF):
    print()
    print(f'### Performing run {runno + 1} of {runnototal}: {n}, {min_align}, {beam}, {pcopy}, {all_pairs},'
          f' {min_match}, {maxDF}')

    resultdir = os.path.join(outdir, 'run_' + str(runno))
    if os.path.exists(resultdir):
        shutil.rmtree(resultdir)
    os.mkdir(resultdir)

    with tempfile.TemporaryDirectory() as tmpdir:
        info = runpassimFull(golddir, tmpdir, resultdir, n=n, min_align=min_align, beam=beam,
                             floating_ngrams=True, pcopy=pcopy, all_pairs=all_pairs,
                             min_match=min_match, maxDF=maxDF)
        info['resultdir'] = resultdir
        info['run_no'] = runno

        testmatches = []
        cpath = os.path.join(resultdir, 'out.json')

    # Extract matches found by the tool
    clusters = readDictFromJson(cpath)
    for c in clusters['clusters']:
        match = []
        for m in c['matches']:
            match.append((
                m['id'],
                m['begin'],
                m['end']
            ))
        testmatches.append(match)

    # Evaluate the recall of goldmatches
    gms = evaluateRecall(testmatches, goldmatches)

    info['goldmatch_no'] = len(gms)
    info['goldmatches'] = gms
    info['testmatch_no'] = len(testmatches)

    saveDictAsJson(os.path.join(resultdir, 'info.json'), info)
    q.put(info)


def evaluateTextpair(outdir: str, golddir: str, goldmatches: list[tuple[str, int, int]], ngram: list[int],
                     gap: list[int], matching_window_size: list[int], minimum_matching_ngrams_in_docs: list[int],
                     minimum_matching_ngrams_in_window: list[int], max_gap: list[int],
                     minimum_matching_ngrams: list[int], store_banalities=None, evalinfo: dict = None):
    """
    Runs TextPAIR with each possible combination of the given parameters and returns the results of evaluating each run.
    Each parameter (except store_banalities) must be given as a list of parameter values.

    :param outdir: The path to the directory in which the results are to be stored.
    :param golddir: The path in which the golddata as .txt files can be found.
    :param goldmatches: A list of matches in the golddata, each match being a tuple containing the document identifier
        (i.e. the document's without the file extension), the start index and finally the end index of the match.
    :param ngram: Number of tokens in an n-gram.
    :param gap: Size of gap allowed in n-grams.
    :param matching_window_size: Size of ngram window to be initially evaluated in the sequence aligner.
    :param minimum_matching_ngrams_in_docs: Minimum number of shared ngrams between docs to start a comparison.
    :param minimum_matching_ngrams_in_window: Minimum number of matching ngrams in ngram window
    :param max_gap: Maximum gap authorized between matching ngrams
    :param minimum_matching_ngrams: Minimum number of matching ngrams to constitute a match
    :param store_banalities: Whether to store banalities in a different file than all other alignments. Setting
        this to True might impact the evaluation results.
    :param evalinfo: Optional. A dictionary containing any kind of additional information about the evaluation setting.
    :return: The configuration of the best run.
    """

    print()
    print('### Evaluating TextPAIR')
    print()

    if store_banalities is None:
        store_banalities = [False]

    runparams = []
    for n in ngram:
        for g in gap:
            for mws in matching_window_size:
                for mmnid in minimum_matching_ngrams_in_docs:
                    for mmniw in minimum_matching_ngrams_in_window:
                        for mg in max_gap:
                            for mmn in minimum_matching_ngrams:
                                for sb in store_banalities:
                                    runparams.append({
                                        'ngram': n,
                                        'gap': g,
                                        'matching_window_size': mws,
                                        'minimum_matching_ngrams_in_docs': mmnid,
                                        'minimum_matching_ngrams_in_window': mmniw,
                                        'max_gap': mg,
                                        'minimum_matching_ngrams': mmn,
                                        'store_banalities': sb
                                    })

    paramnames = ['ngram', 'gap', 'matching_window_size', 'minimum_matching_ngrams_in_docs',
                  'minimum_matching_ngrams_in_window', 'max_gap', 'minimum_matching_ngrams', 'store_banalities']
    target = _textpairEvalRun

    runinfos = _multiprocessEvalRuns(golddir, outdir, goldmatches, target, runparams, paramnames)
    runinfos.sort(key=lambda r: r['run_no'])

    overallinfo = {
        'test_tool': 'TextPAIR',
        'test_runno': len(runparams),
        'test_ngram': ngram,
        'test_gap': gap,
        'test_matching_window_size': matching_window_size,
        'test_minimum_matching_ngrams_in_docs': minimum_matching_ngrams_in_docs,
        'test_minimum_matching_ngrams_in_window': minimum_matching_ngrams_in_window,
        'test_max_gap': max_gap,
        'test_minimum_matching_ngrams': minimum_matching_ngrams,
        'all_goldmatches': len(goldmatches),
        'note': evalinfo if evalinfo is not None else dict()
    }

    print('Finding best run.')
    overallinfo['run_details'] = runinfos
    bestrun = findBestRun(runinfos)
    bestrun['goldmatches'] = humanReadableMatchinfo(bestrun['goldmatches'], golddir, goldmatches)
    overallinfo['bestrun'] = bestrun
    saveDictAsJson(os.path.join(outdir, 'evaluation.json'), overallinfo)

    print('Evaluation finished.')
    print(f'Best run uses an ngram size of {bestrun["config"]["ngram"]}, a gap size of {bestrun["config"]["gap"]}, '
          f'a matching_window_size of {bestrun["config"]["matching_window_size"]}, a minimum_matching_ngrams_in_docs '
          f'of {bestrun["config"]["minimum_matching_ngrams_in_docs"]}, a minimum_matching_ngrams_in_window of '
          f'{bestrun["config"]["minimum_matching_ngrams_in_window"]}, a max_gap of {bestrun["config"]["max_gap"]}, and '
          f'a minimum_matching_ngrams of {bestrun["config"]["minimum_matching_ngrams"]}.')
    print(f'The run found {bestrun["goldmatch_no"]} of {len(goldmatches)} goldmatches '
          f'and {bestrun["testmatch_no"]} matches in total.')
    print()
    os.chdir(os.getenv('PWD'))

    return bestrun["config"]


def _textpairEvalRun(runno, runnototal, golddir, outdir, goldmatches, q: Queue, ngram, gap, matching_window_size,
                     minimum_matching_ngrams_in_docs, minimum_matching_ngrams_in_window, max_gap,
                     minimum_matching_ngrams, store_banalities):
    print()
    print(f'### Performing run {runno + 1} of {runnototal}')

    resultdir = os.path.join(outdir, 'run_' + str(runno))
    if os.path.exists(resultdir):
        shutil.rmtree(resultdir)
    os.mkdir(resultdir)

    with tempfile.TemporaryDirectory(dir=resultdir) as tmpdir:

        os.chdir(os.getenv('PWD'))
        info = runtextpairFull(golddir, tmpdir, resultdir, ngram, gap, matching_window_size,
                               minimum_matching_ngrams_in_docs, minimum_matching_ngrams_in_window, max_gap,
                               minimum_matching_ngrams, store_banalities)
        os.chdir(os.getenv('PWD'))

        info['resultdir'] = resultdir
        info['run_no'] = runno

        testmatches = []
        rpath = os.path.join(resultdir, 'alignments.jsonl')

        # Extract matches found by the tool
        jlines = readJsonFromJsonlines(rpath)
        for jl in jlines:
            match = [
                (os.path.basename(jl['source_filename'])[0:-4],
                 jl['source_start_byte'],
                 jl['source_start_byte'] + len(jl['source_passage'])),
                (os.path.basename(jl['target_filename'])[0:-4],
                 jl['target_start_byte'],
                 jl['target_start_byte'] + len(jl['target_passage']))
            ]
            testmatches.append(match)

    # Evaluate the recall of goldmatches
    literalgms = evaluateRecall(testmatches, goldmatches)

    info['goldmatch_no'] = len(literalgms)
    info['goldmatches'] = literalgms
    info['testmatch_no'] = len(testmatches)

    saveDictAsJson(os.path.join(resultdir, 'info.json'), info)
    q.put(info)
