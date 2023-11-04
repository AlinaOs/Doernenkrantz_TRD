import datetime
import os
import shutil
import tempfile
from multiprocessing import Queue, Manager, Process
from io import StringIO
import psutil

from tools.rw import saveDictAsJson, readDictFromJson, readFromTxt, readJsonFromJsonlines, readFromCSV, writeToCSV
from trd.formatting import blastifyPreparedDir, passimifyPreparedDir, tpairifyPreparedDir, prepareTextsFromDir
from trd.wrapping import runblastFull, runpassimFull, runtextpairFull


def extractReusePairsFromDir(indir, outdir):
    segments = {}
    infiles = os.listdir(indir)

    for f in infiles:
        if not (f.endswith('.txt') or f.endswith('.csv')):
            continue
        elif f.endswith('.txt'):
            segments.update(extractReusePairsFromTxt(os.path.join(indir, f), os.path.join(outdir, f), f))
        elif f.endswith('.csv'):
            segments.update(extractReusePairsFromCsv(os.path.join(indir, f), os.path.join(outdir, f)))

    matches = []
    for segid in segments.keys():
        seg = segments[segid]
        if 'corresp' in seg.keys():
            for sourceid in seg['corresp'].split('&'):
                source = segments[sourceid]
                match = {
                    'type': seg['type'],
                    'source': source['passages'],
                    'target': seg['passages']
                }
                matches.append(match)

    saveDictAsJson(os.path.join(outdir, 'goldmatches.json'), {'matches': matches})


def extractReusePairsFromTxt(infile, outfile, fname):
    segments = {}
    opensegments = {}
    fid = os.path.splitext(fname)[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = os.path.join(tmpdir, 'tmp.text')
        with open(infile, 'r', encoding='utf-8') as infile:
            with open(tmppath, 'w', encoding='utf-8') as tmpf:
                towrite = ''
                nextchar = infile.read(1)
                offset = 0
                while nextchar != '':
                    if nextchar == '$':
                        tmpf.write(towrite)
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
                                'text': '',
                                'docid': fid
                            }
                        else:
                            segid = segmeta[1]
                            opensegments[segid]['end'] = offset - 1
                            seg = {
                                'passages': [opensegments.pop(segid)]
                            }
                            segmeta = segments[segid]
                            if len(segmeta) > 1:
                                seg['type'] = segmeta[1]
                                seg['corresp'] = segmeta[2]
                            segments[segid] = seg
                        nextchar = infile.read(1)
                    else:
                        offset += 1
                        towrite += nextchar
                        if len(opensegments) > 0:
                            for k in opensegments:
                                opensegments[k]['text'] = opensegments[k]['text'] + nextchar
                        nextchar = infile.read(1)
                tmpf.write(towrite)

        with open(tmppath, 'rb') as f_in:
            with open(outfile, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    return segments


def extractReusePairsFromCsv(infile, outfile):
    segments = {}
    opensegments = {}

    csv = readFromCSV(infile)
    doclines = csv['lines']

    for li in range(len(doclines)):
        l = doclines[li]
        if l[1] == '':
            continue
        docid = l[0]
        if len(opensegments.keys()) > 0:
            for k in opensegments:
                opensegments[k].append({
                    'start': 0,
                    'text': '',
                    'docid': docid
                })
        instring = StringIO(l[1])
        newtext = ''
        towrite = ''
        nextchar = instring.read(1)
        offset = 0
        while nextchar != '':
            if nextchar == '$':
                newtext += towrite
                towrite = ''
                segmeta = ''
                nextchar = instring.read(1)
                while nextchar != '$':
                    segmeta += nextchar
                    nextchar = instring.read(1)
                segmeta = segmeta.split('_')
                if segmeta[0] == 'segstart':
                    segments[segmeta[1]] = segmeta[1:]
                    opensegments[segmeta[1]] = [{
                        'start': offset,
                        'text': '',
                        'docid': docid
                    }]
                else:
                    segid = segmeta[1]
                    seg = {
                        'passages': opensegments.pop(segid)
                    }
                    lastpassage = seg['passages'][-1]
                    lastpassage['end'] = offset - 1
                    segmeta = segments[segid]
                    if len(segmeta) > 1:
                        seg['type'] = segmeta[1]
                        seg['corresp'] = segmeta[2]
                    segments[segid] = seg
                nextchar = instring.read(1)
            else:
                offset += 1
                towrite += nextchar
                if len(opensegments) > 0:
                    for k in opensegments:
                        lastpassage = opensegments[k][-1]
                        lastpassage['text'] = lastpassage['text'] + nextchar
                nextchar = instring.read(1)
        newtext += towrite
        l[1] = newtext
        if len(opensegments.keys()) > 0:
            for k in opensegments:
                lastpassage = opensegments[k][-1]
                lastpassage['end'] = offset

    writeToCSV(outfile, doclines, header=csv['header'])
    return segments


def parseGoldmatches(filepath):
    goldmatchdict = readDictFromJson(filepath)
    literal = []
    weak_literal = []
    non_literal = []
    for match in goldmatchdict['matches']:
        parsedmatch = []
        for passage in match['source']:
            parsedmatch.append(
                (passage['docid'], passage['start'], passage['end'])
            )
        for passage in match['target']:
            parsedmatch.append(
                (passage['docid'], passage['start'], passage['end'])
            )
        if match['type'] == 'literal':
            literal.append(parsedmatch)
        elif match['type'] == 'weak-literal':
            weak_literal.append(parsedmatch)
        else:
            non_literal.append(parsedmatch)
    return literal, weak_literal, non_literal


def identicalMatches(match1: list[tuple[str, int, int]], match2: list[tuple[str, int, int]]) -> bool:
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


def evaluatePassageOverlap(goldmatch: list[tuple[str, int, int]], testmatch: list[tuple[str, int, int]],
                           beta=1, ignorefpp=True):
    true_pos = 0
    false_pos = 0
    false_neg = 0
    all_gm = 0

    tp_found = [False for tp in testmatch]
    gp_found = []

    # Für jede Goldpassage:
    for gp in goldmatch:
        all_gm += gp[2] - gp[1]
        gp_found.append(None)
        # für jede Testpassage:
        for i in range(len(testmatch)):
            tp = testmatch[i]
            # Prüfe, ob es sich um dasselbe Dokument handelt. Falls ja, prüfe, ob sich die Indizes überlappen. Falls ja:
            if gp[0] == tp[0] and (
                    (gp[1] <= tp[1] <= gp[2])
                    or
                    (gp[1] <= tp[2] <= gp[2])
                    or
                    (tp[1] <= gp[1] <= tp[2])
                    or
                    (tp[1] <= gp[2] <= tp[2])):

                tp_found[i] = True
                # Berechne, wie groß die Überlappung ist
                # der kleinster Endindex - größter Anfangsindex
                ovs = max(tp[1], gp[1])
                ove = min(tp[2], gp[2])
                overlap = ove - ovs

                if gp_found[-1] is None:
                    gp_found.append(overlap)
                # Sonst: Wenn der F1.5-Score besser ist als vom vorhandenen Wert, ersetze ihn durch die neuen Werte
                elif gp_found[-1] < overlap:
                    gp_found.append(overlap)

                # Addiere jeweils, true positives, false positives und false negatives auf
                true_pos += overlap
                false_neg += (gp[2] - gp[1]) - overlap
                false_pos += (tp[2] - tp[1]) - overlap

        if gp_found[-1] is None:
            false_neg += (gp[2] - gp[1])

    # Für jede Testpassage, die nicht mit einem GM überlappt, addiere fp += len(tp)
    if not ignorefpp:
        for i in range(len(testmatch)):
            if not tp_found[i]:
                tp = testmatch[i]
                false_pos += (tp[2] - tp[1])

    # Berechne Precision
    prec = true_pos / (true_pos + false_pos)
    # Berechne Recall
    rec = true_pos / (true_pos + (all_gm - true_pos))
    # Berechne F1.5-Score
    f = ((beta ** 2 + 1) * prec * rec) / (beta ** 2 * prec + rec)

    # Gib alles zurück
    return prec, rec, f


def evaluateClusterPrecision(goldmatch: list[tuple[str, int, int]], testmatch: list[tuple[str, int, int]]):
    tp_found = [False for tp in testmatch]

    # Für jede Goldpassage:
    for gp in goldmatch:
        # für jede Testpassage:
        for i in range(len(testmatch)):
            tp = testmatch[i]
            # Prüfe, ob es sich um dasselbe Dokument handelt. Falls ja, prüfe, ob sich die Indizes überlappen. Falls ja:
            if gp[0] == tp[0] and (
                    (gp[1] <= tp[1] <= gp[2])
                    or
                    (gp[1] <= tp[2] <= gp[2])
                    or
                    (tp[1] <= gp[1] <= tp[2])
                    or
                    (tp[1] <= gp[2] <= tp[2])):

                tp_found[i] = True

    true_pos = tp_found.count(True)
    false_pos = tp_found.count(False)

    # Berechne Precision
    prec = true_pos / (true_pos + false_pos)
    # Recall und F-Score werden hierfür nicht berechnet, da sie wenig ausssagekräftig sind. Geprüft werden soll nicht,
    # wieviel gefunden wird, sondern ob das gefundene richtig ist

    return prec


def evaluateRecall(testmatches, goldmatches: list[list[tuple[str, int, int]]]):
    foundgms = []
    # Evaluate the recall of goldmatches
    for gmi in range(len(goldmatches)):
        gm = goldmatches[gmi]
        for tmi in range(len(testmatches)):
            tm = testmatches[tmi]
            if identicalMatches(gm, tm):
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
        bestrun = highmatchruns[0]
    else:
        lownoiseruns = []
        mcount = runinfos[highmatchruns[0]]['testmatch_no']
        for r in highmatchruns:
            if runinfos[r]['testmatch_no'] < mcount:
                lownoiseruns = [r]
            elif runinfos[r]['testmatch_no'] == mcount:
                lownoiseruns.append(r)

        if len(lownoiseruns) == 1:
            bestrun = lownoiseruns[0]
        else:
            fastruns = []
            tme = runinfos[lownoiseruns[0]]['time']
            for r in lownoiseruns:
                if runinfos[r]['time'] < tme:
                    fastruns = [r]
                elif runinfos[r]['time'] == tme:
                    fastruns.append(r)
            bestrun = fastruns[0]

    return bestrun


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
        if not (f.endswith('.txt') or f.endswith('.csv')):
            continue
        if f.endswith('.txt'):
            txt = readFromTxt(os.path.join(textdir, f))
            title = f.split('.')[0]
            texts[title] = txt
        else:
            lines = readFromCSV(os.path.join(textdir, f))['lines']
            for l in lines:
                texts[l[0]] = l[1]

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
    cpuno = int(len(psutil.Process().cpu_affinity()) / cpudivisor)
    if cpuno == 0:
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


def evaluateBlast(outdir: str, golddir: str, goldmatches: list[list[tuple[str, int, int]]], e_value: list[float],
                  word_size: list[int], min_length: list[int], evalinfo: dict = None):
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
            for ml in min_length:
                runparams.append({
                    'e_value': e,
                    'word_size': ws,
                    'min_length': ml
                })
    paramnames = ['e_value', 'word_size', 'min_length']
    target = _blastEvalRun

    runinfos = _multiprocessEvalRuns(golddir, outdir, goldmatches, target, runparams, paramnames)
    runinfos.sort(key=lambda r: r['run_no'])

    overallinfo = {
        'tool': 'BLAST',
        'runno': len(runparams),
        'test_e_value': e_value,
        'test_word_size': word_size,
        'test_min_length': min_length,
        'all_goldmatches': len(goldmatches),
        'note': evalinfo if evalinfo is not None else dict()
    }

    print('Finding best run.')
    overallinfo['run_details'] = runinfos
    bestrun = runinfos[findBestRun(runinfos)]
    bestrun['goldmatches'] = humanReadableMatchinfo(bestrun['goldmatches'], golddir, goldmatches)
    overallinfo['bestrun'] = bestrun
    saveDictAsJson(os.path.join(outdir, 'evaluation.json'), overallinfo)

    print('Evaluation finished.')
    print(f'Best run uses an e_value of {bestrun["config"]["e_value"]}, a word_size '
          f'of {bestrun["config"]["word_size"]} and a min_length of {bestrun["config"]["word_size"]}.')
    print(f'The run found {bestrun["goldmatch_no"]} of {len(goldmatches)} goldmatches '
          f'and {bestrun["testmatch_no"]} matches in total.')
    print()

    return bestrun


def _blastEvalRun(runno, runnototal, golddir, outdir, goldmatches, q: Queue, e_value, word_size, min_length):
    print()
    print(f'### Performing run {runno + 1} of {runnototal}')

    resultdir = os.path.join(outdir, 'run_' + str(runno))
    if os.path.exists(resultdir):
        shutil.rmtree(resultdir)
    os.mkdir(resultdir)

    with tempfile.TemporaryDirectory() as tmpdir:
        info = runblastFull(golddir, tmpdir, resultdir, e_value, word_size, min_length=min_length,
                            formatting=blastifyPreparedDir)
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


def evaluatePassim(outdir: str, golddir: str, goldmatches: list[list[tuple[str, int, int]]], n: list[int],
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
    bestrun = runinfos[findBestRun(runinfos)]
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

    return bestrun


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
                             min_match=min_match, maxDF=maxDF, formatting=passimifyPreparedDir)
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


def evaluateTextpair(outdir: str, golddir: str, goldmatches: list[list[tuple[str, int, int]]], ngram: list[int],
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

    # Discard all non-reasonable runparams, i.e. where 'minimum_matching_ngrams_in_window' is smaller than
    # 'minimum_matching_ngrams' or where 'minimum_matching_ngrams_in_docs' is not equal to 'minimum_matching_ngrams'
    # where 'max_gap' is greater than 'matching_window_size' or where 'gap' is greater than 'ngram'
    runparams = [
        rp for rp in runparams if rp['ngram'] >= rp['gap'] and
                                  rp['minimum_matching_ngrams_in_docs'] == rp['minimum_matching_ngrams'] and
                                  rp['minimum_matching_ngrams_in_window'] == rp['minimum_matching_ngrams'] and
                                  rp['max_gap'] <= rp['matching_window_size']
    ]

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
    bestrun = runinfos[findBestRun(runinfos)]
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

    return bestrun


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
                               minimum_matching_ngrams, store_banalities, formatting=tpairifyPreparedDir)
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


def combievaluateTool(tool: str, golddirs: list[str], outdir: str, params, preparegolddirs=True):
    """
    Evaluates the named tool using different parameters as well as different input formats. The function uses the data
    in the given golddata directories and runs an evaluation on each directory with the given parameters. The results of
    all directories are than compared and an overall best combination of parameters and input data is determined and the
    result returned

    :param tool: The name of the tool to be evaluated: 'blast', 'passim', or 'textpair' (case insensitive).
    :param golddirs: A list of directories containing golddata texts (as .txt or .csv files) in different preprocessing
        stages.
    :param outdir: The directory, in which the evaluation results will be saved.
    :param params: The parameters to test during the evaluation of the tool given as a dictionary or list of
        dictionaries. If a dictionary is given, the parameters will be used for all golddirs. If a list of dictionaries
        is given, it must have the same length as the list of golddirs. For every golddir, the corresponding set of
        params will be used. The evaluated params may therefore differ between golddirs. The key/value-pairs in the
        dictionary(-ies) take the same form as the parameters for the tool's evaluate function.
    :param preparegolddirs: Whether or not to prepare the golddirs and extract the text reuse pairs before running the
        evaluation. If False, the function expects that the golddirs have already been prepared.
    :return: Information about the best run of the tool in the form of a dictionary.
    """

    tool = tool.lower()
    if tool not in ['blast', 'passim', 'textpair']:
        raise ValueError('Tool name not recognized.')

    outdirs = [os.path.join(outdir, 'input_' + str(i)) for i in range(len(golddirs))]

    print('### Preparing golddata directories.')
    if preparegolddirs:
        for i in range(len(golddirs)):
            gdir = golddirs[i]
            prepareTextsFromDir(gdir, gdir)
            extractReusePairsFromDir(gdir, gdir)
    for i in range(len(golddirs)):
        os.makedirs(outdirs[i], exist_ok=True)

    bestruns = []

    curr_params = params
    d = datetime.datetime
    date = d.now().strftime('%Y-%m-%d')

    dirno = len(golddirs)
    print('### Starting evaluation.')
    for i in range(len(golddirs)):
        gdir = golddirs[i]
        print(f'{d.now().strftime("%Y-%m-%d %X")} - Iteration {i + 1} of {dirno}: '
              f'Evaluating parameters using golddata from {gdir}')
        goldmatches = parseGoldmatches(os.path.join(gdir, 'goldmatches.json'))
        goldmatches = goldmatches[0] + goldmatches[1]
        evalinfo = {
            'function': 'combievaluate',
            'evaldata': gdir,
            'date': date
        }

        if isinstance(params, list):
            curr_params = params[i]

        if tool == 'blast':
            bestruns.append(evaluateBlast(outdirs[i], gdir, goldmatches,
                                          e_value=curr_params['e_value'],
                                          word_size=curr_params['word_size'],
                                          min_length=curr_params['min_length'],
                                          evalinfo=evalinfo))
        elif tool == 'passim':
            bestruns.append(evaluatePassim(outdirs[i], gdir, goldmatches,
                                           n=curr_params['n'], min_align=curr_params['min_align'],
                                           beam=curr_params['beam'], pcopy=curr_params['pcopy'],
                                           all_pairs=curr_params['all_pairs'],
                                           maxDF=curr_params.get('maxDF', None),
                                           min_match=curr_params.get('min_match', None),
                                           evalinfo=evalinfo))
        else:
            bestruns.append(evaluateTextpair(outdirs[i], gdir, goldmatches,
                                             ngram=curr_params['ngram'], gap=curr_params['gap'],
                                             matching_window_size=curr_params['matching_window_size'],
                                             minimum_matching_ngrams_in_docs=
                                             curr_params['minimum_matching_ngrams_in_docs'],
                                             minimum_matching_ngrams_in_window=
                                             curr_params['minimum_matching_ngrams_in_window'],
                                             max_gap=curr_params['max_gap'],
                                             minimum_matching_ngrams=curr_params['minimum_matching_ngrams'],
                                             store_banalities=curr_params['store_banalities'],
                                             evalinfo=evalinfo))

    print('### Finding the overall best combination.')
    bestrun = findBestRun(bestruns)

    info = {
        'date': date,
        'evaluation': 'combievaluate',
        'tool': tool,
        'result': {
            'dir_idx': bestrun,
            'dir': golddirs[bestrun],
            'runinfo': bestruns[bestrun]
        }
    }

    saveDictAsJson(os.path.join(outdir, 'combievalinfo.json'), info)

    print('### Best combination:')
    print(f'Texts: {info["result"]["dir"]}')
    print(f'Params: {info["result"]["runinfo"]["config"]}')

    return info

def hasMatch(match: list[tuple[str, int, int]], matchlist: list[list[tuple[str, int, int]]]):
    for m in matchlist:
        if identicalMatches(match, m):
            return True
    return False


def fm(T: list[list[tuple[str, int, int]]], G: list[list[tuple[str, int, int]]]):
    G_T = [M_g for M_g in G if hasMatch(M_g, T)]
    T_G = [M_t for M_t in T if hasMatch(M_t, G)]

    R = len(G_T) / len(G)
    if len(T) == 0:
        P = 0
    else:
        P = len(T_G) / len(T)
    if R == 0 and P == 0:
        F = 0.
    else:
        F = (2 * P * R) / (P + R)

    return R, P, F


def fch(T: list[list[tuple[str, int, int]]], G: list[list[tuple[str, int, int]]]):
    allO_g = dict()
    allS_g = set()
    allO_t = dict()
    allS_t = set()

    for M_g in G:
        for M_t in T:
            if not identicalMatches(M_g, M_t):
                continue
            for pg in M_g:
                for pt in M_t:
                    if pg[0] == pt[0] and (
                            (pg[1] <= pt[1] <= pg[2])
                            or
                            (pg[1] <= pt[2] <= pg[2])
                            or
                            (pt[1] <= pg[1] <= pt[2])
                            or
                            (pt[1] <= pg[2] <= pt[2])):
                        allS_g.add(pg)
                        allS_t.add(pt)
                        g_indices = set(range(pg[1], pg[2] + 1))
                        t_indices = set(range(pt[1], pt[2] + 1))
                        overlap = g_indices.intersection(t_indices)
                        allO_g[pg] = allO_g.get(pg, set()).union(overlap)
                        allO_t[pt] = allO_t.get(pt, set()).union(overlap)

    allS_g = [(p[2] + 1) - p[1] for p in allS_g]  # no of chars per overlapped passage in G
    allS_t = [(p[2] + 1) - p[1] for p in allS_t]  # no of chars per overlapping passage in T
    allO_g = [len(allO_g[k]) for k in allO_g.keys()]  # no of overlapped chars per overlapped passage in G
    allO_t = [len(allO_t[k]) for k in allO_t.keys()]  # no of overlapping chars per overlapping passage in T

    if len(allS_g) == 0:
        R = 0
    else:
        R = sum(allO_g) / sum(allS_g)
    if len(allS_t) == 0:
        P = 0
    else:
        P = sum(allO_t) / sum(allS_t)

    if R == 0 and P == 0:
        F = 0.0
    else:
        F = (2 * P * R) / (P + R)

    return R, P, F


def fcl(T: list[list[tuple[str, int, int]]], G: list[list[tuple[str, int, int]]]):
    allG_t = set()  # all detected goldmatches
    allS_g = set()  # all overlapped passages in G
    allT_g = set()  # all detecting testmatches
    allS_t = set()  # all overlapping passages in T

    for i in range(len(G)):
        M_g = G[i]
        for j in range(len(T)):
            M_t = T[j]
            if not identicalMatches(M_g, M_t):
                continue
            for pg in M_g:
                for pt in M_t:
                    matching_pg = set()
                    matching_pt = set()
                    if pg[0] == pt[0] and (
                            (pg[1] <= pt[1] <= pg[2])
                            or
                            (pg[1] <= pt[2] <= pg[2])
                            or
                            (pt[1] <= pg[1] <= pt[2])
                            or
                            (pt[1] <= pg[2] <= pt[2])):
                        matching_pg.add(pg)
                        matching_pt.add(pt)
                        allS_g.add(pg)
                        allS_t.add(pt)

                    if len(matching_pg) > 0:
                        allG_t.add(i)
                        allT_g.add(j)

    allG_t = [len(G[M_g]) for M_g in allG_t]  # no of passages per detected match in G
    allT_g = [len(T[M_t]) for M_t in allT_g]  # no of passages per detecting match in T

    if len(allG_t) == 0:
        R = 0
    else:
        R = len(allS_g) / sum(allG_t)
    if len(allT_g) == 0:
        P = 0
    else:
        P = len(allS_t) / sum(allT_g)
    if R == 0 and P == 0:
        F = 0.0
    else:
        F = (2 * P * R) / (P + R)

    return R, P, F


def calculateEvalScores(type: str, G: list[list[tuple[str, int, int]]], runoutput: str):
    """

    :param type: 'blast', 'passim' or 'textpair'.
    :param G: The golddata against which the run should be evaluated.
    :param runoutput: The path to the output folder of the run.
    :return: F_m, F_ch, F_cl
    """
    type = type.lower()
    T = []

    if type == "passim":
        rundata = readDictFromJson(os.path.join(runoutput, 'out.json'))
        for m in rundata['clusters']:
            match = []
            for p in m['matches']:
                passage = (p['id'], p['begin'], p['end'])
                match.append(passage)
            T.append(match)
    elif type == "blast":
        clusterfiles = os.listdir(runoutput)
        for cf in clusterfiles:
            if cf == 'info.json':
                continue
            rundata = dict(readDictFromJson(os.path.join(runoutput, cf)))
            for k in rundata.keys():
                m = rundata[k]
                match = []
                for p in m['hits']:
                    passage = (p['doc_id'], p['original_indices'][0], p['original_indices'][1])
                    match.append(passage)
                T.append(match)
    elif type == "textpair":
        rundata = readJsonFromJsonlines(os.path.join(runoutput, 'alignments.jsonl'))
        for m in rundata:
            match = []
            source_id = os.path.split(m['source_filename'])[1][0:-4]
            target_id = os.path.split(m['target_filename'])[1][0:-4]
            source = (source_id, m['source_start_byte'], m['source_end_byte'])
            target = (target_id, m['target_start_byte'], m['target_end_byte'])
            match.append(source)
            match.append(target)
            T.append(match)
    else:
        raise ValueError

    sfm = fm(T, G)
    sfch = fch(T, G)
    sfcl = fcl(T, G)

    return sfm, sfch, sfcl


def collectEvalScores(combievaldir, output):
    evaldata = []
    evaldirs = os.listdir(combievaldir)
    for evaldir in evaldirs:
        dirpath = os.path.join(combievaldir, evaldir)
        if not os.path.isdir(dirpath):
            continue
        print('Collecting scores for: ' + dirpath)
        evalinfo = readDictFromJson(os.path.join(dirpath, 'evaluation.json'))
        tool = evalinfo.get('tool', evalinfo['test_tool'])

        inputdata = os.path.split(evalinfo['note']['evaldata'])
        seg = inputdata[1]
        inputdata = os.path.split(inputdata[0])[1]
        norm = 1 if inputdata.startswith('norm') else 0
        if inputdata.endswith('lem'):
            lem = 1
        elif inputdata.endswith('lem-ft'):
            lem = 2
        elif inputdata.endswith('lem-ren'):
            lem = 3
        else:
            lem = 0

        G = parseGoldmatches(os.path.join(evalinfo['note']['evaldata'], 'goldmatches.json'))
        G = list(G[0] + G[1])
        for run in evalinfo['run_details']:
            scores = calculateEvalScores(tool.lower(), G, run['resultdir'])
            rdet = [
                tool,
                run['resultdir'],
                run['testmatch_no'],
                run['goldmatch_no']
            ]
            rdet.extend(scores[0])
            rdet.extend(scores[1])
            rdet.extend(scores[2])
            rdet.extend([
                run['time'],
                seg,
                norm,
                lem
            ])

            rdet.extend(paramstolist(tool.lower(), run['config']))
            evaldata.append(rdet)

    header = ['tool', 'resultdir', 'detected_tms', 'detected_gms',
              'r_m', 'p_m', 'f_m',
              'r_ch', 'p_ch', 'f_ch',
              'r_cl', 'p_cl', 'f_cl',
              'runtime', 'seg', 'norm', 'lem'
              ]

    tool = evaldata[0][0].lower()
    paramhead = []
    if tool == 'blast':
        paramhead = ['word_size', 'e_value', 'min_length', 'max_length', 'language']
    if tool == 'passim':
        paramhead = ['n', 'min_align', 'min_match', 'maxDF', 'floating_ngrams', 'beam', 'pcopy', 'all_pairs']
    if tool == 'textpair':
        paramhead = ['ngram', 'gap', 'matching_window_size', 'minimum_matching_ngrams_in_docs',
                     'minimum_matching_ngrams_in_window', 'max_gap', 'minimum_matching_ngrams', 'store_banalities']

    header.extend(paramhead)

    writeToCSV(output, evaldata, header)


def paramstolist(tool: str, config: dict):
    tool = tool.lower()
    if tool == 'blast':
        plist = [
            config['word_size'],
            config['e_value'],
            config['min_length'],
            config['max_length'],
            config['language']
        ]
    elif tool == 'passim':
        plist = [
            config['n'],
            config['min_align'],
            config['min_match'],
            config['maxDF'],
            config['floating_ngrams'],
            config['beam'],
            config['pcopy'],
            config['all_pairs']
        ]
    elif tool == 'textpair':
        plist = [
            config['ngram'],
            config['gap'],
            config['matching_window_size'],
            config['minimum_matching_ngrams_in_docs'],
            config['minimum_matching_ngrams_in_window'],
            config['max_gap'],
            config['minimum_matching_ngrams'],
            config['store_banalities']
        ]
    else:
        raise ValueError

    return plist
