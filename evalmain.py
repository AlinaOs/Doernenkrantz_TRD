import datetime
import os

from preprocessing.Cleaning import TextCleaner
from preprocessing.Extraction import TextExtractor
from preprocessing.Lemmatization import Pseudolemmatizer
from textmain import _pagestosentences, _pagestofulltext, _lemmatizetext
from tools.lang import Dictionary
from trd.evaluation import extractReusePairsFromDir, combievaluateTool, collectEvalScores
from trd.formatting import prepareTextsFromDir
from trd.wrapping import runblastFull, runtextpairFull, runpassimFull

indir = 'input'
goldxml = 'input/evaluation/testset_annotated_TRD.xml'
evaldir = 'input/evaluation'
goldtextdir = 'golddata'

normalize_casesens = os.path.join(indir, 'normalization/normalization_casesensitive.csv')
normalize_caselowe_superficial = os.path.join(indir, 'normalization/normalization_lowercase_superficial.csv')
normalize_caselowe = os.path.join(indir, 'normalization/normalization_lowercase.csv')
prefixes = os.path.join(indir, 'normalization/prefixes.csv')
normalize_meta = os.path.join(indir, 'normalization/normalization_meta.csv')

pl_un = 'lemmamodels/lov-unnorm'
pl_un_ft = 'lemmamodels/lov-unnorm-finetuned'
pl_un_ren = 'lemmamodels/lov-unnorm-ren'
pl_n = 'lemmamodels/lov-norm'
pl_n_ft = 'lemmamodels/lov-norm-finetuned'

ebase = os.path.join(goldtextdir, 'base')
enul = os.path.join(goldtextdir, 'norm')
eunul = os.path.join(goldtextdir, 'unnorm')
efilenames = ['Dk_gold_meta.csv', 'Dk_gold_pages.csv', 'KCh_gold_meta.csv', 'KCh_gold_pages.csv']

nDCT = Dictionary('input/normalization/vocab_full_normalized.json')
unDCT = Dictionary('input/normalization/vocab_full_unnormalized.json')

d = datetime.datetime

goldbase = 'golddata'
textbase = 'trdinput'
segstages = ['full', 'pages', 'sentences']
normstages = os.listdir(textbase)
golddirs = []
textdirs = []
# outdir = 'test'
evaloutdir = 'trdoutput/evaluation'
# resultdir = 'testresults'
resultdir = 'trdoutput/results'


def date():
    return d.now().strftime('%Y-%m-%d %X')


def preprocessGolddata():
    segstages = ['pages', 'sentences', 'full']
    prepstages = ['unnorm', 'unnorm-lem', 'unnorm-lem-ft', 'unnorm-lem-ren', 'norm', 'norm-lem', 'norm-lem-ft']
    for ps in prepstages:
        for sst in segstages:
            os.makedirs(os.path.join(goldtextdir, ps, sst), exist_ok=True)
    os.makedirs(os.path.join(goldtextdir, 'base'), exist_ok=True)

    TE = TextExtractor()
    print(date() + ': Extracting Golddata.')
    TE.extractGoldText(goldxml, ebase)

    ebasefiles = [os.path.join(goldtextdir, 'base', f) for f in efilenames]
    print(date() + ': Performing basic cleaning on the golddata base texts.')
    TC = TextCleaner(normalize_casesens, normalize_caselowe_superficial, prefixes, unDCT)
    TC.cleanTextFromMultiCSV(ebasefiles, os.path.join(eunul, 'pages'), normalize=True, lowercase=True,
                             addnorms=[normalize_meta, None, None, None], joinpbs=[False, True, False, True],
                             docids=['Dk', 'Dk', 'KCh', 'KCh'])
    print(date() + ': Performing basic cleaning and normalization on the golddata base texts.')
    TCn = TextCleaner(normalize_casesens, normalize_caselowe, prefixes, nDCT)
    TCn.cleanTextFromMultiCSV(ebasefiles, os.path.join(enul, 'pages'), normalize=True, lowercase=True,
                              addnorms=[normalize_meta, None, None, None], joinpbs=[False, True, False, True],
                              docids=['Dk', 'Dk', 'KCh', 'KCh'])

    print(date() + ': Segmenting golddata texts into sentences.')
    _pagestosentences(
        os.path.join(eunul, 'pages'),
        efilenames,
        os.path.join(eunul, 'sentences'),
        normalize_caselowe_superficial,
        unDCT
    )
    _pagestosentences(
        os.path.join(enul, 'pages'),
        efilenames,
        os.path.join(enul, 'sentences'),
        normalize_caselowe,
        nDCT
    )

    print(date() + ': Extracting and indexing fulltext.')
    _pagestofulltext(
        os.path.join(eunul, 'pages'),
        efilenames,
        os.path.join(eunul, 'full'),
        normalize_caselowe_superficial,
        unDCT
    )
    _pagestofulltext(
        os.path.join(enul, 'pages'),
        efilenames,
        os.path.join(enul, 'full'),
        normalize_caselowe,
        nDCT
    )

    eunl = os.path.join(goldtextdir, 'unnorm-lem')
    eunlf = os.path.join(goldtextdir, 'unnorm-lem-ft')
    eunlr = os.path.join(goldtextdir, 'unnorm-lem-ren')
    enl = os.path.join(goldtextdir, 'norm-lem')
    enlf = os.path.join(goldtextdir, 'norm-lem-ft')

    print(date() + ': Lemmatizing unnormalized golddata texts with PL-UN.')
    PL_UN = Pseudolemmatizer(dirpath=pl_un, name='PL-UN', mode='load')
    _lemmatizetext(os.path.join(eunul, 'pages'), efilenames, os.path.join(eunl, 'pages'), PL_UN)
    _lemmatizetext(os.path.join(eunul, 'sentences'), efilenames, os.path.join(eunl, 'sentences'), PL_UN)

    print(date() + ': Lemmatizing unnormalized golddata texts with PL-UNF.')
    PL_UNF = Pseudolemmatizer(dirpath=pl_un_ft, name='PL-UNF', mode='load')
    _lemmatizetext(os.path.join(eunul, 'pages'), efilenames, os.path.join(eunlf, 'pages'), PL_UNF)
    _lemmatizetext(os.path.join(eunul, 'sentences'), efilenames, os.path.join(eunlf, 'sentences'), PL_UNF)

    print(date() + ': Lemmatizing unnormalized golddata texts with PL-UNF-REN.')
    PL_UNFR = Pseudolemmatizer(dirpath=pl_un_ren, name='PL-UNF-REN', mode='load')
    _lemmatizetext(os.path.join(eunul, 'pages'), efilenames, os.path.join(eunlr, 'pages'), PL_UNFR)
    _lemmatizetext(os.path.join(eunul, 'sentences'), efilenames, os.path.join(eunlr, 'sentences'), PL_UNFR)

    print(date() + ': Extracting lemmatized golddata fulltexts.')
    normpath = normalize_caselowe_superficial
    _pagestofulltext(os.path.join(eunl, 'pages'), efilenames, os.path.join(eunl, 'full'), normpath, unDCT)
    _pagestofulltext(os.path.join(eunlf, 'pages'), efilenames, os.path.join(eunlf, 'full'), normpath, unDCT)
    _pagestofulltext(os.path.join(eunlr, 'pages'), efilenames, os.path.join(eunlr, 'full'), normpath, unDCT)

    print(date() + ': Lemmatizing normalized texts with PL-N.')
    PL_N = Pseudolemmatizer(dirpath=pl_n, name='PL-N', mode='load')
    _lemmatizetext(os.path.join(enul, 'pages'), efilenames, os.path.join(enl, 'pages'), PL_N)
    _lemmatizetext(os.path.join(enul, 'sentences'), efilenames, os.path.join(enl, 'sentences'), PL_N)

    print(date() + ': Lemmatizing normalized texts with PL-NF.')
    PL_NF = Pseudolemmatizer(dirpath=pl_n_ft, name='PL-NF', mode='load')
    _lemmatizetext(os.path.join(enul, 'pages'), efilenames, os.path.join(enlf, 'pages'), PL_NF)
    _lemmatizetext(os.path.join(enul, 'sentences'), efilenames, os.path.join(enlf, 'sentences'), PL_NF)

    print(date() + ': Extracting lemmatized fulltexts.')
    normpath = normalize_caselowe
    _pagestofulltext(os.path.join(enl, 'pages'), efilenames, os.path.join(enl, 'full'), normpath, nDCT)
    _pagestofulltext(os.path.join(enlf, 'pages'), efilenames, os.path.join(enlf, 'full'), normpath, nDCT)


def runbestBlast():
    blast_prepstages = []
    for norm in normstages:
        blast_prepstages.append(os.path.join(norm, 'full'))
    blast_golddirs = [os.path.join(goldbase, bps) for bps in blast_prepstages]
    blast_textdirs = [os.path.join(textbase, bps) for bps in blast_prepstages]

    blast_params = {
        'e_value': [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001],
        'word_size': [3, 4, 5, 6, 7],
        'min_length': [15, 20, 30, 40]
    }
    print()
    print()
    print('### Combievaluating BLAST.')
    b_bestcombi = combievaluateTool('blast',
                                    blast_golddirs,
                                    os.path.join(evaloutdir, 'blast'),
                                    blast_params,
                                    preparegolddirs=False)

    b_dir = blast_textdirs[b_bestcombi['result']['dir_idx']]
    b_bestconf = b_bestcombi['result']['runinfo']['config']
    blastresults = os.path.join(resultdir, 'blast')
    os.makedirs(blastresults, exist_ok=True)

    print()
    print()
    print('### Running BLAST with the best combination of preprocessing and parameters:')
    print(f'Preprocessing: {blast_prepstages[b_bestcombi["result"]["dir_idx"]]}')
    print(f'Input data: {b_dir}')
    print(f'Params: {b_bestconf}')
    runblastFull(b_dir, blastresults, e_value=b_bestconf['e_value'],
                 word_size=b_bestconf['word_size'])
    print(f'### Finished running BLAST. Results can be found in {blastresults}.')
    print()


def runbestPassim():
    passim_segmented_params = {
        'n': [5, 7, 9],
        'min_match': [4, 5, 6],
        'maxDF': [100, 300, 500],
        'floating_ngrams': [True],
        'beam': [20],
        'min_align': [20, 30, 50],
        'pcopy': [0.6, 0.7, 0.8],
        'all_pairs': [True]
    }
    passim_full_params = {
        'n': [5, 7, 9],
        'min_match': [2],
        'maxDF': [100],
        'floating_ngrams': [True],
        'beam': [20, 25, 30],
        'min_align': [20, 30, 50],
        'pcopy': [0.6, 0.7, 0.8],
        'all_pairs': [True]
    }
    passim_params = []
    for i in range(7):
        passim_params.append(passim_full_params)
    for i in range(14):
        passim_params.append(passim_segmented_params)

    print()
    print()
    print('### Combievaluating Passim.')
    p_bestcombi = combievaluateTool('passim', golddirs, os.path.join(evaloutdir, 'passim'), passim_params,
                                    preparegolddirs=False)

    p_dir = textdirs[p_bestcombi['result']['dir_idx']]
    p_bestconf = p_bestcombi['result']['runinfo']['config']
    passimresults = os.path.join(resultdir, 'passim')
    os.makedirs(passimresults, exist_ok=True)

    print()
    print()
    print('### Running Passim with the best combination of preprocessing and parameters:')
    print(f'Preprocessing: {prepstages[p_bestcombi["result"]["dir_idx"]]}')
    print(f'Input data: {p_dir}')
    print(f'Params: {p_bestconf}')
    print()

    runpassimFull(p_dir, passimresults,
                  n=p_bestconf['n'], min_align=p_bestconf['min_align'],
                  beam=p_bestconf['beam'], pcopy=p_bestconf['pcopy'],
                  all_pairs=p_bestconf['all_pairs'], maxDF=p_bestconf['maxDF'],
                  floating_ngrams=p_bestconf['floating_ngrams'],
                  min_match=p_bestconf['min_match'])
    print(f'### Finished running Passim. Results can be found in {passimresults}.')


def runbestTextpair():
    textpair_params = {
        'ngram': [3, 4],
        'gap': [1, 2, 3],
        'matching_window_size': [5, 10],
        'minimum_matching_ngrams_in_docs': [2, 3, 4],
        'minimum_matching_ngrams_in_window': [2, 3, 4],
        'max_gap': [2, 3, 5],
        'minimum_matching_ngrams': [2, 3, 4],
        'store_banalities': [True, False]}

    print()
    print()
    print('### Combievaluating TextPAIR.')
    t_bestcombi = combievaluateTool('textpair', golddirs, os.path.join(evaloutdir, 'textpair'), textpair_params,
                                    preparegolddirs=False)

    t_dir = textdirs[t_bestcombi['result']['dir_idx']]
    t_bestconf = t_bestcombi['result']['runinfo']['config']
    textpairresults = os.path.join(resultdir, 'textpair')
    os.makedirs(textpairresults, exist_ok=True)

    print()
    print()
    print('### Running TextPAIR with the best combination of preprocessing and parameters:')
    print(f'Preprocessing: {prepstages[t_bestcombi["result"]["dir_idx"]]}')
    print(f'Input data: {t_dir}')
    print(f'Params: {t_bestconf}')
    print()

    runtextpairFull(t_dir, textpairresults,
                    ngram=t_bestconf['ngram'], gap=t_bestconf['gap'],
                    matching_window_size=t_bestconf['matching_window_size'],
                    minimum_matching_ngrams_in_docs=t_bestconf['minimum_matching_ngrams_in_docs'],
                    minimum_matching_ngrams_in_window=t_bestconf['minimum_matching_ngrams_in_window'],
                    max_gap=t_bestconf['max_gap'], minimum_matching_ngrams=t_bestconf['minimum_matching_ngrams'],
                    store_banalities=t_bestconf['store_banalities'])
    print(f'### Finished running TextPAIR. Results can be found in {textpairresults}.')


def preprocessGolddirs():
    for i in range(len(golddirs)):
        gdir = golddirs[i]
        prepareTextsFromDir(gdir, gdir)
        extractReusePairsFromDir(gdir, gdir)


def runbestTextpairFull():
    t_dir = os.path.join(textbase, 'unnorm-lem/full')
    #t_dir = os.path.join(goldbase, 'unnorm-lem/full')
    t_bestconf = {'ngram': '3', 'gap': '1', 'matching_window_size': '10', 'minimum_matching_ngrams_in_docs': '2',
                  'minimum_matching_ngrams_in_window': '2', 'max_gap': '3', 'minimum_matching_ngrams': '2',
                  'store_banalities': False}
    textpairresults = os.path.join(resultdir, 'textpair')
    os.makedirs(textpairresults, exist_ok=True)
    runtextpairFull(t_dir, textpairresults,
                    ngram=t_bestconf['ngram'], gap=t_bestconf['gap'],
                    matching_window_size=t_bestconf['matching_window_size'],
                    minimum_matching_ngrams_in_docs=t_bestconf['minimum_matching_ngrams_in_docs'],
                    minimum_matching_ngrams_in_window=t_bestconf['minimum_matching_ngrams_in_window'],
                    max_gap=t_bestconf['max_gap'], minimum_matching_ngrams=t_bestconf['minimum_matching_ngrams'],
                    store_banalities=t_bestconf['store_banalities'])


if __name__ == '__main__':

    preprocessGolddata()

    os.makedirs(evaloutdir, exist_ok=True)
    os.makedirs(resultdir, exist_ok=True)

    prepstages = []
    for seg in segstages:
        for norm in normstages:
            prepstages.append(os.path.join(norm, seg))
    golddirs = [os.path.join(goldbase, ps) for ps in prepstages]
    textdirs = [os.path.join(textbase, ps) for ps in prepstages]

    #golddirs = ['golddata/unnorm/full']
    #textdirs = ['testinput/unnorm/full']

    preprocessGolddirs()

    runbestBlast()

    runbestTextpair()

    runbestTextpairFull()

    runbestPassim()

    print('Collecting scores for BLAST.')
    collectEvalScores('trdoutput/evaluation/blast', 'trdoutput/evaluation/blast_evaluation_scores.csv')
    print('Collecting scores for Passim.')
    collectEvalScores('trdoutput/evaluation/passim', 'trdoutput/evaluation/passim_evaluation_scores.csv')
    print('Collecting scores for TextPAIR.')
    collectEvalScores('trdoutput/evaluation/textpair', 'trdoutput/evaluation/textpair_evaluation_scores.csv')
