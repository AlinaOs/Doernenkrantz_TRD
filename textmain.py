import datetime
import os.path
from preprocessing.Extraction import TextExtractor
from preprocessing.Cleaning import TextCleaner
from preprocessing.Lemmatization import Pseudolemmatizer
from preprocessing.Segmentation import TextSegmenter
from tools.lang import Dictionary

inputdir = 'input'
basedir = 'textbase/base'
nul = 'textbase/norm'
unul = 'textbase/unnorm'

fulltextDk = os.path.join(inputdir, 'teifulltexts/Dk_complete_tei.xml')
fulltextKCh = os.path.join(inputdir, 'teifulltexts/KCh-complete_Layout.xml')
goldxml = os.path.join(inputdir, 'evaluation/testset_annotated_TRD.xml')

filenames = ['Dk_Meta.csv', 'Dk_Pages.csv', 'KCh_Meta.csv', 'KCh_Pages.csv']
pagesDkCSV = os.path.join(basedir, 'Dk_Pages.csv')
metaDkCSV = os.path.join(basedir, 'Dk_Meta.csv')
pagesKChCSV = os.path.join(basedir, 'KCh_Pages.csv')
metaKChCSV = os.path.join(basedir, 'KCh_Meta.csv')

normalize_casesens = os.path.join(inputdir, 'normalization/normalization_casesensitive.csv')
normalize_caselowe_superficial = os.path.join(inputdir, 'normalization/normalization_lowercase_superficial.csv')
normalize_caselowe = os.path.join(inputdir, 'normalization/normalization_lowercase.csv')
prefixes = os.path.join(inputdir, 'normalization/prefixes.csv')
normalize_meta = os.path.join(inputdir, 'normalization/normalization_meta.csv')

pl_un = 'lemmamodels/lov-unnorm'
pl_un_ft = 'lemmamodels/lov-unnorm-finetuned'
pl_un_ren = 'lemmamodels/lov-unnorm-ren'
pl_n = 'lemmamodels/lov-norm'
pl_n_ft = 'lemmamodels/lov-norm-finetuned'

nDCT = Dictionary()
nDCT.registerDocument('Dk')
nDCT.registerDocument('KCh')
unDCT = Dictionary()
unDCT.registerDocument('Dk')
unDCT.registerDocument('KCh')

d = datetime.datetime


def date():
    return d.now().strftime('%Y-%m-%d %X')


def _extracttext():
    TE = TextExtractor()
    print(date() + ': Extracting Dk.')
    TE.extractText(fulltextDk, pagesDkCSV, metaDkCSV)
    print(date() + ': Extracting KCh.')
    TE.extractText(fulltextKCh, pagesKChCSV, metaKChCSV)


def _pagestofulltext(indir, filenames, outdir, normpath, DCT):
    TC = TextCleaner(normalize_casesens, normpath, prefixes, DCT)
    TS = TextSegmenter(TC)
    for fname in filenames:
        TS.getFullText(os.path.join(indir, fname), outdir, fname[:-4])


def _pagestosentences(indir, filenames, outdir, normpath, DCT):
    TC = TextCleaner(normalize_casesens, normpath, prefixes, DCT)
    TS = TextSegmenter(TC)
    for fname in filenames:
        TS.segmentBySentences(os.path.join(indir, fname), os.path.join(outdir, fname), insertion='sentbound')


def _pagestongrams(indir, filenames, outdir, normpath, DCT):
    TC = TextCleaner(normalize_casesens, normpath, prefixes, DCT)
    TS = TextSegmenter(TC)
    for fname in filenames:
        TS.segmentByNGrams(os.path.join(indir, fname), os.path.join(outdir, fname), n=5)


def _lemmatizetext(indir, filenames, outdir, LE):
    for fname in filenames:
        LE.lemmatizeTextFromCSV(os.path.join(indir, fname), os.path.join(outdir, fname))


def _main_prelem():
    """
    Generates text representations of the Dk and KCh TEI-fulltexts. The representations are combinations of two
    different stages of preprocessing and three different types of segmentation:

    *Preprocessing*
        - unnormalized (only basic text cleaning is performed)
        - normalized (additionally to the basic text cleaning, the text is nomalized according to phonological rules)

    *Segmentation*
        - full text
        - text per pages (and, where known, paragraphs)
        - sentences
    """

    if not os.path.exists(os.path.join(unul, 'pages')):
        os.mkdir(os.path.join(unul, 'pages'))
    if not os.path.exists(os.path.join(unul, 'sentences')):
        os.mkdir(os.path.join(unul, 'sentences'))
    if not os.path.exists(os.path.join(unul, 'full')):
        os.mkdir(os.path.join(unul, 'full'))

    if not os.path.exists(os.path.join(nul, 'pages')):
        os.mkdir(os.path.join(nul, 'pages'))
    if not os.path.exists(os.path.join(nul, 'sentences')):
        os.mkdir(os.path.join(nul, 'sentences'))
    if not os.path.exists(os.path.join(nul, 'full')):
        os.mkdir(os.path.join(nul, 'full'))

    print(date() + ': Extracting texts from XML.')
    _extracttext()

    print(date() + ': Performing basic cleaning on the base texts.')
    basefiles = [os.path.join('textbase/base', f) for f in filenames]
    TC = TextCleaner(normalize_casesens, normalize_caselowe_superficial, prefixes, unDCT)
    TC.cleanTextFromMultiCSV(basefiles, os.path.join(unul, 'pages'), normalize=True, lowercase=True,
                             addnorms=[normalize_meta, None, None, None], joinpbs=[False, True, False, True],
                             docids=['Dk', 'Dk', 'KCh', 'KCh'])

    print(date() + ': Performing basic cleaning and normalization on the base texts.')
    TCn = TextCleaner(normalize_casesens, normalize_caselowe, prefixes, nDCT)
    TCn.cleanTextFromMultiCSV(basefiles, os.path.join(nul, 'pages'), normalize=True, lowercase=True,
                              addnorms=[normalize_meta, None, None, None], joinpbs=[False, True, False, True],
                              docids=['Dk', 'Dk', 'KCh', 'KCh'])
    unDCT.export('input/normalization/vocab_full_unnormalized.json')
    nDCT.export('input/normalization/vocab_full_normalized.json')

    print(date() + ': Segmenting texts into sentences.')
    _pagestosentences(
        os.path.join(unul, 'pages'),
        filenames,
        os.path.join(unul, 'sentences'),
        normalize_caselowe_superficial,
        unDCT
    )
    _pagestosentences(
        os.path.join(nul, 'pages'),
        filenames,
        os.path.join(nul, 'sentences'),
        normalize_caselowe,
        nDCT
    )

    print(date() + ': Extracting and indexing fulltext.')
    _pagestofulltext(
        os.path.join(unul, 'pages'),
        filenames,
        os.path.join(unul, 'full'),
        normalize_caselowe_superficial,
        unDCT
    )
    _pagestofulltext(
        os.path.join(nul, 'pages'),
        filenames,
        os.path.join(nul, 'full'),
        normalize_caselowe,
        nDCT
    )


def _main_normlem():
    """
    Generates text representations of the Dk and KCh TEI-fulltexts. The representations are combinations of two
    different stages of preprocessing and three different types of segmentation:

    *Preprocessing*
        - normalized text, pseudo-lemmatized with a basic louvain model
        - normalized text, pseudo-lemmatized with a fine-tuned louvain model (i.e., fine-tuning of
          short-word-communities)

    *Segmentation*
        - full text
        - text per pages (and, where known, paragraphs)
        - sentences
    """

    nl = 'textbase/norm-lem'
    nlf = 'textbase/norm-lem-ft'

    if not os.path.exists(os.path.join(nl, 'pages')):
        os.mkdir(os.path.join(nl, 'pages'))
    if not os.path.exists(os.path.join(nl, 'sentences')):
        os.mkdir(os.path.join(nl, 'sentences'))
    if not os.path.exists(os.path.join(nl, 'full')):
        os.mkdir(os.path.join(nl, 'full'))

    if not os.path.exists(os.path.join(nlf, 'pages')):
        os.mkdir(os.path.join(nlf, 'pages'))
    if not os.path.exists(os.path.join(nlf, 'sentences')):
        os.mkdir(os.path.join(nlf, 'sentences'))
    if not os.path.exists(os.path.join(nlf, 'full')):
        os.mkdir(os.path.join(nlf, 'full'))

    print(date() + ': Lemmatizing normalized texts with PL-N.')
    PL_N = Pseudolemmatizer(dirpath=pl_n, name='PL-N', mode='load')
    _lemmatizetext(os.path.join(nul, 'pages'), filenames, os.path.join(nl, 'pages'), PL_N)
    _lemmatizetext(os.path.join(nul, 'sentences'), filenames, os.path.join(nl, 'sentences'), PL_N)

    print(date() + ': Lemmatizing normalized texts with PL-NF.')
    PL_NF = Pseudolemmatizer(dirpath=pl_n_ft, name='PL-NF', mode='load')
    _lemmatizetext(os.path.join(nul, 'pages'), filenames, os.path.join(nlf, 'pages'), PL_NF)
    _lemmatizetext(os.path.join(nul, 'sentences'), filenames, os.path.join(nlf, 'sentences'), PL_NF)

    print(date() + ': Extracting lemmatized fulltexts.')
    normpath = normalize_caselowe
    _pagestofulltext(os.path.join(nl, 'pages'), filenames, os.path.join(nl, 'full'), normpath, nDCT)
    _pagestofulltext(os.path.join(nlf, 'pages'), filenames, os.path.join(nlf, 'full'), normpath, nDCT)


def _main_unnormlem():
    """
    Generates text representations of the Dk and KCh TEI-fulltexts. The representations are combinations of three
    different stages of preprocessing and three different types of segmentation:

    *Preprocessing*
        - unnormalized text, pseudo-lemmatized with a basic louvain model
        - unnormalized text, pseudo-lemmatized with a fine-tuned louvain model (i.e., fine-tuning of
          short-word-communities)
        - unnormalized text, pseudo-lemmatized with a fine-tuned and ReN-fine-tuned louvain model (i.e., fine-tuning of
          short-word-communities and fine-tuning using the ReN-corpus)

    *Segmentation*
        - full text
        - text per pages (and, where known, paragraphs)
        - sentences
    """

    unl = 'textbase/unnorm-lem'
    unlf = 'textbase/unnorm-lem-ft'
    unlr = 'textbase/unnorm-lem-ren'

    if not os.path.exists(os.path.join(unl, 'pages')):
        os.mkdir(os.path.join(unl, 'pages'))
    if not os.path.exists(os.path.join(unl, 'sentences')):
        os.mkdir(os.path.join(unl, 'sentences'))
    if not os.path.exists(os.path.join(unl, 'full')):
        os.mkdir(os.path.join(unl, 'full'))

    if not os.path.exists(os.path.join(unlf, 'pages')):
        os.mkdir(os.path.join(unlf, 'pages'))
    if not os.path.exists(os.path.join(unlf, 'sentences')):
        os.mkdir(os.path.join(unlf, 'sentences'))
    if not os.path.exists(os.path.join(unlf, 'full')):
        os.mkdir(os.path.join(unlf, 'full'))

    if not os.path.exists(os.path.join(unlr, 'pages')):
        os.mkdir(os.path.join(unlr, 'pages'))
    if not os.path.exists(os.path.join(unlr, 'sentences')):
        os.mkdir(os.path.join(unlr, 'sentences'))
    if not os.path.exists(os.path.join(unlr, 'full')):
        os.mkdir(os.path.join(unlr, 'full'))

    print(date() + ': Lemmatizing unnormalized texts with PL-UN.')
    PL_UN = Pseudolemmatizer(dirpath=pl_un, name='PL-UN', mode='load')
    _lemmatizetext(os.path.join(unul, 'pages'), filenames, os.path.join(unl, 'pages'), PL_UN)
    _lemmatizetext(os.path.join(unul, 'sentences'), filenames, os.path.join(unl, 'sentences'), PL_UN)

    print(date() + ': Lemmatizing unnormalized texts with PL-UNF.')
    PL_UNF = Pseudolemmatizer(dirpath=pl_un_ft, name='PL-UNF', mode='load')
    _lemmatizetext(os.path.join(unul, 'pages'), filenames, os.path.join(unlf, 'pages'), PL_UNF)
    _lemmatizetext(os.path.join(unul, 'sentences'), filenames, os.path.join(unlf, 'sentences'), PL_UNF)

    print(date() + ': Lemmatizing unnormalized texts with PL-UNF-REN.')
    PL_UNFR = Pseudolemmatizer(dirpath=pl_un_ren, name='PL-UNF-REN', mode='load')
    _lemmatizetext(os.path.join(unul, 'pages'), filenames, os.path.join(unlr, 'pages'), PL_UNFR)
    _lemmatizetext(os.path.join(unul, 'sentences'), filenames, os.path.join(unlr, 'sentences'), PL_UNFR)

    print(date() + ': Extracting lemmatized fulltexts.')
    normpath = normalize_caselowe_superficial
    _pagestofulltext(os.path.join(unl, 'pages'), filenames, os.path.join(unl, 'full'), normpath, unDCT)
    _pagestofulltext(os.path.join(unlf, 'pages'), filenames, os.path.join(unlf, 'full'), normpath, unDCT)
    _pagestofulltext(os.path.join(unlr, 'pages'), filenames, os.path.join(unlr, 'full'), normpath, unDCT)


def main():
    """
    Generates text representations of the Dk and KCh TEI-fulltexts. The representations are combinations of seven
    different stages of preprocessing and three different types of segmentation:

    *Preprocessing*
        - unnormalized and unlemmatized (only basic text cleaning is performed)
        - normalized and unlemmatized (additionally to the basic text cleaning, the text is nomalized according to
          phonological rules)
        - normalized text, pseudo-lemmatized with a basic louvain model
        - normalized text, pseudo-lemmatized with a fine-tuned louvain model (i.e., fine-tuning of
          short-word-communities)
        - unnormalized text, pseudo-lemmatized with a basic louvain model
        - unnormalized text, pseudo-lemmatized with a fine-tuned louvain model (i.e., fine-tuning of
          short-word-communities)
        - unnormalized text, pseudo-lemmatized with a fine-tuned and ReN-fine-tuned louvain model (i.e., fine-tuning of
          short-word-communities and fine-tuning using the ReN-corpus)

    *Segmentation*
        - full text
        - text per pages (and, where known, paragraphs)
        - sentences
    """

    _main_prelem()
    _main_unnormlem()
    _main_normlem()


if __name__ == '__main__':
    main()
