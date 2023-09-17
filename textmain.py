import datetime
import os.path
from preprocessing.Extraction import TextExtractor
from preprocessing.Cleaning import TextCleaner
from preprocessing.Lemmatization import Pseudolemmatizer
from preprocessing.Segmentation import TextSegmenter
from tools.lang import Dictionary

indir = 'input'
basedir = 'textbase/base'
nl = 'textbase/norm-lem'
unl = 'textbase/unnorm-lem'
nul = 'textbase/norm-unlem'
unul = 'textbase/unnorm-unlem'

fulltextDk = os.path.join(indir, 'teifulltexts/Dk_complete_tei.xml')
fulltextKCh = os.path.join(indir, 'teifulltexts/KCh-complete_Layout.xml')

files = ['Dk_Pages.csv', 'Dk_Meta.csv', 'KCh_Pages.csv', 'KCh_Meta.csv']
pagesDkCSV = os.path.join(basedir, files[0])
metaDkCSV = os.path.join(basedir, files[1])
pagesKChCSV = os.path.join(basedir, files[2])
metaKChCSV = os.path.join(basedir, files[3])

normalize_casesens = os.path.join(indir, 'normalization/normalization_casesensitive.csv')
normalize_caselowe_superficial = os.path.join(indir, 'normalization/normalization_lowercase_superficial.csv')
normalize_caselowe = os.path.join(indir, 'normalization/normalization_lowercase.csv')
prefixes = os.path.join(indir, 'normalization/prefixes.csv')
normalize_meta = os.path.join(indir, 'normalization/normalization_meta.csv')

d = datetime.datetime


def date():
    return d.now().strftime('%Y-%m-%d %X')


def extracttext():
    TE = TextExtractor()
    print(date() + ': Extracting Dk.')
    TE.extractText(fulltextDk, pagesDkCSV, metaDkCSV)
    print(date() + ': Extracting KCh.')
    TE.extractText(fulltextKCh, pagesKChCSV, metaKChCSV)


def cleanedtext(dirpath, lcnormpath, DCT, dictpath):
    TC = TextCleaner(normalize_casesens, lcnormpath, prefixes, DCT)
    print(date() + ': Cleaning Dk.')
    TC.cleanTextFromCSV(pagesDkCSV, os.path.join(dirpath, files[0]), docid='Dk')
    TC.cleanTextFromCSV(metaDkCSV, os.path.join(dirpath, files[1]), addnorm=normalize_meta, docid='Dk')
    print(date() + ': Cleaning KCh.')
    TC.cleanTextFromCSV(pagesKChCSV, os.path.join(dirpath, files[2]), docid='KCh')
    TC.cleanTextFromCSV(metaKChCSV, os.path.join(dirpath, files[3]), docid='KCh')


def pagestofulltext(inpath, outpath, normpath, DCT):
    TC = TextCleaner(normalize_casesens, normpath, prefixes, DCT)
    TS = TextSegmenter(TC)
    for f in files:
        TS.getFullText(os.path.join(inpath, f), outpath, f[:-4])


def pagestosentences(inpath, outpath, normpath, DCT):
    TC = TextCleaner(normalize_casesens, normpath, prefixes, DCT)
    TS = TextSegmenter(TC)
    for f in files:
        TS.segmentBySentences(os.path.join(inpath, f), os.path.join(outpath, f), insertion='sentbound')


def pagestongrams(inpath, outpath, normpath, DCT):
    TC = TextCleaner(normalize_casesens, normpath, prefixes, DCT)
    TS = TextSegmenter(TC)
    for f in files:
        TS.segmentByNGrams(os.path.join(inpath, f), os.path.join(outpath, f), n=5)


def lemmatizetext(inpath, outpath, LE):
    for f in files:
        LE.lemmatizeTextFromCSV(os.path.join(inpath, f), os.path.join(outpath, f))


def main_prelem():
    nDCT = Dictionary()
    nDCT.registerDocument('Dk')
    nDCT.registerDocument('KCh')
    unDCT = Dictionary()
    unDCT.registerDocument('Dk')
    unDCT.registerDocument('KCh')

    print(date() + ': Extracting texts from XML.')
    extracttext()

    print(date() + ': Performing basic cleaning on the base texts.')
    basefiles = os.listdir('textbase/base')
    basefiles.sort()
    basefiles = [os.path.join('textbase/base', f) for f in basefiles]
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
    pagestosentences(
        os.path.join(unul, 'pages'),
        os.path.join(unul, 'sentences'),
        normalize_caselowe_superficial,
        unDCT
    )
    pagestosentences(
        os.path.join(nul, 'pages'),
        os.path.join(nul, 'sentences'),
        normalize_caselowe,
        nDCT
    )

    print(date() + ': Segmenting texts into ngrams.')
    pagestongrams(
        os.path.join(unul, 'pages'),
        os.path.join(unul, 'ngrams'),
        normalize_caselowe_superficial,
        unDCT
    )
    pagestongrams(
        os.path.join(nul, 'pages'),
        os.path.join(nul, 'ngrams'),
        normalize_caselowe,
        nDCT
    )

    print(date() + ': Extracting and indexing fulltext.')
    pagestofulltext(
        os.path.join(unul, 'pages'),
        os.path.join(unul, 'full'),
        normalize_caselowe_superficial,
        unDCT
    )
    pagestofulltext(
        os.path.join(nul, 'pages'),
        os.path.join(nul, 'full'),
        normalize_caselowe,
        nDCT
    )


if __name__ == '__main__':
    main_prelem()