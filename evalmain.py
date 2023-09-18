#from preprocessing.Extraction import TextExtractor
#from preprocessing.Cleaning import TextCleaner
#from preprocessing.Segmentation import TextSegmenter
#from tools.lang import Dictionary

import datetime
import os
from trd.evaluation import extractReusePairs, evaluateBlast, parseGoldmatches, evaluatePassim, evaluateTextpair

indir = 'input'
goldxml = 'input/evaluation/testset_annotated_TRD.xml'
evaldir = 'input/evaluation'
textstagesdir = os.path.join(evaldir, 'textstages')
fulltextdir = os.path.join(evaldir, 'fulltexts')

normalize_casesens = os.path.join(indir, 'normalization/normalization_casesensitive.csv')
normalize_caselowe_superficial = os.path.join(indir, 'normalization/normalization_lowercase_superficial.csv')
normalize_caselowe = os.path.join(indir, 'normalization/normalization_lowercase.csv')
prefixes = os.path.join(indir, 'normalization/prefixes.csv')
normalize_meta = os.path.join(indir, 'normalization/normalization_meta.csv')

if __name__ == '__main__':
    '''
    TE = TextExtractor()
    TE.extractGoldText(goldxml, textstagesdir)

    DCT = Dictionary('input/normalization/vocab_full_unnormalized.json')
    DCT.registerDocument('Dk_Meta_Gold')
    DCT.registerDocument('KCh_Meta_Gold')
    DCT.registerDocument('Dk_Gold')
    DCT.registerDocument('KCh_Gold')
    TC = TextCleaner(normalize_casesens, normalize_caselowe_superficial, prefixes, DCT)

    TC.cleanTextFromCSV(os.path.join(textstagesdir, 'Dk_gold_pages.csv'),
                        os.path.join(textstagesdir, 'Dk_gold_pages_clean.csv'),
                        normalize=True, lowercase=True, docid='Dk_Gold')
    TC.cleanTextFromCSV(os.path.join(textstagesdir, 'Dk_gold_meta.csv'),
                        os.path.join(textstagesdir, 'Dk_gold_meta_clean.csv'),
                        normalize=True, lowercase=True, docid='Dk_Meta_Gold')
    TC.cleanTextFromCSV(os.path.join(textstagesdir, 'KCh_gold_pages.csv'),
                        os.path.join(textstagesdir, 'KCh_gold_pages_clean.csv'),
                        normalize=True, lowercase=True, docid='KCh_Gold')
    TC.cleanTextFromCSV(os.path.join(textstagesdir, 'KCh_gold_meta.csv'),
                        os.path.join(textstagesdir, 'KCh_gold_meta_clean.csv'),
                        normalize=True, lowercase=True, docid='KCh_Meta_Gold')

    TS = TextSegmenter(TC)
    TS.getFullText(os.path.join(textstagesdir, 'Dk_gold_pages_clean.csv'),
                   fulltextdir,
                   'Dk_gold_pages')
    TS.getFullText(os.path.join(textstagesdir, 'Dk_gold_meta_clean.csv'),
                   fulltextdir,
                   'Dk_gold_meta')
    TS.getFullText(os.path.join(textstagesdir, 'KCh_gold_pages_clean.csv'),
                   fulltextdir,
                   'KCh_gold_pages')
    TS.getFullText(os.path.join(textstagesdir, 'KCh_gold_meta_clean.csv'),
                   fulltextdir,
                   'KCh_gold_meta')
    '''

    extractReusePairs(fulltextdir, os.path.join(evaldir, 'golddata'))

    literal, weak_literal, non_literal = parseGoldmatches('input/evaluation/golddata/goldmatches.json')
    evalinfo = {
        'textbase': 'unnormed',
        'name': 'test',
        'date': datetime.datetime.now().strftime('%Y-%m-%d')
    }

    evaluateBlast('test/blast', 'input/evaluation/golddata', goldmatches=literal + weak_literal, e_value=[0.001, 0.01],
                  word_size=[5], evalinfo=evalinfo)

    evaluatePassim('test/passim', 'input/evaluation/golddata', goldmatches=literal + weak_literal,
                   n=[8, 9], min_align=[50], all_pairs=[True], pcopy=[0.8], beam=[20], evalinfo=evalinfo)

    evaluateTextpair('test/textpair', 'input/evaluation/golddata', goldmatches=literal + weak_literal, ngram=[3, 5],
                     gap=[2], matching_window_size=[30], minimum_matching_ngrams_in_window=[4],
                     minimum_matching_ngrams_in_docs=[4], max_gap=[15], minimum_matching_ngrams=[4], evalinfo=evalinfo)
