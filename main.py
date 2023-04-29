from preprocessing.Extraction import TextExtractor
from preprocessing.Cleaning import TextCleaner
from preprocessing.Segmentation import TextSegmenter


fulltextDk = 'fulltexts/Dk_complete_tei.xml'
# fulltextKCh = 'fulltexts/KCh_complete_tei.xml'

pagesDkCSV = 'input/Dk_Pages.csv'
metaDkCSV = 'input/Dk_Meta.csv'
# pagesKChCSV = 'input/KCh_Pages.csv'

normalize = 'input/normalization.csv'
normalize_meta = 'input/normalization_meta.csv'
cleanPagesDkCSV = 'input/DK_Pages_Clean.csv'
cleanMetaDkCSV = 'input/DK_Meta_Clean.csv'
# cleanPagesKChCSV = 'input/KCh_Pages_Clean.csv'
# cleanMetaKChCSV = 'input/KCh_Meta_Clean.csv'

sentencesDkCSV = 'input/DK_sentences.csv'
sentencesMetaDkCSV = 'input/DK_Meta_sentences.csv'
ngramsDkCSV = 'input/DK_ngrams.csv'
ngramsDkCSVsents = 'input/DK_ngrams_sentences.csv'

TE = TextExtractor()
TC = TextCleaner()
TS = TextSegmenter()

#TE.extractDkText(fulltextDk, pagesDkCSV, metaDkCSV)
#TC.cleanTextFromCSV(pagesDkCSV, cleanPagesDkCSV, normalize=normalize)
#TC.cleanTextFromCSV(metaDkCSV, cleanMetaDkCSV, normalize=normalize_meta)
#TS.segmentBySentences(cleanPagesDkCSV, sentencesDkCSV)
#TS.segmentBySentences(cleanMetaDkCSV, sentencesMetaDkCSV)
TS.segmentByNGrams(cleanPagesDkCSV, ngramsDkCSV)
TS.segmentByNGrams(sentencesDkCSV, ngramsDkCSVsents, joinlines=False)
