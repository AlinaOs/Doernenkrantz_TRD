from preprocessing.prep import TextExtractor, TextCleaner


fulltextDk = 'fulltexts/Dk_complete_tei.xml'
# fulltextKCh = 'fulltexts/KCh_complete_tei.xml'

pagesDkCSV = 'input/Dk_Pages.csv'
metaDkCSV = 'input/Dk_Meta.csv'
# pagesKChCSV = 'input/KCh_Pages.csv'

normalize = 'input/normalization.csv'
cleanPagesDkCSV = 'input/DK_Pages_Clean.csv'
cleanMetaDkCSV = 'input/DK_Meta_Clean.csv'
# cleanPagesKChCSV = 'input/KCh_Pages_Clean.csv'
# cleanMetaKChCSV = 'input/KCh_Meta_Clean.csv'

TE = TextExtractor()
TC = TextCleaner()

TE.extractDkText(fulltextDk, pagesDkCSV, metaDkCSV)
TC.cleanTextFromCSV(pagesDkCSV, cleanPagesDkCSV, normalize=normalize)
TC.cleanTextFromCSV(metaDkCSV, cleanMetaDkCSV, normalize=normalize)
