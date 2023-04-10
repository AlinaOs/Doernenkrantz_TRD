from preprocessing.prep import TextExtractor


fulltextDk = 'fulltexts/Dk_complete_tei.xml'
# fulltextKCh = 'fulltexts/KCh_complete_tei.xml'

pagesDkCSV = 'input/Dk_Pages.csv'
metaDkCSV = 'input/Dk_Meta.csv'
# pagesKChCSV = 'input/KCh_Pages.csv'

TE = TextExtractor()

TE.extractDkText(fulltextDk, pagesDkCSV, metaDkCSV)
