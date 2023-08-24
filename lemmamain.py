import datetime
from preprocessing.Lemmatization import Pseudolemmatizer
from tools.rw import readDictFromJson

dateobj = datetime.datetime


def date():
    return dateobj.now().strftime('%Y-%m-%d %X')


if __name__ == '__main__':

    print(date() + ' MAIN: Training a model on unnormalized text using Louvain: LOV-UN.')
    PL_UN = Pseudolemmatizer(dirpath='input/lemmamodels/lov-unnorm', name='LOV-UN', mode='loadbase')
    print(date() + ' MAIN: Refiltering edges.')
    PL_UN.refilterEdges(0.5, 4)
    print(date() + ' MAIN: Training model.')
    PL_UN.train()
    PL_UN.export(reexportgraph=True)
    print(date() + ' MAIN: Fine-tuning short word communities.')
    scoring = readDictFromJson('input/normalization/scoring.json')
    PL_UN.finetuneShortWordCommunities(smcustomization=scoring['scoring'], combilists=scoring['combilists'])
    print(date() + ' MAIN: Fine-tuning wordgroups.')
    PL_UN.finetuneWordgroups()
    print(date() + ' MAIN: Assigning Lemmata.')
    PL_UN.assignLemmata()
    PL_UN.export()

    print(date() + ' MAIN: Training a model on normalized text using Louvain: LOV-N.')
    PL_N = Pseudolemmatizer(dirpath='input/lemmamodels/lov-norm', name='LOV-N', mode='loadbase')
    print(date() + ' MAIN: Refiltering edges.')
    PL_N.refilterEdges(0.5, 4)
    print(date() + ' MAIN: Training model.')
    PL_N.train()
    PL_N.export(reexportgraph=True)
    print(date() + ' MAIN: No fine-tuning of short word communities.')
    print(date() + ' MAIN: Fine-tuning wordgroups.')
    PL_N.finetuneWordgroups()
    print(date() + ' MAIN: Assigning Lemmata.')
    PL_N.assignLemmata()
    PL_N.export()

    print(date() + ' MAIN: Finetuning the LOV-N model: LOV-NF.')
    PL_NF = Pseudolemmatizer(dirpath='input/lemmamodels/lov-norm-finetuned', name='LOV-NF', mode='load',
                             loadpath='input/lemmamodels/lov-norm')
    print(date() + ' MAIN: Fine-tuning short word communities.')
    scoring = readDictFromJson('input/normalization/scoring.json')
    PL_NF.finetuneShortWordCommunities(smcustomization=scoring['scoring'], combilists=scoring['combilists'])
    print(date() + ' MAIN: Fine-tuning wordgroups.')
    PL_NF.finetuneWordgroups()
    print(date() + ' MAIN: Assigning Lemmata.')
    PL_NF.assignLemmata()
    PL_NF.export()

    '''
    # Test run
    DCT = Dictionary('input/normalization/vocabulary_full_test_small.json')

    PL = Pseudolemmatizer(dirpath='input/lemmamodels/test',
                          wordforms=DCT.words,
                          name='LOV-TST')
    PL.refilterEdges(0.4, 4)
    PL.train(algo='louvain')
    scoring = readDictFromJson('input/normalization/scoring.json')
    PL.finetuneShortWordCommunities(smcustomization=scoring['scoring'], combilists=scoring['combilists'])
    PL.finetuneWordgroups()
    PL.export()
    '''
