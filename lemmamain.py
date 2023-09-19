import datetime
import os.path

from preprocessing.Lemmatization import Pseudolemmatizer
from tools.lang import Dictionary
from tools.rw import readDictFromJson

dateobj = datetime.datetime
basepath = 'lemmamodels'
#basepath = 'test/lemmatest'

phonscoring = readDictFromJson('input/normalization/scoring.json')
homoscoring = readDictFromJson('input/normalization/scoring_homography.json')


def date():
    return dateobj.now().strftime('%Y-%m-%d %X')


def main_unnormed(DCTpath):
    DCT = Dictionary(DCTpath)
    print(date() + ' MAIN: Training a model on unnormalized text using Louvain: LOV-UN.')
    un_path = os.path.join(basepath, 'lov-unnorm')
    if not os.path.exists(un_path):
        os.mkdir(un_path)
    PL_UN = Pseudolemmatizer(dirpath=un_path, name='LOV-UN', wordforms=DCT.words, mode=None)
    print(date() + ' MAIN: Calculating similarities and preparing graph.')
    PL_UN.prepare(smcustomization=homoscoring)
    print(date() + ' MAIN: Refiltering edges.')
    PL_UN.refilterEdges(0.5, 4)
    PL_UN.export(reexportgraph=True)
    print(date() + ' MAIN: Training model.')
    PL_UN.train()
    PL_UN.export()
    print(date() + ' MAIN: No fine-tuning of short word communities.')
    print(date() + ' MAIN: Fine-tuning wordgroups.')
    PL_UN.finetuneWordgroups()
    print(date() + ' MAIN: Assigning Lemmata.')
    PL_UN.assignLemmata()
    PL_UN.export()

    print()
    print()
    print(date() + ' MAIN: Fine-tuning the LOV-UN model: LOV-UNF.')
    unf_path = os.path.join(basepath, 'lov-unnorm-finetuned')
    if not os.path.exists(unf_path):
        os.mkdir(unf_path)
    PL_UNF = Pseudolemmatizer(dirpath=unf_path, name='LOV-UNF', mode='load', loadpath=un_path)
    print(date() + ' MAIN: Fine-tuning short word communities.')
    PL_UNF.finetuneShortWordCommunities(smcustomization=phonscoring['scoring'], combilists=phonscoring['combilists'])
    print(date() + ' MAIN: Fine-tuning wordgroups.')
    PL_UNF.finetuneWordgroups()
    print(date() + ' MAIN: Assigning Lemmata.')
    PL_UNF.assignLemmata()
    PL_UNF.export()

    print()
    print()
    print(date() + ' MAIN: Fine-tuning LOV-UNF using ReN: LOV-UNF-ReN.')
    unf_ren_path = os.path.join(basepath, 'lov-unnorm-ren')
    if not os.path.exists(unf_ren_path):
        os.mkdir(unf_ren_path)
    PL_UNF_ReN = Pseudolemmatizer(dirpath=unf_ren_path, name='LOV-UNF-ReN', mode='load', loadpath=unf_path)
    print(date() + ' MAIN: Loading ReN data.')
    lemmamap = readDictFromJson('input/normalization/ReN_forms.json')
    lemmata = list(readDictFromJson('input/normalization/ReN_Lemmata.json').keys())
    print(date() + ' MAIN: Joining lone words according to ReN.')
    PL_UNF_ReN.joinLoneWords(lemmamap, lemmata)
    print(date() + ' MAIN: Joining wordgroups according to ReN.')
    PL_UNF_ReN.joinWordgroups(lemmamap, lemmata)
    PL_UNF_ReN.assignLemmata()
    PL_UNF_ReN.export()


def main_normed(DCTpath):
    DCT = Dictionary(DCTpath)
    print(date() + ' MAIN: Training a model on normalized text using Louvain: LOV-N.')
    n_path = os.path.join(basepath, 'lov-norm')
    if not os.path.exists(n_path):
        os.mkdir(n_path)
    PL_N = Pseudolemmatizer(dirpath=n_path, name='LOV-N', wordforms=DCT.words, mode=None)
    print(date() + ' MAIN: Calculating similarities and preparing graph.')
    PL_N.prepare(smcustomization=homoscoring)
    print(date() + ' MAIN: Refiltering edges.')
    PL_N.refilterEdges(0.5, 4)
    PL_N.export(reexportgraph=True)
    print(date() + ' MAIN: Training model.')
    PL_N.train()
    PL_N.export()
    print(date() + ' MAIN: No fine-tuning of short word communities.')
    print(date() + ' MAIN: Fine-tuning wordgroups.')
    PL_N.finetuneWordgroups()
    print(date() + ' MAIN: Assigning Lemmata.')
    PL_N.assignLemmata()
    PL_N.export()

    print()
    print()
    print(date() + ' MAIN: Fine-tuning the LOV-N model: LOV-NF.')
    nf_path = os.path.join(basepath, 'lov-norm-finetuned')
    if not os.path.exists(nf_path):
        os.mkdir(nf_path)
    PL_NF = Pseudolemmatizer(dirpath=nf_path, name='LOV-NF', mode='load', loadpath=n_path)
    print(date() + ' MAIN: Fine-tuning short word communities.')
    PL_NF.finetuneShortWordCommunities(smcustomization=phonscoring['scoring'], combilists=phonscoring['combilists'])
    print(date() + ' MAIN: Fine-tuning wordgroups.')
    PL_NF.finetuneWordgroups()
    print(date() + ' MAIN: Assigning Lemmata.')
    PL_NF.assignLemmata()
    PL_NF.export()


def resumeLovNorm():
    n_path = os.path.join(basepath, 'lov-norm')

    PL_N = Pseudolemmatizer(dirpath=n_path, name='LOV-N', mode='loadbase')

    print(date() + ' MAIN: Resuming model training.')
    PL_N.train(resumecomdir=os.path.join(n_path, 'comms'))
    PL_N.export()
    print(date() + ' MAIN: No fine-tuning of short word communities.')
    print(date() + ' MAIN: Fine-tuning wordgroups.')
    PL_N.finetuneWordgroups()
    print(date() + ' MAIN: Assigning Lemmata.')
    PL_N.assignLemmata()
    PL_N.export()


def lovNormFinetuned():
    n_path = os.path.join(basepath, 'lov-norm')
    print()
    print()
    print(date() + ' MAIN: Fine-tuning the LOV-N model: LOV-NF.')
    nf_path = os.path.join(basepath, 'lov-norm-finetuned')
    if not os.path.exists(nf_path):
        os.mkdir(nf_path)
    PL_NF = Pseudolemmatizer(dirpath=nf_path, name='LOV-NF', mode='load', loadpath=n_path)
    print(date() + ' MAIN: Fine-tuning short word communities.')
    PL_NF.finetuneShortWordCommunities(smcustomization=phonscoring['scoring'], combilists=phonscoring['combilists'])
    print(date() + ' MAIN: Fine-tuning wordgroups.')
    PL_NF.finetuneWordgroups()
    print(date() + ' MAIN: Assigning Lemmata.')
    PL_NF.assignLemmata()
    PL_NF.export()


if __name__ == '__main__':

    '''
    if not os.path.exists(os.path.join(basepath, 'lov-unnorm')):
        main_unnormed('input/normalization/vocabulary_full_test_bigger.json')
        # main_unnormed('input/normalization/vocab_full_unnormalized.json')
    print()
    print()
    if not os.path.exists(os.path.join(basepath, 'lov-norm')):
        main_normed('input/normalization/vocabulary_full_test_bigger.json')
        # main_normed('input/normalization/vocab_full_normalized.json')
    '''
    resumeLovNorm()
