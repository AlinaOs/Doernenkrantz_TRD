import os
import re
import xml.etree.ElementTree as ET
from tools.rw import saveDictAsJson


vocab = {}
lems = {}


def updateVocab(form, lemma):
    if form in vocab.keys():
        if lemma in vocab[form]:
            vocab[form][lemma] += 1
        else:
            vocab[form][lemma] = 1
    else:
        vocab[form] = {lemma: 1}


def extractLemmata(ren):
    files = os.listdir(ren)

    for file in files:
        tree = ET.parse(os.path.join(ren, file))
        root = tree.getroot()

        for token in root.findall('.//token'):
            formcand = token.find('./anno').attrib['ascii'].lower()
            form = [formcand]
            if not re.fullmatch('[a-z]+', formcand):
                if formcand.find('aͤ') >= 0:
                    form = [formcand.replace('aͤ', 'a'), formcand.replace('aͤ', 'ae')]
                elif formcand.find('eͤ') >= 0:
                    form = [formcand.replace('eͤ', 'e'), formcand.replace('eͤ', 'ee')]
                elif formcand.find('oͤ') >= 0:
                    form = [formcand.replace('oͤ', 'o'), formcand.replace('oͤ', 'oe')]
                elif formcand.find('uͤ') >= 0:
                    form = [formcand.replace('uͤ', 'u'), formcand.replace('uͤ', 'ue')]
                elif formcand.find('uͦ') >= 0:
                    form = [formcand.replace('uͦ', 'u'), formcand.replace('uͦ', 'uo')]
                elif formcand.find('uͮ') >= 0:
                    form = [formcand.replace('uͮ', 'u'), formcand.replace('uͮ', 'v'), formcand.replace('uͮ', 'uu'), formcand.replace('uͮ', 'uv')]
                elif formcand.find('vͤ') >= 0:
                    form = [formcand.replace('vͤ', 'u'), formcand.replace('vͤ', 'v'), formcand.replace('vͤ', 'ue'), formcand.replace('vͤ', 've')]
                else:
                    continue
            lemma = token.find('./anno[lemma_simple]/lemma_simple').attrib['tag'].lower()
            lemma_wsd = token.find('./anno[lemma_wsd]/lemma_wsd').attrib['tag'].lower()
            if lemma_wsd == '--' or lemma == '--':
                if re.fullmatch(r'[ijvuxlcdm]+', form[0]):
                    updateVocab(form[0], '1234')
                    lems.update({'1234': {'lemma_simple': '1234', 'forms': []}})
                else:
                    continue
            else:
                for f in form:
                    updateVocab(f, lemma_wsd)
                lems.update({lemma_wsd: {'lemma_simple': lemma, 'forms': []}})

    ambigueForms = 0
    for formcand in vocab.keys():
        lemmata = dict(vocab.get(formcand))
        if len(lemmata.keys()) == 1:
            lem = lemmata.popitem()[0]
        else:
            ambigueForms += 1
            print(formcand + ': ' + str(lemmata))
            lem = ''
            count = 0
            for lemma in lemmata:
                if lemmata[lemma] > count:
                    lem = lemma
                    count = lemmata[lemma]
        vocab[formcand] = lem
        lems[lem]['forms'] = lems[lem]['forms'] + [formcand]
    print(str(ambigueForms) + ' ambigue forms.')


if __name__ == '__main__':
    REN = 'input/normalization/ReN'
    extractLemmata(REN)
    saveDictAsJson('input/normalization/ReN_forms.json', vocab)
    saveDictAsJson('input/normalization/ReN_Lemmata.json', lems)
