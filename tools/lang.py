import datetime
import gc
import math
import os.path
import shutil
import string
from multiprocessing import Process

import numpy as np
import psutil

from tools.rw import readDictFromJson, saveDictAsJson, readFromCSV


class ScoringMatrix:

    def __init__(self, init, name: str):
        if init is not None and isinstance(init, str):
            SM = np.asarray(readFromCSV(init, False)['lines'])
            self.SM = SM.astype(np.float)
        elif init is not None and isinstance(init, list):
            self.SM = np.asarray(init)
        elif init is not None and isinstance(init, np.ndarray):
            self.SM = init
        else:
            self.SM = np.full((26, 26), -1)
            for i in range(26):
                self.SM[i, i] = 1

        self.name = name

        pos = np.copy(self.SM)
        # pos[pos < 0] = 1
        self.hm = np.max(pos)

    def __getitem__(self, item: tuple[str, str]):
        i = string.ascii_lowercase.index(item[0])
        j = string.ascii_lowercase.index(item[1])
        return self.SM[i, j]

    def getname(self):
        return self.name

    def identify(self, name):
        return True if self.name == name else False

    def highestmatch(self):
        return self.hm


class StringSimilarity:
    SM = {
        'standard': ScoringMatrix(None, 'standard')
    }
    configs = {
        'standard': {
            'SM': 'standard',
            'gap': 0.5,
            'gapex': 1
        },
        'ripuarian': {
            'SM': 'ripuarian',
            'gap': 0.5,
            'gapex': 1
        }
    }

    leftalone = [['c', 'g'], ['g'], ['d']]
    leftfull = ['ck', 'gh', 'dt']
    rightalone = [['k', 'g'], ['z'], ['t']]
    rightfull = ['ck', 'tz', 'dt']
    vowels = ['a', 'e', 'i', 'o', 'u']
    homographies = [('i', 'j'), ('i', 'y'), ('j', 'y'), ('c', 'k'), ('c', 'z'), ('u', 'v'), ('v', 'w'), ('f', 'v')]
    phonologic = [('d', 't'), ('c', 'g'), ('k', 'g'), ('m', 'n'), ('e', 'i'), ('u', 'o'), ('g', 'j'), ('b', 'v'),
                  ('s', 'z'), ('b', 'p')]
    visual = [('k', 'l'), ('b', 'h'), ('t', 'r'), ('s', 'f'), ('i', 't'), ('e', 'c'), ('u', 'n')]

    def __init__(self):

        PhonSM = np.full((26, 26), -1)
        for i in range(26):
            PhonSM[i, i] = 1
        for v in self.vowels:
            for vv in self.vowels:
                if v != vv:
                    PhonSM[string.ascii_lowercase.index(v), string.ascii_lowercase.index(vv)] = -0.5
                    PhonSM[string.ascii_lowercase.index(vv), string.ascii_lowercase.index(v)] = -0.5
        for pair in self.homographies:
            PhonSM[string.ascii_lowercase.index(pair[1]), string.ascii_lowercase.index(pair[0])] = 1
            PhonSM[string.ascii_lowercase.index(pair[0]), string.ascii_lowercase.index(pair[1])] = 1
        for pair in self.phonologic:
            PhonSM[string.ascii_lowercase.index(pair[0]), string.ascii_lowercase.index(pair[1])] = 0
            PhonSM[string.ascii_lowercase.index(pair[1]), string.ascii_lowercase.index(pair[0])] = 0
        for pair in self.visual:
            PhonSM[string.ascii_lowercase.index(pair[0]), string.ascii_lowercase.index(pair[1])] = -0.5
            PhonSM[string.ascii_lowercase.index(pair[1]), string.ascii_lowercase.index(pair[0])] = -0.5
        PhonSM = ScoringMatrix(PhonSM, 'ripuarian')
        self.registerSM(PhonSM)

    def registerSM(self, SM: ScoringMatrix):
        name = SM.getname()
        self.SM.update({name: SM})

    def registerconfig(self, name, smname, gap, gapex):
        self.configs[name] = {
            'SM': smname,
            'gap': gap,
            'gapex': gapex
        }

    def getconfig(self, name):
        if name in self.configs.keys():
            return self.configs[name]
        else:
            return None

    # The code for the newu() function was originally copied from the Needleman-Wunsch implementation at
    # https://gist.github.com/slowkow/06c6dba9180d013dfd82bec217d22eb5.
    # It was broadly restructured and adapted to include a custom scoring matrix and a variable gap penalty. Also, the
    # function was changed to return a result containing different types of information instead of aligned strings.
    def newu(self, x, y, simconf: str = None, phonetic=False):
        if simconf is not None:
            if simconf not in self.configs.keys():
                print('Simconf not in keys. Self.configs:')
                print(self.configs)
                return None
            else:
                simconf = self.configs[simconf]
                if simconf['SM'] not in self.SM.keys():
                    print('SM not in keys. Self.SM:')
                    print(self.SM)
                    return
        else:
            simconf = self.configs['standard']

        if phonetic:
            F, P = self.__nwphonmatrix(x, y, simconf['SM'], simconf['gap'], simconf['gapex'])
        else:
            F, P = self.__nwmatrix(x, y, simconf['SM'], simconf['gap'], simconf['gapex'])

        # Trace through an optimal alignment.
        i = len(x)
        j = len(y)
        rx = []
        ry = []

        while i > 0 and j > 0:
            if P[i, j] == 'diag':
                rx.append(x[i - 1])
                ry.append(y[j - 1])
                i -= 1
                j -= 1
            elif P[i, j] == 'left' or P[i, j] == 'lcom':
                rx.append('-')
                ry.append(y[j - 1])
                j -= 1
            elif P[i, j] == 'top' or P[i, j] == 'tcom':
                rx.append(x[i - 1])
                ry.append('-')
                i -= 1

        if i > 0:
            ry = ry + list(np.full(i, '-'))
            while i > 0:
                rx.append(x[i - 1])
                i -= 1
        else:
            rx = rx + list(np.full(j, '-'))
            while j > 0:
                ry.append(y[j - 1])
                j -= 1

        # Reverse the strings.
        rx = ''.join(rx)[::-1]
        ry = ''.join(ry)[::-1]

        hm = self.SM[simconf['SM']].highestmatch()

        if phonetic:
            normsim = self.__favorleftnwnorm(x, y, F[len(x), len(y)])
        else:
            normsim = self.__nwnorm(x, y, F[len(x), len(y)], hm)

        result = {
            'sim': F[len(x), len(y)],
            'normsim': normsim,
            'alignx': rx,
            'aligny': ry
        }
        return result

    def __nwmatrix(self, x, y, SM: str = None, gapstart=0.5, gapex=1):
        if SM is None:
            SM = self.SM['standard']
        else:
            SM = self.SM[SM]

        nx = len(x)
        ny = len(y)

        # Optimal score at each possible pair of characters.
        F = np.zeros((nx + 1, ny + 1))
        F[:, 0] = np.asarray([0.0] + [-gapstart - i * gapex for i in range(nx)])
        F[0, :] = np.asarray([0.0] + [-gapstart - i * gapex for i in range(ny)])

        # Pointers to trace through an optimal alignment.
        P = np.full((nx + 1, ny + 1), 'none')
        P[:, 0] = 'top'
        P[0, :] = 'left'

        # Temporary scores.
        t = np.zeros(3)
        for i in range(1, nx + 1):
            for j in range(1, ny + 1):
                t[0] = F[i - 1, j - 1] + SM[x[i - 1], y[j - 1]]
                t[1] = F[i, j - 1] - gapex if P[i, j - 1] == 'left' or P[i, j - 1] == 'top' else F[i, j - 1] - gapstart
                t[2] = F[i - 1, j] - gapex if P[i - 1, j] == 'left' or P[i - 1, j] == 'top' else F[i - 1, j] - gapstart

                tmax = np.max(t)
                F[i, j] = tmax

                if t[0] == tmax:
                    P[i, j] = 'diag'
                if t[1] == tmax:
                    P[i, j] = 'left'
                if t[2] == tmax:
                    P[i, j] = 'top'

        return F, P

    def __nwphonmatrix(self, x, y, SM: str = None, gapstart=1.0, gapex=1):
        if SM is None:
            SM = self.SM['standard']
        else:
            SM = self.SM[SM]

        nx = len(x)
        ny = len(y)

        # Optimal score at each possible pair of characters.
        F = np.zeros((nx + 1, ny + 1))
        F[:, 0] = np.asarray([0.0] + [-gapstart - i * gapex for i in range(nx)])
        F[0, :] = np.asarray([0.0] + [-gapstart - i * gapex for i in range(ny)])

        # Pointers to trace through an optimal alignment.
        P = np.full((nx + 1, ny + 1), 'none')
        P[:, 0] = 'top'
        P[0, :] = 'left'
        P[0, 0] = 'diag'

        # Temporary scores.
        t = np.zeros(3)
        for i in range(1, nx + 1):
            for j in range(1, ny + 1):
                t[0] = F[i - 1, j - 1] + self.favorLeft(nx, ny, i - 1, j - 1, SM[x[i - 1], y[j - 1]])

                if SM[x[i - 1], y[j - 1]] >= 0 and (P[i - 1, j - 1] == 'left' or P[i - 1, j - 1] == 'top'):
                    combi = self.checkCombiGap(y, x, j - 1, i - 1, left=False) if P[i - 1, j - 1] == 'top' \
                        else self.checkCombiGap(x, y, i - 1, j - 1, left=False)
                    if combi:
                        if i >= 2 and P[i - 1, j - 1] == 'top' and \
                                (P[i - 2, j - 1] == 'left' or P[i - 2, j - 1] == 'top'):
                            t[0] += self.favorLeft(nx, ny, i-2, j-1, gapex)
                        elif j >= 2 and P[i - 1, j - 1] == 'left' and \
                                (P[i - 1, j - 2] == 'left' or P[i - 1, j - 2] == 'top'):
                            t[0] += self.favorLeft(nx, ny, i-1, j-2, gapex)
                        else:
                            t[0] += self.favorLeft(nx, ny, i-1, j-1, gapstart)

                leftcombi = False
                if P[i, j - 1] == 'diag' and self.checkCombiGap(x, y, i - 1, j - 2, left=True):
                    t[1] = F[i, j - 1]  # combi in y, gapped combi in x
                    leftcombi = True
                else:
                    t[1] = F[i, j - 1] - self.favorLeft(nx, ny, i - 1, j - 1, gapex) if P[i, j - 1] == 'top' or P[
                        i, j - 1] == 'left' \
                        else F[i, j - 1] - self.favorLeft(nx, ny, i - 1, j - 1, gapstart)  # gap in x

                if P[i - 1, j] == 'diag' and self.checkCombiGap(y, x, j - 1, i - 2, left=True):
                    t[2] = F[i - 1, j]  # combi in x, gapped combi in y
                    leftcombi = True
                else:
                    t[2] = F[i - 1, j] - self.favorLeft(nx, ny, i - 1, j - 1, gapex) if P[i - 1, j] == 'top' or P[
                        i - 1, j] == 'left' \
                        else F[i - 1, j] - self.favorLeft(nx, ny, i - 1, j - 1, gapstart)  # gap in y

                tmax = np.max(t)
                F[i, j] = tmax

                if t[0] == tmax:
                    P[i, j] = 'diag'
                if t[1] == tmax:
                    if leftcombi:
                        P[i, j] = 'lcom'
                    else:
                        P[i, j] = 'left'
                if t[2] == tmax:
                    if leftcombi:
                        P[i, j] = 'tcom'
                    else:
                        P[i, j] = 'top'

        return F, P

    def favorLeft(self, lenx, leny, idxx, idxy, score, disfavor=3):
        """

        :param lenx:
        :param leny:
        :param idxx:
        :param idxy:
        :param score:
        :param disfavor: The fraction of the word to score less. If 2 is given, the second half of the word scores less,
        for 3 the last third, for 4 the last fourth and so on. If 1 is given, each letter will be scored less according
        to the length of each word. If 0 is given, the given score is returned, i.e. letter position has no influence on
        scoring.
        :return:
        """

        if disfavor > 0:
            sx = score if idxx+1 <= math.ceil(lenx - (lenx / disfavor)) else score * (1 / lenx) * (lenx - idxx)
            sy = score if idxy+1 <= math.ceil(leny - (leny / disfavor)) else score * (1 / leny) * (leny - idxy)
            score = (sx + sy) / 2
        return score

    def checkCombiGap(self, gappedword: str, fullword: str, gwi: int, fwi: int, left: bool):
        """
        Checks whether or not two strings qualify for a combi-substitution at the given index. A combi-substitution
        is a special case of deletion, where one character is deleted, but since it has been part of a letter
        combination, the deletion doesn't change the phonetic structure of the word. So the deletion of one character
        can be seen as a substitution of two characters by one (the still-standing) character.

        :param gappedword: The word with the gap in it.
        :param fullword: The word without the gap in it.
        :param gwi: The index of the still-standing letter in gappedword.
        :param fwi: The index of the corresponding letter in fullword.
        :param left: Whether or not the still-standing letter is presumed to be the left letter of the combination.
        Setting this to False indicates, that it is the right letter.
        :return: True, if the strings qualify for a combi-substitution. Otherwise False.
        """

        if left:
            # Is the string long enough to contain a combination?
            if len(fullword) == fwi + 1:
                return False
            full = self.leftfull
            alone = self.leftalone
            combi = fullword[fwi:fwi + 1 + 1]
            vowelcombiidx = 0
        else:
            # Is the given index high enough for the string to contain a combination?
            if fwi == 0:
                return False
            full = self.rightfull
            alone = self.rightalone
            combi = fullword[fwi - 1:fwi + 1]
            vowelcombiidx = 1

        normcombi = combi.replace('y', 'i')
        normcombi = normcombi.replace('j', 'i')

        # Is it a double character combi?
        if normcombi[0] == normcombi[1] and normcombi[0] == gappedword[gwi]:
            return True
        # Is it a double vowel combi?
        if normcombi[0] in self.vowels and normcombi[1] in self.vowels and normcombi[vowelcombiidx] == gappedword[gwi]:
            return True

        # If not, check special combinations
        try:
            combiidx = full.index(combi)
        except ValueError:
            return False
        # Is the letter before the gap the corresponding independent letter?
        if gappedword[gwi] not in alone[combiidx]:
            return False
        # If it passes all tests, the wordpair qualifies for the combination bonus
        return True

    def __nwnorm(self, x, y, nwscore, averagematch=1):
        hs = np.max([len(x), len(y)]) * averagematch
        return nwscore / hs

    def commonStemma(self, x, y) -> str:
        # Todo: Return the common stemma of x and y
        return x if x == y else None

    def stemma(self, stem: str, word) -> bool:
        # Todo: Check, whether the stem may be the stemma of the given word
        return True if word.find(stem) >= 0 else False

    def __favorleftnwnorm(self, x, y, nwscore, disfavor=0):
        length = np.max([len(x), len(y)])
        hs = []
        if disfavor > 0:
            for i in range(length):
                if i+1 <= math.ceil(length - (length / disfavor)):
                    hs.append(1)
                else:
                    hs.append((1/length) * (length-i))
            hs = sum(hs)
        else:
            hs = length
        return nwscore / hs

# Function used after kind hint from Kelly Bundy at https://stackoverflow.com/a/76599108
def key(s, t):
    return (s, t) if s < t else (t, s)


# Todo: ind idx?
class Dictionary:
    """
    This class initiates and administrates a dictionary object. It provides functions for accessing,
    manipulating and enhancing the dictionary.
    """

    words = []
    freqs = []
    dfreqs = {}

    def __init__(self, initial=None):
        """
        Initiate a new Dictionary instance. If a path to a preexisting dictionary in a serialized form is given, the
        dictionary will be initialized using the data in this foundational dictionary.
        :param initial: Optional path to a previously exported dictionary.
        """

        if initial is not None:
            self._loadDict(initial)

    def _loadDict(self, path):
        """
        Load a Dictionary from the given path.

        :param path: Path to a previously exported dictionary.
        """
        dictionary = readDictFromJson(path)
        for d in dictionary['docs']:
            self.registerDocument(d['id'], override=True)
        words = sorted(dictionary['words'], key=lambda wo: wo['w'])
        for w in words:
            self.words.append(w['w'])
            self.freqs.append(w['info']['globfreq'])
            for docid in w['info']['docsfreq'].keys():
                self.dfreqs[docid].append(w['info']['docsfreq'][docid])
        del dictionary
        gc.collect()

    def export(self, output, sort='freq'):
        """
        Saves the current state of the Dictionary object as json. Depending on the sort param, the dictionary gets
        first ordered according to:
        - alphabet (sort='word')
        - word frequency descending (sort='freq')
        Any value different to the ones described will default to saving the dictionary in the order the words are
        stored in the Dictionary object.

        :param output:
        :param sort: Sorting rule. Possible values: 'word', 'freq'.
        """
        wordinfo = self._wordinfos()

        if sort == 'freq':
            li = sorted(list(zip(self.words, wordinfo)), key=lambda e: e[1]['globfreq'], reverse=True)
        else:
            li = sorted(list(zip(self.words, wordinfo)))

        words = []
        for w in li:
            words.append({'w': w[0], 'info': w[1]})

        docs = [{'id': k, 'wcount': sum(self.dfreqs[k])} for k in self.dfreqs.keys()]

        saveDictAsJson(output, self._prepareJsonDict(words, docs))

        del words
        gc.collect()

    def _prepareJsonDict(self, words, docs):
        return {
            'words': words,
            'docs': docs
        }

    def _wordinfos(self):
        wordinfo = []
        for i in range(len(self.words)):
            info = {
                'globfreq': self.freqs[i],
                'docsfreq': {}
            }
            for docid in self.dfreqs.keys():
                info['docsfreq'][docid] = self.dfreqs[docid][i]
            wordinfo.append(info)
        return wordinfo

    def registerDocument(self, docid, override=False):
        """
        Registers a document for tracking the frequency of words in an identifiable document.
        :param docid: An identifier string for addressing the document and its word frequencies.
        :param override: If set to True, an already existing document will be overridden by the new document, causing
        its previous frequency data to be deleted. Otherwise, the function will return if the docid already exists.
        """

        if not override and docid in self.dfreqs:
            return
        self.dfreqs[docid] = np.zeros(len(self.words)).tolist()

    def updateDict(self, text, docid=None):
        """
        Use a given text to update the dictionary, i.e., to add new words and/or increase the frequency of already
        registered words. The text will be tokenized according to whitespaces and each token will be considered a word
        to add to the dictionary. No additional text cleaning or normalization is applied.

        :param docid: Optional identifier of the document from which the words are added.
        :param text: The text that contains the words to be added.
        """

        tokens = text.split()
        for token in tokens:
            token = token.lower()
            self.addWord(token, docid)

    def addWord(self, word, docid=None):
        """
        Adds a word to the dictionary, i.e., the frequency of the word's entry is increased by 1. If no entry for the
        word exists, a new entry with the frequency of 1 is created. All frequency-dependent information for the entry
        is updated accordingly. If docid is not None, it must be an ID that has previously been registered
        with the Dictionary object's registerDocument function. Otherwise, it will raise a KeyError.

        :param word: The word to be added.
        :param docid: Optional identifier of the document from which the words are added.
        :return: The new frequency of the word.
        """
        if docid is not None and docid not in self.dfreqs.keys():
            raise KeyError

        try:
            ind = self.words.index(word)
            self.freqs[ind] += 1
            if docid is not None:
                self.dfreqs[docid][ind] += 1
            return self.freqs[ind]
        except ValueError:
            self.__createEntry(word, docid)
            return 1

    def removeWord(self, word, docid=None):
        """
        Removes a word from the dictionary, i.e., the frequency of the word's entry gets reduced by one and all
        frequency-dependent information for the entry is updated accordingly. If the word's entry has a current
        frequency of 1, the whole entry is deleted. If docid is not None, it must be an ID that has previously been
        registered with the Dictionary object's registerDocument function. Otherwise, it will raise a KeyError.

        :param word: The word to be removed.
        :param docid: Optional identifier of the document from which the words are added.
        """
        if docid is not None and docid not in self.dfreqs.keys():
            raise KeyError

        try:
            ind = self.words.index(word)
        except ValueError:
            return

        if self.freqs[ind] > 1:
            self.freqs[ind] -= 1
            if docid is not None:
                self.dfreqs[docid][ind] -= 1 if self.dfreqs[docid][ind] > 0 else 0
        else:
            self.__deleteEntry(ind)

    def inVocab(self, word) -> bool:
        """
        Checks, whether a given word has an entry in the dictionary (i.e., its frequency is not 0).

        :param word: The word to be looked for.
        :return: True, if an entry exists. Else false.
        """

        if word in self.words:
            return True
        else:
            return False

    def inDoc(self, word, docid) -> bool:
        """
        Checks, whether a given word occurs in the given document (i.e., its frequency in the document is not 0).
        The given docid must be an ID that has previously been registered with the Dictionary object's registerDocument
        function. Otherwise, it will raise a KeyError.

        :param word: The word to be looked for.
        :param docid: The document's identifier.
        :return: True, if an entry exists and the word's frequency in the document is not 0. Else false.
        """

        if docid is not None and docid not in self.dfreqs.keys():
            raise KeyError

        if word in self.words and self.dfreqs[docid][self.words.index(word)] > 0:
            return True
        else:
            return False

    def getGlobalFrequency(self, word):
        """
        Returns the frequency of the word as saved in the word's dictionary entry, or zero if
        no entry for the word exists.

        :param word: The word whose frequency is requested.
        :return: The frequency of the word in the entire dictionary as int.
        """

        try:
            return self.freqs[self.words.index(word)]
        except ValueError:
            return 0

    def getDocFrequency(self, word, docid):
        """
        Returns the frequency of the word in the given document. The given docid must be an ID that has previously been
        registered with the Dictionary object's registerDocument function. Otherwise, it will raise a KeyError.

        :param word: The word whose frequency is requested.
        :param docid: The document's identifier.
        :return: The frequency of the word in the given document as int, or -1, if the word has no dictionary entry at
        all.
        """

        if docid is not None and docid not in self.dfreqs.keys():
            raise KeyError

        try:
            return self.dfreqs[docid][self.words.index(word)]
        except ValueError:
            return -1

    def __createEntry(self, word, docid=None):
        """
        Creates an entry for the given word in the dictionary. All corresponding information like its frequency is
        set or updated. This method is not supposed to be used from a scope outside the Dictionary class. If
        a word should be added to the dictionary, use addWord() instead.

        :param ind: The index of the entry to be deleted.
        :param docid: Optional identifier of the document from which the words are added.
        """
        self.words.append(word)
        self.freqs.append(1)
        for id in self.dfreqs.keys():
            self.dfreqs[id].append(0)
        if docid is not None:
            self.dfreqs[docid][-1] = 1

    def __deleteEntry(self, ind):
        """
        Deletes an entry from the dictionary. This means, the word and all corresponding information like its frequency
        is deleted. This method is not supposed to be used from a scope outside the Dictionary class. If
        a word should be deleted from the dictionary, use removeWord() instead.

        :param ind: The index of the entry to be deleted.
        """

        del self.words[ind]
        del self.freqs[ind]
        for docid in self.dfreqs.keys():
            del self.dfreqs[docid][ind]
