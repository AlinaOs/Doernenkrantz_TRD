import string
import numpy as np
from tools.rw import readDictFromJson, saveDictAsJson, readFromCSV


class Dictionary:
    """
    This class initiates and administrates a dictionary object. It provides functions for accessing,
    manipulating and enhancing the dictionary.
    """

    words = []
    freqs = []

    def __init__(self, initial=None):
        """
        Initiate a new Dictionary instance. If a path to a preexisting dictionary in a serialized form is given, the
        dictionary will be initialized using the data in this foundational dictionary.
        :param initial: Path to a serialized dictionary.
        """

        if initial is not None:
            dictionary = readDictFromJson(initial)
            for key in dictionary:
                self.words.append(key)
                self.freqs.append(dictionary[key])

    def updateDict(self, text):
        """
        Use a given text to update the dictionary, i.e., to add new words and/or increase the frequency of already
        registered words. The text will be tokenized according to whitespaces and each token will be considered a word
        to add to the dictionary. No additional text cleaning or normalization is applied.

        :param text: The text that contains the words to be added.
        """

        tokens = text.split()
        for token in tokens:
            token = token.lower()
            self.addWord(token)

    def serialize(self, output, sort='freq'):
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

        if sort == 'word':
            li = sorted(list(zip(self.words, self.freqs)))
        elif sort == 'freq':
            li = sorted(list(zip(self.words, self.freqs)), key=lambda e: e[1], reverse=True)
        else:
            li = list(zip(self.words, self.freqs))

        dictionary = {}
        for w in li:
            dictionary[w[0]] = w[1]

        saveDictAsJson(output, dictionary)

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

    def getWordFrequency(self, word):
        """
        Returns the frequency of the word as saved in the word's dictionary entry, or zero if
        no entry for the word exists.

        :param word: The word whose frequency is requested.
        :return: The frequency as int.
        """

        try:
            return self.freqs[self.words.index(word)]
        except ValueError:
            return 0

    def addWord(self, word):
        """
        Adds a word to the dictionary, i.e., an entry is created for the word and the frequency
        of the entry is set to 1 or increased by 1, if an entry already exists. All frequency-dependent information
        for the entry is updated accordingly.

        :param word: The word to be added.
        :return: The new frequency of the word.
        """

        try:
            ind = self.words.index(word)
            self.freqs[ind] += 1
            return self.freqs[ind]
        except ValueError:
            self.words.append(word)
            self.freqs.append(1)
            return 1

    def removeWord(self, word):
        """
        Removes a word from the dictionary, i.e., the frequency of the word's entry gets reduced by one and all
        frequency-dependent information for the entry is updated accordingly. If the word's entry has a current
        frequency of 1, the whole entry is deleted.

        :param word: The word to be removed.
        """

        try:
            ind = self.words.index(word)
        except ValueError:
            return

        if self.freqs[ind] > 1:
            self.freqs[ind] -= 1
        else:
            self.__deleteEntry(ind)

    def __deleteEntry(self, ind):
        """
        Deletes an entry from the dictionary. This means, the word and all corresponding information like its frequency
        is deleted. This method is not supposed to be used from a scope outside the Dictionary class. If
        a word should be deleted from the dictionary, use removeWord() instead.

        :param ind: The index of the entry to be deleted.
        """

        self.words.pop(ind)
        self.freqs.pop(ind)


class ScoringMatrix:

    def __init__(self, init, name):
        if init is not None and isinstance(init, str):
            self.SM = np.asarray(readFromCSV(init, False))
        elif init is not None and isinstance(init, list):
            self.SM = np.asarray(init)
        else:
            self.SM = np.full((26, 26), -1)
            for i in range(26):
                self.SM[i, i] = 1

        self.name = name

        pos = np.copy(self.SM)
        pos[pos < 0] = 1
        self.am = np.average(pos)

    def __getitem__(self, item: tuple[str, str]):
        i =  string.ascii_lowercase.index(item[0])
        j = string.ascii_lowercase.index(item[1])
        return self.SM[i,j]

    def name(self):
        return self.name

    def identify(self, name):
        return True if self.name == name else False

    def averagematch(self):
        return self.am


class StringSimilarity():

    SM = {
        'standard': ScoringMatrix(None, 'standard')
    }

    def registerSM(self, SM: ScoringMatrix):
        self.SM[SM.name()] = SM

    # The code for the newu() function was originally copied from the Needleman-Wunsch implementation at
    # https://gist.github.com/slowkow/06c6dba9180d013dfd82bec217d22eb5.
    # It was broadly restructured and adapted to include a custom scoring matrix and a variable gap penalty. Also, the
    # function was changed to return a result containing different types of information instead of aligned strings.
    def newu(self, x, y, SM:str=None, gapstart=0.5, gapex=1):
        if SM is not None and SM not in self.SM.keys():
            return None

        F, P = self.__nwmatrix(x, y, SM, gapstart, gapex)

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
            elif P[i, j] == 'left':
                rx.append('-')
                ry.append(y[j-1])
                j -= 1
            elif P[i, j] == 'top':
                rx.append(x[i-1])
                ry.append('-')
                i -= 1

        if i > 0:
            ry = ry + list(np.full(i, '-'))
            while i > 0:
                rx.append(x[i-1])
                i -= 1
        else:
            rx = rx + list(np.full(j, '-'))
            while j > 0:
                ry.append(y[j-1])
                j -= 1
        
        # Reverse the strings.
        rx = ''.join(rx)[::-1]
        ry = ''.join(ry)[::-1]

        am = self.SM['standard'].averagematch() if SM is None else self.SM[SM].averagematch()

        result = {
            'sim': F[len(x), len(y)],
            'normsim': self.__nwnorm(x, y, F[len(x), len(y)], gapstart, gapex, am),
            'alignx': rx,
            'aligny': ry,
            'config': {'gapstart': gapstart, 'gapex': gapex, 'sm': 'standard'}
        }
        return result

    def __nwmatrix(self, x, y, SM:str=None, gapstart=0.5, gapex=1):
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
        P[:, 0] = 'left'
        P[0, :] = 'up'

        # Temporary scores.
        t = np.zeros(3)
        for i in range(1, nx + 1):
            for j in range(1, ny + 1):
                t[0] = F[i - 1, j - 1] + SM[x[i - 1], y[j - 1]]
                t[1] = F[i, j - 1] - gapstart if P[i, j - 1] == 2 else F[i, j - 1] - gapex
                t[2] = F[i - 1, j] - gapstart if P[i - 1, j] == 2 else F[i - 1, j] - gapex

                tmax = np.max(t)
                F[i, j] = tmax

                if t[0] == tmax:
                    P[i, j] = 'diag'
                if t[1] == tmax:
                    P[i, j] = 'left'
                if t[2] == tmax:
                    P[i, j] = 'top'

        return F, P

    def __nwnorm(self, x, y, nwscore, gapstart=0.5, gapex=1, averagematch=1):
        hs = np.min([len(x), len(y)]) * averagematch
        rest = abs(len(x) - len(y))

        if rest > 0:
            hs -= gapstart
            rest -= 1
            for r in range(rest):
                hs -= gapex

        return nwscore / hs

    def commonStemma(self, x, y) -> str:
        # Todo: Return the common stemma of x and y
        pass

    def stemma(self, stem, word) -> bool:
        # Todo: Check, whether the stem may be the stemma of the given word
        pass
