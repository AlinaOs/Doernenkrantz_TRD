import gc
import math
import string
import numpy as np
from tools.rw import readDictFromJson, saveDictAsJson, readFromCSV

# Todo: Docstrings

class ScoringMatrix:

    def __init__(self, name: str, init=None):
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

    def __getitem__(self, item: tuple[str, str]):
        i = string.ascii_lowercase.index(item[0])
        j = string.ascii_lowercase.index(item[1])
        return self.SM[i, j]


def favorLeft(lenx, leny, idxx, idxy, score, disfavor=3):
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


def __nwnorm(x, y, nwscore):
    hs = np.max([len(x), len(y)])
    return nwscore / hs


def __favorleftnwnorm(x, y, nwscore, disfavor=0):
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


def checkCombiGap(gappedword: str, fullword: str, gwi: int, fwi: int, left: bool, combilists):
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
    :param combilists: A dictionary containing lists with the following names and contents: "leftfull", a list of
    combinations where the left letter may appear instead of the full combi; "leftalone", a list of lists, each second
    level list containing the possible independent left-standing letters for this combi; "rightfull" and "rightalone",
    the same for right-standing combis/letters; and "vowels", a list of vowels that may be combined together as a combi.
    :return: True, if the strings qualify for a combi-substitution. Otherwise False.
    """

    if left:
        # Is the string long enough to contain a combination?
        if len(fullword) == fwi + 1:
            return False
        full = combilists['leftfull']
        alone = combilists['leftalone']
        combi = fullword[fwi:fwi + 1 + 1]
        vowelcombiidx = 0
    else:
        # Is the given index high enough for the string to contain a combination?
        if fwi == 0:
            return False
        full = combilists['rightfull']
        alone = combilists['rightalone']
        combi = fullword[fwi - 1:fwi + 1]
        vowelcombiidx = 1

    normcombi = combi.replace('y', 'i')
    normcombi = normcombi.replace('j', 'i')

    # Is it a double character combi?
    if normcombi[0] == normcombi[1] and normcombi[0] == gappedword[gwi]:
        return True
    # Is it a double vowel combi?
    if normcombi[0] in combilists['vowels'] and normcombi[1] in combilists['vowels']\
            and normcombi[vowelcombiidx] == gappedword[gwi]:
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


def __nwmatrix(x, y, SM: str = None, gapstart=0.5, gapex=1.0):
    if SM is None:
        SM = ScoringMatrix('standard')

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


def __nwphonmatrix(x, y, SM: str = None, gapstart=1.0, gapex=1.0, disfavor=0, combilists=None):
    if SM is None:
        SM = ScoringMatrix('standard')
    if combilists is None:
        combilists = {"leftfull": [], "leftalone": [], "rightfull": [], "rightalone": [], "vowels": []}

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
            t[0] = F[i - 1, j - 1] + favorLeft(nx, ny, i - 1, j - 1, SM[x[i - 1], y[j - 1]], disfavor=disfavor)

            if SM[x[i - 1], y[j - 1]] >= 0 and (P[i - 1, j - 1] == 'left' or P[i - 1, j - 1] == 'top'):
                combi = checkCombiGap(y, x, j - 1, i - 1, left=False, combilists=combilists) if P[i - 1, j - 1] == 'top' \
                    else checkCombiGap(x, y, i - 1, j - 1, left=False, combilists=combilists)
                if combi:
                    if i >= 2 and P[i - 1, j - 1] == 'top' and \
                            (P[i - 2, j - 1] == 'left' or P[i - 2, j - 1] == 'top'):
                        t[0] += favorLeft(nx, ny, i-2, j-1, gapex, disfavor=disfavor)
                    elif j >= 2 and P[i - 1, j - 1] == 'left' and \
                            (P[i - 1, j - 2] == 'left' or P[i - 1, j - 2] == 'top'):
                        t[0] += favorLeft(nx, ny, i-1, j-2, gapex, disfavor=disfavor)
                    else:
                        t[0] += favorLeft(nx, ny, i-1, j-1, gapstart, disfavor=disfavor)

            leftcombi = False
            if P[i, j - 1] == 'diag' and checkCombiGap(x, y, i - 1, j - 2, left=True, combilists=combilists):
                t[1] = F[i, j - 1]  # combi in y, gapped combi in x
                leftcombi = True
            else:
                t[1] = F[i, j - 1] - favorLeft(nx, ny, i - 1, j - 1, gapex, disfavor=disfavor) if P[i, j - 1] == 'top'\
                    or P[i, j - 1] == 'left' \
                    else F[i, j - 1] - favorLeft(nx, ny, i - 1, j - 1, gapstart, disfavor=disfavor)  # gap in x

            if P[i - 1, j] == 'diag' and checkCombiGap(y, x, j - 1, i - 2, left=True, combilists=combilists):
                t[2] = F[i - 1, j]  # combi in x, gapped combi in y
                leftcombi = True
            else:
                t[2] = F[i - 1, j] - favorLeft(nx, ny, i - 1, j - 1, gapex, disfavor=disfavor) if P[i - 1, j] == 'top'\
                    or P[i - 1, j] == 'left' \
                    else F[i - 1, j] - favorLeft(nx, ny, i - 1, j - 1, gapstart, disfavor=disfavor)  # gap in y

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


# The code for the newu() function was originally copied from the Needleman-Wunsch implementation at
# https://gist.github.com/slowkow/06c6dba9180d013dfd82bec217d22eb5.
# It was broadly restructured and adapted to include a custom scoring matrix and a variable gap penalty. Also, the
# function was changed to return a result containing different types of information instead of aligned strings.
def newu(x, y, SM=None, gapstart=0.5, gapex=1.0, phonetic=False, disfavor=0,
         combilists=None):

    if phonetic:
        F, P = __nwphonmatrix(x, y, SM, gapstart, gapex, disfavor=disfavor, combilists=combilists)
    else:
        F, P = __nwmatrix(x, y, SM, gapstart, gapex)

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

    if phonetic:
        normsim = __favorleftnwnorm(x, y, F[len(x), len(y)], disfavor=disfavor)
    else:
        normsim = __nwnorm(x, y, F[len(x), len(y)])

    result = {
        'sim': F[len(x), len(y)],
        'normsim': normsim,
        'alignx': rx,
        'aligny': ry
    }
    return result


def constructSMfromDict(smdict: dict):
    SM = np.full((26, 26), -1.0)
    for i in range(26):
        SM[i, i] = 1.0
    for k in smdict['pairscores'].keys():
        score = float(k)
        for pair in smdict['pairscores'][k]:
            SM[string.ascii_lowercase.index(pair[1]), string.ascii_lowercase.index(pair[0])] = score
            SM[string.ascii_lowercase.index(pair[0]), string.ascii_lowercase.index(pair[1])] = score
    for k in smdict['doublescores'].keys():
        score = float(k)
        for l in smdict['doublescores'][k]:
            for ll in smdict['doublescores'][k]:
                if l != ll:
                    SM[string.ascii_lowercase.index(l), string.ascii_lowercase.index(ll)] = score
                    SM[string.ascii_lowercase.index(ll), string.ascii_lowercase.index(l)] = score
    return ScoringMatrix(smdict['name'], SM)


# Function used after kind hint from Kelly Bundy at https://stackoverflow.com/a/76599108
def key(s, t):
    return (s, t) if s < t else (t, s)


class Dictionary:
    """
    This class initiates and administrates a dictionary object. It provides functions for accessing,
    manipulating and enhancing the dictionary.
    """

    def __init__(self, initial=None):
        """
        Initiate a new Dictionary instance. If a path to a preexisting dictionary in a serialized form is given, the
        dictionary will be initialized using the data in this foundational dictionary.
        :param initial: Optional path to a previously exported dictionary.
        """

        self.words = []
        self.freqs = []
        self.dfreqs = {}

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
            idx = self.words.index(word)
            self.freqs[idx] += 1
            if docid is not None:
                self.dfreqs[docid][idx] += 1
            return self.freqs[idx]
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
            idx = self.words.index(word)
        except ValueError:
            return

        if self.freqs[idx] > 1:
            self.freqs[idx] -= 1
            if docid is not None:
                self.dfreqs[docid][idx] -= 1 if self.dfreqs[docid][idx] > 0 else 0
        else:
            self.__deleteEntry(idx)

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

        :param docid: Optional identifier of the document from which the words are added.
        """
        self.words.append(word)
        self.freqs.append(1)
        for id in self.dfreqs.keys():
            self.dfreqs[id].append(0)
        if docid is not None:
            self.dfreqs[docid][-1] = 1

    def __deleteEntry(self, idx):
        """
        Deletes an entry from the dictionary. This means, the word and all corresponding information like its frequency
        is deleted. This method is not supposed to be used from a scope outside the Dictionary class. If
        a word should be deleted from the dictionary, use removeWord() instead.

        :param idx: The index of the entry to be deleted.
        """

        del self.words[idx]
        del self.freqs[idx]
        for docid in self.dfreqs.keys():
            del self.dfreqs[docid][idx]
