import re

from tools.rw import readDictFromJson, saveDictAsJson


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
