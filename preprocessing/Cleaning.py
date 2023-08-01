import re
from tools.rw import *
from tools.lang import Dictionary
from py3langid.langid import LanguageIdentifier, MODEL_FILE


class TextCleaner:
    lidentifier = LanguageIdentifier.from_pickled_model(MODEL_FILE, norm_probs=True)
    lidentifier.set_languages(['la', 'de'])
    standardCSnorm = []
    standardLCnorm = []
    prefixes = []

    def __init__(self, standardCSnorm, standardLCnorm, prefixes, vocab: Dictionary):
        """
        Initializes and returns a TextCleaner.

        :param standardCSnorm: Path to CSV file containing normalization rules that are case-sensitive.
        :param standardLCnorm: Path to CSV file containing normalization rules that are case-insensitive (lowercase).
        :param prefixes: Path to CSV file containing a list of prefixes that should be checked for word-joining.
        :param vocab: Dictionary object for the initiation of a vocabulary.
        """
        self.standardCSnorm = readDictFromCSV(standardCSnorm) if standardCSnorm is not None else []
        self.standardLCnorm = readDictFromCSV(standardLCnorm) if standardLCnorm is not None else []
        self.prefixes = readDictFromCSV(prefixes) if prefixes is not None else []
        self.vocab = vocab

    def cleanTextFromCSV(self, input, output, normalize=True, lowercase=False, addnorm=None, docid=None):
        csv = readFromCSV(input)
        if addnorm is not None:
            addnorm = readDictFromCSV(addnorm)
        delete = []
        for i in range(len(csv['lines'])):
            l = csv['lines'][i]
            l[0] = self.cleanText(l[0], normalize, lowercase, addnorm, docid)
            l[0] = self.joinlineends(l[0], docid)
            if normalize:
                l[0] = self.joinprefixes(l[0], docid)
                l[0] = l[0].strip()
            if l[0] == '':
                delete.append(i)
        for i in sorted(delete, reverse=True):
            csv['lines'].pop(i)
            
        # Join words hyphenated over two pages
        brokenword = False
        for i in range(len(csv['lines'])):
            if brokenword:
                tokens = csv['lines'][i][0].split()
                csv['lines'][i - 1][0] = ''.join([csv['lines'][i - 1][0], tokens[0]])
                self.vocab.removeWord(
                    re.sub('#SEND#|#CSTART#|#INSTART#|#INEND#|#lb#', '',
                    re.sub(r'\w+#lb#$', '',
                          re.sub(r'\w+#lb#\w+', '', tokens[0]))), docid)
                tokens.pop(0)
                csv['lines'][i][0] = ' '.join(tokens)
                lasttokens = csv['lines'][i - 1][0].split()
                lasttokens[-1] = self.cleanText(lasttokens[-1], normalize, lowercase, addnorm, docid)
                self.vocab.addWord(
                    re.sub('#SEND#|#CSTART#|#INSTART#|#INEND#|#lb#', '',
                           re.sub(r'\w+#lb#$', '',
                                  re.sub(r'\w+#lb#\w+', '', lasttokens[-1]))), docid)
                brokenword = False
            if csv['lines'][i][0].strip().endswith('-'):
                brokenword = True
                tokens = csv['lines'][i][0].split()
                self.vocab.removeWord(
                    re.sub('#SEND#|#CSTART#|#INSTART#|#INEND#|#lb#', '',
                    re.sub(r'\w+#lb#$', '',
                          re.sub(r'\w+#lb#\w+', '', tokens[-1]))), docid)
                csv['lines'][i][0] = csv['lines'][i][0].replace('-', '')

        writeToCSV(output, csv['lines'], header=csv['header'])

    def cleanText(self, text: str, normalize=True, lowercase=False, addnorm=None, docid=None):
        """
        Cleans the given text by dealing with punctuation, whitespaces and case. If normalize is set to True (default),
        standard normalization will be performed based on the normalization rules that were initiated with the
        TextCleaner object. If the function parameter addnorm is not None, the given normalization rules will be
        performed additionally (this will happen after the standard normalization, if normalize is set to True).
        The returned text will be lowercase and punctuation ('.', ':', '!') as well as brackets will be
        marked by '#SEND#' or '#INSTART#'/'#INEND#' respectively. Chapter beginnings (Capitulum) will be marked
        by '#CSTART#'. Whitespace gets normalized, but trailing and leading whitespaces won't be trimmed.

        :param text: The text to be cleaned.
        :param normalize:
        :param addnorm: Further rules for normalization as list or path to a CSV file containing the rules.
        :return: The cleaned text.
        """

        # insert space between small letter and big letter
        for m in re.findall(r'[a-zßöů][A-Z]', text):
            text = text.replace(m, m[0]+' '+m[1])

        # Delete whitespaces, where the following token is a single word character and therefore most likely
        # cut off from the previous token due to OCR problems or poor print quality.
        # Note: Some characters are excluded because they could indeed represent a word or roman number on their own.
        text = re.sub(r'(?<=\w)\s(?=[befghknopqrstwyz]\W)', '', text)

        # replace points in/around numbers and "/" with whitespace
        text = text.replace('/', ' ')
        for m in re.findall(
                r'(?:^|[.\s]+)[ijvxlcdmIJVXLCDM]+[.\sijvxlcdmIJVXLCDM]*(?:$|[.\s]+)',
                text):

            if m == '':
                continue

            rvil = r'(?<=[\s.])[vV][ij]{1,2}l{1,2}(?=[\s.])'
            rim = r'(?<=[\s.])[ijIJ]{1,2}m(?=[\s.])'
            if len([f for f in re.findall(rvil, m) if f != '']) != 0:
                # The regex also matches substrings such as 'vil' and 'im' that are no numbers
                # Those matches need to be ignored.
                old = re.sub(rvil, '', m)
            elif len([f for f in re.findall(rim, m) if f != '']) != 0:
                old = re.sub(rim, '', m)
            else:
                old = m

            if old.strip() == '':
                continue

            text = text.replace(old, ' ' + old.replace('.', '').replace(' ', '') + ' ')

        if normalize:
            text = self.normalizeText(text)
        elif lowercase:
            text = text.lower()

        # apply individual normalizations given as function parameter
        if addnorm is not None:
            if isinstance(addnorm, list):
                rules = addnorm
            else:
                rules = readDictFromCSV(addnorm)
            for rule in rules:
                text = re.sub(rule['regex'], rule['replace'], text)

        # replace punctuation, chapter marks and brackets with boundary markers
        text = re.sub(r'(?<!#lb)#(?!lb#)', '#CSTART#', text)
        # text.replace('#', '#CSTART#')
        text = text.replace('⸿', '#CSTART#')
        text = text.replace('.', '#SEND#')
        text = text.replace(':', '#SEND#')
        text = text.replace('!', '#SEND#')
        text = text.replace('?', '#SEND#')
        text = text.replace('(', '#INSTART#')
        text = text.replace(')', '#INEND#')

        # normalize whitespace (no trimming)
        text = re.sub(r'\s+', ' ', text)
        self.vocab.updateDict(
            re.sub('#SEND#|#CSTART#|#INSTART#|#INEND#|#lb#', ' ',
                   re.sub(r'\w+#lb#$', '',
                          re.sub(r'\w+#lb#\w+', '', text))), docid
        )

        return text

    def normalizeText(self, text):
        # apply normalization rules that need case-sensitivity
        for rule in self.standardCSnorm:
            text = re.sub(rule['regex'], rule['replace'], text)

        # lowercase
        text = text.lower()

        # apply normalization rules that aren't case-sensitive (lowercase)
        for rule in self.standardLCnorm:
            text = re.sub(rule['regex'], rule['replace'], text)

        return text

    def joinprefixes(self, text, docid=None):
        """
        Tests, whether a lonely prefix has to be joined with the following word.
        The prefixes that are considered are those initiated with the TextCleaner object and the method used for
        checking whether two words should be joined is the class's joinwords method.
        :param text: The text, that needs checking for prefix joins.
        :return: The updated text.
        """
        for pre in self.prefixes:
            candidates = re.findall(pre['regex'], text)
            if len(candidates) > 0:
                text = self.joinwords(text, candidates, ' ', invocab=True, joinright=True, docid=docid)
        return text

    def joinlineends(self, text, docid=None):
        """
        Tests, whether tokens separated by linebreaks (marked as "#lb" in the
        input text) are actually only one word. If yes, then the tokens get joined.
        If not, they stay separate words and the linebreak marker is replaced by a
        whitespace character. The method used for
        checking whether two words should be joined is the class's joinwords method.

        :param text: The input text, where linebreaks are marked as '#lb#'.
        :return: The text with all linebreaks either removed or replaced by a whitespace.
        """
        text = re.sub(r'(?<=\W)#lb#|#lb#(?=\W)', ' ', text)
        text = re.sub(r'\s#lb#$', '#lb#', text)
        text = re.sub(r'^#lb#\s?', '', text)
        text = re.sub(r'(?<=\s)#lb#|#lb#(?=\s)', ' ', text)
        candidates = re.findall(r'\w+#lb#\w+', text)
        text = self.joinwords(text, candidates, '#lb#', docid=docid)
        if re.search(r'\w+#lb#\w+', text):
            # There exists a line that consists only of one word. this word is therefore encased in
            # linebreaks and the second linebreak still needs to be checked.
            text = self.joinlineends(text, docid=docid)
        text = re.sub(r'\s+', ' ', text)
        return text

    def joinwords(self, text, candidates, separator, joinright=False, invocab=False, docid=None):
        """
        Tests whether two tokens (the candidate pair) form one single word and should therefore be joined.
        The algorithm works naively and linguistically uninformed, but is based on the
        vocabulary saved as class variable of the TextCleaner. It applies the following
        rules (t1 = token 1, t2= token 2, j = both tokens joined, v = vocabulary):

        1. not t1 and not t2 and not j: seperate
        2. not t1 and not t2 and j: join
        3. t1 xor t2 and not j: join, if the non-existing token has 3 or less characters, else separate
        4. t1 xor t2 and j: join, if freq(j) > freq(t), else separate
        5. t1 and t2 and j: join, if freq(j) > (freq(t1) + freq(t2))/2, else separate
        6. t1 and t2 and not j: separate

        The performance is circa 40% recall and circa 96-99% precision for token pairs forming
        a single word.

        This behaviour can be changed by setting joinright to True (default: False). Thereby, only the second (i.e., the
        right) token of a candidate pair is crucial in deciding whether to join or not. The candidate pair gets joined
        if 1. the joined word exists in the vocabulary, 2. the second join candidate token has more than three
        characters, and 3. the joined word is more frequent than the second token on its own.

        After joining or separating the tokens, the class's vocabulary gets updated.

        :param text: The input text.
        :param candidates: Possible candidates for joining preselected with some reasonable rule.
        :param separator: The string that marks the point of separation between the two tokens of each candidate.
        :param joinright: Indicates, whether only the second token of a candidate pair is crucial for joining.
        :param invocab: Indicates whether the tokens of a candidate pair are already accounted for in the vocabulary.
        :return: The text where each candidate pair of tokens is either joined or separated by a whitespace.
        """
        for c in candidates:
            j = False
            joined = self.normalizeText(str(c).replace(separator, ''))
            separated = self.normalizeText(str(c).replace(separator, ' '))
            separated = separated.split()

            if len(separated) == 1:
                # During normalization, the separated words get joined.
                # -> The candidate pair would be joined anyway.
                j = True
                separated = str(c).split(separator)

            noj = self.vocab.getGlobalFrequency(joined.lower())
            nosep1 = self.vocab.getGlobalFrequency(separated[0].lower())
            nosep2 = self.vocab.getGlobalFrequency(separated[1].lower())

            if invocab:
                nosep1 = nosep1 - 1 if nosep1 and nosep1 - 1 > 0 else False
                nosep2 = nosep2 - 1 if nosep2 and nosep2 - 1 > 0 else False

            if not j:
                # Even after normalization, the separated words would still be separated. Decide, whether to join
                # or not to join.
                if joinright:
                    j = True if noj and len(separated[1]) > 3 and nosep2 < noj else False
                else:
                    if not noj and not nosep1 and not nosep2:
                        # 1
                        j = False
                    elif noj and not nosep1 and not nosep2:
                        # 2
                        j = True
                    elif not noj and nosep1 and nosep2:
                        # 6
                        j = False
                    elif noj and (nosep1 and not nosep2) or (not nosep1 and nosep2):
                        # 4
                        septext = separated[0] if nosep1 else separated[1]
                        lg = self.lidentifier.classify(' '.join([septext for i in range(6)]))
                        if lg[0] == 'la' and lg[1] > 0.8:
                            j = False
                        else:
                            sep = nosep1 if nosep1 else nosep2
                            j = True if noj > sep else False
                    elif not noj and (nosep1 and not nosep2) or (not nosep1 and nosep2):
                        # 3
                        septext = separated[0] if nosep1 else separated[1]
                        lg = self.lidentifier.classify(' '.join([septext for i in range(6)]))
                        if lg[0] == 'la' and lg[1] > 0.8:
                            j = False
                        else:
                            sep = separated[0] if nosep2 else separated[1]
                            j = True if len(sep) <= 3 else False
                    elif noj and nosep1 and nosep2:
                        # 5
                        lg1 = self.lidentifier.classify(' '.join([separated[0] for i in range(6)]))
                        lg2 = self.lidentifier.classify(' '.join([separated[1] for i in range(6)]))
                        if (lg1[0] == 'la' and lg1[1] > 0.8) or (lg2[0] == 'la' and lg2[1] > 0.8):
                            j = False
                        else:
                            j = True if noj > (nosep1 + nosep2)/2 else False

            if j:
                text = text.replace(c, joined)
                self.vocab.addWord(joined.lower(), docid)
                if invocab:
                    self.vocab.removeWord(nosep1, docid)
                    self.vocab.removeWord(nosep2, docid)

            else:
                text = text.replace(c, ' '.join(separated))
                if not invocab:
                    self.vocab.addWord(separated[0].lower(), docid)
                    self.vocab.addWord(separated[1].lower(), docid)
        return text
