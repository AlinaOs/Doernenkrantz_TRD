import re
from rw.rw import *


class TextCleaner:

    def cleanTextFromCSV(self, input, output, normalize=None):
        csv = readFromCSV(input)
        for l in csv['lines']:
            l[0] = self.cleanText(l[0], normalize)
        writeToCSV(output, csv['header'], csv['lines'])

    def cleanText(self, text: str, normalize=None):
        """
        Cleans the given text by dealing with punctuation, whitespaces and case.
        If normalize is set to true, simple replacements of some letters/letter combinations will be made to
        normalize the spelling of the text.
        The returned text will be lowercase and punctuation ('.', ':', '!') as well as brackets will be marked
        by '#BOUND#' or '#INSTART#'/'#INEND#' respectively. Chapter beginnings (Capitulum) will be marked
        by '#CSTART#'. Whitespace gets normalized, but trailing and leading whitespaces won't be trimmed.

        :param text: The text to be cleaned.
        :param normalize: Indicates whether the spelling of the text will be normalized by naive replacing.
        :return: The cleaned text.
        """

        # insert space between small letter and big letter
        for m in re.findall(r'[a-zßöů][A-Z]', text):
            text = text.replace(m, m[0]+' '+m[1])

        # lowercase
        text = text.lower()

        # replace points in/around numbers and "/" with whitespace
        text = text.replace('/', ' ')
        for m in re.findall(
                r'(?:^|[.\s]+)[ijvxlcdm]+[.\sijvxlcdm]*(?:$|[.\s]+)',
                text):

            if m == '':
                continue

            rvil = r'(?<=[\s.])v[ij]{1,2}l{1,2}(?=[\s.])'
            rim = r'(?<=[\s.])[ij]{1,2}m(?=[\s.])'
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

        # normalize orthography
        if normalize is not None:
            rules = readDictFromCSV(normalize)
            for rule in rules:
                if rule['exemption'] == '':
                    text = re.sub(rule['regex'], rule['replace'], text)
                else:
                    exemptions = rule['exemption'].split(';')
                    if rule['exemption2'] != '':
                        exemptions2 = rule['exemption2'].split(';')
                    else:
                        exemptions2 = []
                    for m in re.findall(r'(?<=^)|(?<=\s)\S*'+str(rule['regex'])+r'\S*(?=\s|$)', text):
                        if m.strip() == '':
                            continue
                        for e in exemptions:
                            if re.fullmatch(e, m) is None:
                                text = text.replace(m, re.sub(rule['regex'], rule['replace'], m))
                            else:
                                for e2 in exemptions2:
                                    if re.fullmatch(e2, m):
                                        text = text.replace(m, re.sub(rule['regex'], rule['replace'], m))

        # replace punctuation, chapter marks and brackets with boundary markers
        text = text.replace('#', '#CSTART#')
        text = text.replace('⸿', '#CSTART#')
        text = text.replace('.', '#SEND#')
        text = text.replace(':', '#SEND#')
        text = text.replace('!', '#SEND#')
        text = text.replace('(', '#INSTART#')
        text = text.replace(')', '#INEND#')

        # normalize whitespace (no trimming)
        text = re.sub(r'\s+', ' ', text)

        return text