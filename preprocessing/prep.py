import math
import re
from rw.rw import *


class TextExtractor:

    tei = '{http://www.tei-c.org/ns/1.0}'
    xmlns = '{http://www.w3.org/XML/1998/namespace}'
    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}

    def joinLines(self, lines, join=' '):
        cleanLines = []
        for line in lines:
            if line.text is not None:
                li = line.text
            else:
                li = ''
            joiner = join
            for elem in line.findall('./*'):
                if elem.tag == self.tei + 'choice':
                    # take reg/corr instead of orig
                    li += self.joinLines([elem.find('./tei:reg', self.ns)], join='')
                elif elem.tag == self.tei + 'pc' and elem.attrib.get('type') == 'syllabification':
                    # join syllabified words
                    joiner = ''
                elif elem.tag == self.tei + 'figure' or elem.tag == self.tei + 'fw':
                    # ignore figures and form work
                    pass
                else:
                    # ignore all other markup (e.g., ex, supplied, hi, etc.) and extract text
                    if elem.text is not None:
                        # li += elem.text
                        li += self.joinLines([elem], join='')
                    if elem.tail is not None:
                        li += elem.tail
            li = ''.join(li.splitlines())
            if li != '':
                li += joiner
                cleanLines.append(li)
        return ''.join(cleanLines)

    ################
    # DOERNENKRANTZ
    ################

    currPage = 0
    currFol = ''
    currGath = ''
    currGathNo = 0

    def getGathering(self):
        no = math.ceil(self.currGathNo / 2)
        if self.currGathNo % 2 == 0:
            side = 'v'
        else:
            side = 'r'
        return ''.join([self.currGath, str(no), side])

    def extractDkText(self, xmlPath, outputPath, metaOutputPath):
        metaText = []
        pagesText = []
        header = {}
        root = getXMLRoot(xmlPath)
        for div in root.findall('.//tei:div', self.ns):
            bookpart = div.attrib.get('n')

            for p in div.findall('./tei:p', self.ns):
                paragraph = p.attrib.get('n')
                pageLines = []
                for elem in p.findall('./*'):

                    if elem.tag == self.tei + 'l':
                        if elem.find('./tei:fw[@type = "header"]', self.ns) is not None:
                            # Extract page headers from xml
                            head = elem.find('./tei:fw', self.ns)
                            if head.attrib.get('next'):
                                if head.attrib.get('next')[1:] in header:
                                    h = self.joinLines([head, header[head.attrib.get('next')[1:]]])
                                else:
                                    header[head.attrib.get(self.xmlns + 'id')] = head
                                    continue
                            elif head.attrib.get('prev'):
                                if head.attrib.get('prev')[1:] in header:
                                    h = self.joinLines([header[head.attrib.get('prev')[1:]], head])
                                else:
                                    header[head.attrib.get(self.xmlns + 'id')] = head
                                    continue
                            else:
                                h = self.joinLines([head])

                            metaText.append([
                                h,
                                bookpart,
                                paragraph,
                                self.currPage,
                                self.currFol,
                                self.getGathering()
                            ])

                        else:
                            # Extract main text from xml
                            pageLines.append(elem)

                    elif elem.tag == self.tei + 'pb':
                        # Change page, save page text
                        if pageLines:
                            joined = self.joinLines(pageLines)
                            if joined != '':
                                pagesText.append([
                                    joined,
                                    bookpart,
                                    paragraph,
                                    self.currPage,
                                    self.currFol,
                                    self.getGathering()
                                ])
                            pageLines.clear()

                        self.currFol = elem.attrib.get('n')
                        self.currPage += 1
                        self.currGathNo += 1
                        continue

                    elif elem.tag == self.tei + 'gb':
                        self.currGath = elem.attrib.get('n')
                        self.currGathNo = 0
                        continue

                    elif elem.tag == self.tei + 'note':
                        # Extract woodcut text from xml
                        quotes = elem.findall('./tei:quote', self.ns)
                        lines = []
                        if quotes:
                            for q in quotes:
                                ls = q.findall('./tei:l', self.ns)
                                if ls:
                                    lines.extend(ls)
                                else:
                                    lines.append(q)
                            metaText.append([
                                self.joinLines(lines),
                                bookpart,
                                paragraph,
                                self.currPage,
                                self.currFol,
                                self.getGathering()
                            ])

                # Join page text
                if pageLines:
                    joined = self.joinLines(pageLines)
                    if joined != '':
                        pagesText.append([
                            joined,
                            bookpart,
                            paragraph,
                            self.currPage,
                            self.currFol,
                            self.getGathering()
                        ])
                    pageLines.clear()

        writeToCSV(outputPath, [
            'text',
            'bookpart',
            'paragraph',
            'page',
            'fol',
            'gathering'
        ], pagesText)

        writeToCSV(metaOutputPath, [
            'text',
            'bookpart',
            'paragraph',
            'page',
            'fol',
            'gathering'
        ], metaText)


# ALL TEXTS ###
class TextCleaner:

    def cleanTextFromCSV(self, inputpath, outputpath, normalize=None):
        csv = readFromCSV(inputpath)
        for l in csv['lines']:
            l[0] = self.cleanText(l[0], normalize)
        writeToCSV(outputpath, csv['header'], csv['lines'])

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


class TextSegmenter:
    pass
    # TODO Segment text

# TODO Prepare TRD input files
