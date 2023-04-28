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
                                # re-add syllabification marker if a word is broken over two pages
                                li = -1
                                if len(pageLines[-1].findall('./' + self.tei + 'fw[@type="sig"]')) > 0:
                                    li = -2
                                if len(pageLines[li].findall('./' + self.tei + 'pc[@type="syllabification"]')) > 0:
                                    joined += '-'

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
                        # re-add syllabification marker if a word is broken over two pages
                        li = -1
                        if len(pageLines[-1].findall('./' + self.tei + 'fw[@type="sig"]')) > 0:
                            li = -2
                        if len(pageLines[li].findall('./' + self.tei + 'pc[@type="syllabification"]')) > 0:
                            joined += '-'

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


class TextSegmenter:
    INSERTION = {'sentbound', 'join', 'ignore'}
    """
    sentbound: Treat insertion markers like sentence boundaries
    ignore: Ignore insertion markers and treat insertions as part of the text surrounding them.
    join: Create two segments: One ignoring the insertion markers and an additional segment ignoring the insertion text.
    """

    def segmentsToList(self, parsegments):
        locatedSegments = []
        for seg in parsegments:
            segList = [seg[0]]
            segList += seg[1]
            locatedSegments.append(segList)
        return locatedSegments

    def segmentBySentences(self, innput, output, minwords=4, insertion='join'):
        """
        Segments the text of input into sentences and saves the resulting segments with their corresponding
        text locations to output.
        Sentence boundaries must be marked by '#SEND#' or '#CSTART#' in the input file. If a sentence has less than
        minwords, it is considered part of the sentence before or, if there isn't a sentence before it, the sentence
        after it. If a sentence has one or two words, it is considered an enumeration and gets joined with both the
        previous and next sentence.
        Depending on the value of insertion, insertions will be treated as sentence ends, ignored or added.

        :param innput: The path to the input file.
        :param output: The path to where the output shall be written.
        :param minwords: Minimal number of words in a sentence.
        :param insertion: Treat insertions as sentence ends ('sentbound'), 'ignore' them, or add a sentence ignoring the insertion ('join').
        """
        csv = readFromCSV(innput)

        split = r'\s*(?:#SEND#)|(?:#CSTART#)\s*'
        if insertion == r'sentbound':
            split = r'\s*(?:#SEND#)|(?:#CSTART#)|(?:#INSTART#)|(?:#INEND#)\s*'

        sentences = []
        parsentences = []
        pendingsent = None
        location = ['', '', '', '', '']

        for l in csv['lines']:
            if (l[1] != location[0] or l[2] != location[1]) and pendingsent is not None:
                if int(pendingsent[1]) < minwords and len(parsentences) > 0:
                    last = parsentences.pop()
                    leftover = ' '.join([last[0], pendingsent[0].strip()])
                    if pendingsent[1] > len(last[0].split()):
                        location = pendingsent[2]
                    else:
                        location = last[1]
                else:
                    leftover = pendingsent[0].strip()

                parsentences.append((leftover, location))
                pendingsent = None

            sentences.extend(self.segmentsToList(parsentences))
            parsentences = []
            location = l[1:]

            text = l[0].strip()
            if text == '':
                continue
            if insertion == 'ignore':
                re.sub(r'\s*(?:#INSTART#|#INEND#)\s*', ' ', text)

            if insertion == 'join':
                # TODO: Check insertion
                pass

            text = re.split(split, text.strip())
            for sent in text:
                sentloc = location
                sent = sent.strip()
                if sent == '':
                    continue

                words = sent.split()
                if pendingsent is not None:
                    if pendingsent[0].endswith('-'):
                        sent = pendingsent[0].strip()[0:-1] + sent
                    else:
                        sent = pendingsent[0].strip() + ' ' + sent
                    if pendingsent[1] > len(words):
                        sentloc = pendingsent[2]
                    pendingsent = None

                words = sent.split()
                # Check for minimal length
                if len(words) < minwords:
                    if len(parsentences) > 0:
                        if len(words) < 3:
                            # less than 3 words -> enumeration, connect to previous AND next sentence
                            newpend = parsentences.pop()[0] + ' ' + sent
                            pendingsent = (newpend, len(newpend), location)
                        else:
                            parsentences.append((parsentences.pop()[0]+' '+sent, sentloc))
                    else:
                        pendingsent = (sent, len(words), location)
                else:
                    parsentences.append((sent, sentloc))

            # Check for sentences overlapping the page
            if pendingsent is None and re.search(r'#SEND#\.*$', l[0]) is None:
                sent = parsentences.pop()[0]
                words = len(sent.split())
                pendingsent = (sent, words, location)

        if int(pendingsent[1]) < minwords and len(parsentences) > 0:
            last = parsentences.pop()
            leftover = ' '.join([last[0], pendingsent[0].strip()])
            if pendingsent[1] > len(last[0].split()):
                location = pendingsent[2]
            else:
                location = last[1]
        else:
            leftover = pendingsent[0].strip()

        parsentences.append((leftover, location))
        sentences.extend(self.segmentsToList(parsentences))

        writeToCSV(output, csv['header'], sentences)

    def segmentByNGrams(self, innput, output, n=5, insertion='join', ignoresentbounds=True):
        """


        :param innput:
        :param output:
        :param n: Width of the token window (the n in n-gram).
        :param insertion:
        :param ignoresentbounds: May n-grams overlap sentence boundaries?
        :return:
        """
        pass


# TODO Prepare TRD input files
