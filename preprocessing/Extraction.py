import math
from rw.rw import *


class TextExtractor:

    tei = '{http://www.tei-c.org/ns/1.0}'
    xmlns = '{http://www.w3.org/XML/1998/namespace}'
    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}

    def joinLines(self, lines, join='#lb#'):
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

    currPage = 0
    currFol = ''
    currGath = ''
    currGathNo = 0

    def resetGathering(self):
        self.currPage = 0
        self.currFol = ''
        self.currGath = ''
        self.currGathNo = 0

    def getGathering(self):
        no = math.ceil(self.currGathNo / 2)
        if self.currGathNo % 2 == 0:
            side = 'v'
        else:
            side = 'r'
        return ''.join([self.currGath, str(no), side])

    def extractText(self, xmlPath, outputPath, metaOutputPath):
        self.resetGathering()
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

        writeToCSV(outputPath, pagesText, header=[
            'text',
            'bookpart',
            'paragraph',
            'page',
            'fol',
            'gathering'
        ])
        writeToCSV(metaOutputPath, metaText, header=[
            'text',
            'bookpart',
            'paragraph',
            'page',
            'fol',
            'gathering'
        ])
