import math
import os.path
import re

from tools.rw import *


class TextExtractor:

    tei = '{http://www.tei-c.org/ns/1.0}'
    xmlns = '{http://www.w3.org/XML/1998/namespace}'
    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
    
    def __init__(self):
        self.currPage = 0
        self.currFol = ''
        self.currGath = ''
        self.currGathNo = 0
        self.opensegs = {}

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
                    joiner = '='
                elif elem.tag == self.tei + 'figure' or elem.tag == self.tei + 'fw':
                    # ignore figures and form work
                    pass
                elif elem.tag == self.tei + 'seg' and elem.attrib.get('type') == 'tr':
                    if elem.attrib.get('next') is None and elem.attrib.get('prev') is None:
                        segid = elem.attrib.get(self.xmlns + 'id')
                        seginfo = f'_{segid}'
                        corresp = elem.attrib.get('corresp')
                        if corresp is not None:
                            seginfo += f'_{elem.attrib.get("subtype")}'
                            seginfo += f'_{corresp.replace("#", "").replace(" ", "&")}'
                        li += f'$segstart{seginfo}$' + self.joinLines([elem], join='') + f'$segend_{segid}$'
                    elif elem.attrib.get('next') is not None and elem.attrib.get('prev') is not None:
                        segid = elem.attrib.get(self.xmlns + 'id')
                        previd = elem.attrib.get('prev')[1:]
                        self.opensegs[segid] = self.opensegs[previd]
                        del self.opensegs[previd]
                        li += self.joinLines([elem], join='')
                        if elem.tail is not None:
                            li += elem.tail
                    elif elem.attrib.get('prev') is None:
                        segid = elem.attrib.get(self.xmlns + 'id')
                        self.opensegs[segid] = segid
                        seginfo = f'_{segid}'
                        corresp = elem.attrib.get('corresp')
                        if corresp is not None:
                            seginfo += f'_{elem.attrib.get("subtype")}'
                            seginfo += f'_{corresp.replace("#", "").replace(" ", "&")}'
                        li += f'$segstart{seginfo}$' + self.joinLines([elem], join='')
                    elif elem.attrib.get('next') is None:
                        previd = elem.attrib.get('prev')[1:]
                        segid = self.opensegs[previd]
                        li += self.joinLines([elem], join='') + f'$segend_{segid}$'
                        del self.opensegs[previd]
                        if elem.tail is not None:
                            li += elem.tail
                else:
                    # ignore all other markup (e.g., ex, supplied, hi, etc.) and extract text
                    li += self.joinLines([elem], join='')
                    if elem.tail is not None:
                        li += elem.tail
            li = ''.join(li.splitlines())
            if li != '':
                li = li.strip()
                if not li.endswith('='):
                    li = li + joiner
                cleanLines.append(li)
        joinedLines = ''.join(cleanLines)
        joinedLines = re.sub(r'=(?!$)', '', joinedLines)
        return joinedLines

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
        root = getXMLRoot(xmlPath)
        pagesText, metaText = self._extractTextFromRoot(root)

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

    def extractGoldText(self, xmlPath, outputdir):
        root = getXMLRoot(xmlPath)
        for goldtext in root.findall('./tei:TEI', self.ns):
            textid = goldtext.attrib.get(self.xmlns + 'id')

            locstart = goldtext.find('.//tei:citedRange', self.ns).attrib.get('from')
            locstart = locstart.split('-')
            self.currPage = int(locstart[0])
            self.currFol = locstart[1]
            self.currGath = locstart[2]
            self.currGathNo = int(locstart[3])

            pagesText, metaText = self._extractTextFromRoot(goldtext)

            writeToCSV(os.path.join(outputdir, textid + '_gold_pages.csv'), pagesText, header=[
                'text',
                'bookpart',
                'paragraph',
                'page',
                'fol',
                'gathering'
            ])
            writeToCSV(os.path.join(outputdir, textid + '_gold_meta.csv'), metaText, header=[
                'text',
                'bookpart',
                'paragraph',
                'page',
                'fol',
                'gathering'
            ])

    def _extractTextFromRoot(self, root:ET.Element):
        metaText = []
        pagesText = []
        header = {}

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

        return pagesText, metaText
