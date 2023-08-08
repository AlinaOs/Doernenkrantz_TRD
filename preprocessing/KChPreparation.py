import math
import re
import xml.etree.ElementTree as ET
from tools.rw import readDictFromCSV

# ocrfilepath = '../fulltexts/KCh_complete_2023-06-04_tei.xml'
ocrfilepath = '../fulltexts/KCh_complete_2023-07-21_tei.xml'
preparedfilepath = '../fulltexts/KCh-completeTest-prepared_Layout.xml'
improvedfilepath = '../fulltexts/KCh-complete_Layout.xml'


def prepareXML():
    with open(ocrfilepath, 'r', encoding='utf-8') as ocrfile,\
            open(preparedfilepath, 'w', encoding='utf-8') as preparedfile:
        nextline = ocrfile.readline()
        while nextline != '':
            test = nextline.strip()
            if not (test.startswith('<l>') or
                    test.startswith('<pb>') or
                    re.match(r'</?p>', test) or
                    re.match(r'</?lg>', test)):
                preparedfile.write(nextline)
                nextline = ocrfile.readline()
                continue
            # Headlines should go all along the line. That also corrects any false nesting of numbers and headlines.
            heading = False
            if nextline.find('<azklein') >= 0:
                heading = 'azklein'
            elif nextline.find('<azgross') >= 0:
                heading = 'azgross'
            if heading:
                nextline = re.sub(r'<'+heading+'[^>]*>', '', nextline)
                nextline = re.sub(r'</'+heading+'>', '', nextline)
                nextline = re.sub(r'<l>', '<l><' + heading + '>', nextline)
                nextline = re.sub(r'</l>', '</' + heading + '></l>', nextline)

            # Resolve abbreviations, rename tags
            replacements = readDictFromCSV('../junk/abbreviations.csv')
            for rule in replacements:
                nextline = re.sub(rule['regex'], rule['replace'], nextline)

            # Save line in new file
            if nextline.strip() != '':
                preparedfile.write(nextline)

            nextline = ocrfile.readline()


tei = '{http://www.tei-c.org/ns/1.0}'
xmlns = '{http://www.w3.org/XML/1998/namespace}'
ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
import string
gatherings = []
gformula = [{'start': 0, 'end': 8, 'lower': False, 'len': 6, 'double': False},
            {'start': 10, 'end': 10, 'lower': False, 'len': 10, 'double': False},
            {'start': 11, 'end': 19, 'lower': False, 'len': 6, 'double': False},
            {'start': 21, 'end': 21, 'lower': True, 'len': 6, 'double': False},
            {'start': 23, 'end': 25, 'lower': True, 'len': 6, 'double': False},
            {'start': 0, 'end': 3, 'lower': True, 'len': 6, 'double': False},
            {'start': 4, 'end': 4, 'lower': True, 'len': 4, 'double': False},
            {'start': 5, 'end': 8, 'lower': True, 'len': 6, 'double': False},
            {'start': 10, 'end': 19, 'lower': True, 'len': 6, 'double': False},
            {'start': 21, 'end': 21, 'lower': True, 'len': 6, 'double': False},
            {'start': 23, 'end': 25, 'lower': True, 'len': 6, 'double': False},
            {'start': 0, 'end': 8, 'lower': True, 'len': 6, 'double': True},
            {'start': 10, 'end': 13, 'lower': True, 'len': 6, 'double': True}]


def fillGatherings():
    for formula in gformula:
        if formula['lower']:
            chars = string.ascii_lowercase
        else:
            chars = string.ascii_uppercase
        for i in range(formula['start'], formula['end'] + 1):
            g = chars[i]
            if formula['double']:
                g = g + g
            gatherings.append((g, formula['len']))


def improveXML():
    pageno = 0
    fillGatherings()
    ET.register_namespace('', "http://www.tei-c.org/ns/1.0")
    tree = ET.parse(preparedfilepath)
    root = tree.getroot()
    for page in root.findall('.//tei:pb', ns):
        # Add folio numbering to pages (not consistent with the often wrong pagination of the original)
        pageno += 1
        fol = str(math.ceil(pageno / 2))
        fol = fol + 'r' if pageno % 2 == 1 else fol + 'v'
        page.set('n', fol)

    lineno = 1
    pageno = 0
    lastelem = None
    divElem = ET.Element('div', {'n': '1'})
    pElem = ET.Element('p', {'n': '1'})
    divElem.append(pElem)
    bodyElem = root.find('.//tei:body', ns)
    possibleHeaders = []
    nextGathering = 0
    nextGatheringBreak = 0

    for elem in bodyElem.findall('./*', ns):
        if elem.tag == tei + 'pb':
            if pageno == nextGatheringBreak:
                newGath = ET.Element('gb', {'n': gatherings[nextGathering][0]})
                pElem.append(newGath)
                nextGatheringBreak += gatherings[nextGathering][1] * 2
                nextGathering += 1
            if lineno < 3:
                for l in possibleHeaders:
                    pElem.append(l)
                    lastelem = l
            elif lineno == 3:
                newlines = checkHeader(possibleHeaders, pageno)
                for l in newlines:
                    pElem.append(l)
                lastelem = None

            possibleHeaders = []
            pageno += 1
            lineno = 1
            if lastelem is not None:
                # Check for gathering signature
                lasttext = lastelem.text if lastelem.text is not None else ''
                if len(lasttext) <= 12 and len(re.sub(r'[ijvuxlcdm.\s]', '', lasttext)) <= 3:
                    lastelem.text = ''
                    newElem = ET.Element('fw', {'type': 'sig'})
                    newElem.text = lasttext
                    lastelem.append(newElem)
                lastelem = None
            pElem.append(elem)
            bodyElem.remove(elem)
        elif lineno <= 3:
            possibleHeaders.append(elem)
            bodyElem.remove(elem)
            lineno += 1
        elif lineno == 4:
            newlines = checkHeader(possibleHeaders, pageno)
            for l in newlines:
                pElem.append(l)

            pElem.append(elem)
            bodyElem.remove(elem)
            lastelem = elem
            lineno += 1
        else:
            pElem.append(elem)
            bodyElem.remove(elem)
            lastelem = elem

    bodyElem.append(divElem)
    with open(improvedfilepath, 'wb') as improvedfile:
        tree.write(improvedfile, xml_declaration=True, encoding='utf-8')


def gettext(node):
    if node.find('./*') is not None and node[0].tag != tei + 'pc':
        text = node[0].text if node[0].text is not None else ''
    else:
        text = node.text if node.text is not None else ''
    return text


def checkHeader(lines, pageno):
    header = [None, None, None]
    texts = [gettext(l) for l in lines]
    hi = [True if l.find('./tei:hi', ns) is not None else False for l in lines]
    if pageno % 2 == 0:
        # verso
        for i in range(len(lines)):
            if header[0] is None and len(texts[i]) <= 12 and len(re.sub(r'[ijvuxlcdm.\s]', '', texts[i])) <= 3:
                header[0] = i
            elif header[1] is None and re.search(r'[Kk]?[ae][ijy]{1,2}[sß]er?', texts[i])\
                    or re.search(r'[rR]oe?mi?sch(e|er)? ?[Kk]?[ou]n[ijy]{1,2}n([gkc]|ck)', texts[i]):
                header[1] = i
            elif header[2] is None and re.search(r'[Kk]?[ou]n[ijy]{1,2}n([gkc]|ck)', texts[i])\
                    or re.search(r'[vVuUfF]ranc?krij?ch', texts[i]):
                header[2] = i
    else:
        # recto
        for i in range(len(lines)):
            if header[0] is None and len(texts[i]) <= 12 and len(re.sub(r'[ijvuxlcdm.\s]', '', texts[i])) <= 3:
                header[0] = i
            elif header[1] is None and re.search(r'[Pp]a[ijy]{1,2}[sß]', texts[i]):
                header[1] = i
            elif header[2] is None and re.search(r'[Bb]?[ijyu]{1,2}ss?ch[ou](ff?|[uv]e?)', texts[i]):
                header[2] = i

    if header[0] is None and header[1] is None and header[2] is None and hi[0]:
        # Default: If no header can be found by buzzword, then the first line will become a header if it is written
        # in a heading font. The other two lines won't be considered as headings.
        header[1] = 0
    elif header[0] == 0 and header[1] is None and header[2] is None and hi[1]:
        # Default: If no header but a pagination can be found by buzzword, then the first line after the pagination will
        # become a header if it is written in a heading font. The third line won't be considered as heading.
        header[1] = 1
    else:
        # Completion: If (e.g., for OCR-reasons) only one line was recognized as header, then all lines before that line
        # become headings too.
        last = 0
        for h in header:
            if h is not None and h > last:
                last = h
        for i in range(last):
            if i not in header and hi[i]:
                lines[i][0].tag = tei + 'fw'
                lines[i][0].set('type', 'header')
                lines[i][0].set('rendition', '#TW15.175G')
            if i not in header and not hi[i]:
                text = lines[i].text if lines[i].text is not None else ''
                lines[i].text = ''
                newHi = ET.Element('fw', {'type': 'header', 'rendition': '#TW15.175G'})
                newHi.text = text
                lines[i].append(newHi)

    # assign formwork tags to headings
    if header[0] is not None:
        if hi[header[0]]:
            lines[header[0]][0].tag = tei + 'fw'
            lines[header[0]][0].set('type', 'pagination')
            lines[header[0]][0].set('rendition', '#TW15.175G')
        else:
            text = lines[header[0]].text if lines[header[0]].text is not None else ''
            lines[header[0]].text = ''
            newHi = ET.Element('fw', {'type': 'pagination', 'rendition': '#TW15.175G'})
            newHi.text = text
            lines[header[0]].append(newHi)
    if header[1] is not None:
        if hi[header[1]]:
            lines[header[1]][0].tag = tei + 'fw'
            lines[header[1]][0].set('type', 'header')
            lines[header[1]][0].set('rendition', '#TW15.175G')
        else:
            text = lines[header[1]].text if lines[header[1]].text is not None else ''
            lines[header[1]].text = ''
            newHi = ET.Element('fw', {'type': 'header', 'rendition': '#TW15.175G'})
            newHi.text = text
            lines[header[1]].append(newHi)
    if header[2] is not None:
        if hi[header[2]]:
            lines[header[2]][0].tag = tei + 'fw'
            lines[header[2]][0].set('type', 'header')
            lines[header[2]][0].set('rendition', '#TW15.175G')
        else:
            text = lines[header[2]].text if lines[header[2]].text is not None else ''
            lines[header[2]].text = ''
            newHi = ET.Element('fw', {'type': 'header', 'rendition': '#TW15.175G'})
            newHi.text = text
            lines[header[2]].append(newHi)

    return lines


prepareXML()
improveXML()
