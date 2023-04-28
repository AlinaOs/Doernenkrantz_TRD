import re
from rw.rw import *


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