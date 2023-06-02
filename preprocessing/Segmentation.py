import re
from rw.rw import *
from nltk import ngrams
import os.path


class TextSegmenter:
    """
    The Text Segmenter provides functions for segmenting text into sentences and/or n-grams.
    """

    '''
    sentbound: Treat insertion markers like sentence boundaries
    ignore: Ignore insertion markers and treat insertions as part of the text surrounding them.
    join: Create two segments: One ignoring the insertion markers and an additional segment ignoring the insertion text.
    '''
    INSERTION = {'sentbound', 'join', 'ignore'}

    def segmentsToList(self, parsegments):
        locatedSegments = []
        for seg in parsegments:
            if str(seg[0]).find('#INSTART#') >= 0:
                text = seg[0].replace('#INPOINT#', '#SEND#')
                sentences = re.findall(r'(?<=#INSTART#).*?(?=#INEND#)', text)
                sentences.append(text)
                sentences.append(re.sub(r'\s*#INSTART#.*?#INEND#\s*', ' ', text))
                for sent in sentences:
                    locsent = [sent]
                    locsent.extend(seg[1])
                    locatedSegments.extend(self.segmentBySentences([locsent], insertion='ignore'))
            else:
                segList = [seg[0]]
                segList += seg[1]
                locatedSegments.append(segList)
        return locatedSegments

    def segmentBySentences(self, innput, output=None, header=None, minwords=4, insertion='join'):
        """
        Segments the text of input into sentences and saves the resulting segments with their corresponding
        text locations to output.
        Sentence boundaries must be marked by '#SEND#' or '#CSTART#' in the input file. If a sentence has less than
        minwords, it is considered part of the sentence before or, if there isn't a sentence before it, the sentence
        after it. If a sentence has one or two words, it is considered an enumeration and gets joined with both the
        previous and next sentence.
        Depending on the value of insertion, insertions will be treated as sentence ends, ignored or added.

        :param innput: An input list or the path to the file containing the list as CSV.
        :param output: The path to where the output shall be written (optional).
        :param header: A list containing the headings of each csv-column. If None, the header will be extracted from the innput csv or will be filled with '?', if the innput is a list.
        :param minwords: Minimal number of words in a sentence.
        :param insertion: Treat insertions as sentence ends ('sentbound'), 'ignore' them, or add a sentence ignoring the insertion ('join').
        """
        csv = None
        if isinstance(innput, str) and os.path.isfile(innput):
            csv = readFromCSV(innput)
            innput = csv['lines']
        elif not isinstance(innput, list):
            return []

        split = r'\s*(?:#SEND#)|(?:#CSTART#)\s*'
        if insertion == r'sentbound':
            split = r'\s*(?:#SEND#)|(?:#CSTART#)|(?:#INSTART#)|(?:#INEND#)\s*'

        sentences = []
        parsentences = []
        pendingsent = None
        location = ['', '', '', '', '']

        for l in innput:
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

            if insertion == 'join':
                intexts = re.findall(r'#INSTART#.+?(?:$|#INEND#)', text)
                closingintexts = re.findall(r'#DNENI#[^I]+(?=$)', text[::-1])
                for c in closingintexts:
                    intexts.append(c[::-1])
                for i in intexts:
                    if i == '':
                        continue
                    newi = re.sub(r'#SEND#', '#INPOINT#', i)
                    text = text.replace(i, newi)

            if insertion == 'ignore':
                text = re.sub(r'\s*(?:#INSTART#|#INEND#)\s*', ' ', text)

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

        if pendingsent is not None:
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

        if output is not None:
            if header is not None:
                writeToCSV(output, sentences, header=header)
            elif csv is not None:
                writeToCSV(output, sentences, header=csv['header'])
            else:
                writeToCSV(output, sentences, header=['?' for i in range(len(sentences[0]))])

        return sentences


    def ngramsToList(self, parngrams):
        locatedSegments = []
        for ngram in parngrams:
            segList = [' '.join(ngram[0])]
            segList += ngram[1]
            locatedSegments.append(segList)
        return locatedSegments

    def segmentByNGrams(self, innput, output, n=5, joinlines=True):
        """
        Reads a CSV file and segments each line into n-grams with a window size
        of n words. When joinlines is set to True (which is the default), line
        ends are ignored and all lines belonging to the same paragraph are considered
        to be one string.

        :param innput: Path to the input CSV.
        :param output: Path to a CSV file for the output.
        :param n: Width of the token window (the n in n-gram).
        :param joinlines: Ignore line ends?
        """

        csv = readFromCSV(innput)

        n_grams = []
        parNgrams = []
        pendinggram = None
        location = ['', '', '', '', '']

        for l in csv['lines']:
            if (l[1] != location[0] or l[2] != location[1]) and pendinggram is not None:
                parNgrams.append(pendinggram)
                pendinggram = None ##

            n_grams.extend(self.ngramsToList(parNgrams))
            parNgrams = []
            location = l[1:]

            text = l[0].strip()
            if text == '':
                continue
            text = re.sub(r'\s*(?:#SEND#|#CSTART#|#INSTART#|#INEND#)\s*', ' ', text)
            words = text.split()

            if pendinggram is not None:
                if pendinggram[0][-1].endswith('-'):
                    gwords = list(pendinggram[0])
                    gwords[-1] = pendinggram[0][-1][:-1] + words.pop(0)
                    pendinggram = (gwords, pendinggram[1])
                parNgrams.append((pendinggram[0], pendinggram[1])) ##

                if len(words) < n:
                    pendingwords = list(pendinggram[0])[1:] + words
                else:
                    pendingwords = list(pendinggram[0])[1:] + words[0:n - 1]

                pendinggrams = list(ngrams(pendingwords, n))

                for i in range(len(pendinggrams)):
                    if i < n:
                        parNgrams.append((pendinggrams[i], pendinggram[1]))
                    else:
                        parNgrams.append((pendinggrams[i], location))

            if len(words) < n:
                if pendinggram is not None:
                    pendinggram = None ##
                    continue
                else:
                    pendinggram = (tuple(words), location) ##
                    continue
            else:
                pendinggram = None ##

            tNgrams = list(ngrams(words, n))

            if joinlines:
                pendinggram = (tNgrams.pop(-1), location) ##

            for ngram in tNgrams:
                parNgrams.append((ngram, location))

        if pendinggram is not None:
            parNgrams.append(pendinggram) ##
        n_grams.extend(self.ngramsToList(parNgrams))
        writeToCSV(output, n_grams, header=csv['header'])

        #TODO insertion for Ngrams?
