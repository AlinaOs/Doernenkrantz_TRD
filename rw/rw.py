import xml.etree.ElementTree as ET
import csv


def getXMLRoot(path):
    tree = ET.parse(path)
    root = tree.getroot()
    return root


def writeToCSV(path, header, lines):
    with open(path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(lines)


def readFromCSV(path, header=True):
    h = None
    lines = []
    with open(path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        if header:
            h = next(reader, None)
        while (n := next(reader, None)) is not None:
            lines.append(n)

    return {'header': h, 'lines': lines}


def readDictFromCSV(path):
    h = None
    lines = []
    with open(path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            lines.append(row)

    return lines
