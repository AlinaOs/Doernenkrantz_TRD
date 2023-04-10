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
