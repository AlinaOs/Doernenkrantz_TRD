import gzip
import os
import re
import shutil

from tools.rw import readDictFromJson, saveDictAsJson, readFromCSV, readFromTxt


def blastify(innput, outdir, title, docid=None, filename=None, gzipdir=None):
    """
    Formats the given input and saves it as a txt-file compatible as BLAST input. In addition, each file is saved as
    gzipped file in the gzipdir.
    :param innput: A path to a csv- or txt-file containing the input text. If it is a cs-file, each line must
    contain the text as first element. All other elements will be used as location information.
    :param outdir: The path of the directory to which the blastified output should be written.
    :param title: The title of the document.
    :param docid: The prefix used for document ids. Each id consists of the prefix and the location information of the
    document (if any is given). If docid is None, title will be taken as prefix.
    :param filename: The name of the output file. If None, the name of the input file will be used.
    :param gzipdir: The path of the directory to which the gzipped output should be written. If None, a new directory
    named 'gzipped' is created in outdir and used as gzipdir.
    """

    ext = os.path.splitext(innput)[-1]
    if ext == '.txt':
        lines = [[readFromTxt(innput)]]
    elif ext == '.csv':
        lines = readFromCSV(innput)['lines']
    else:
        raise ValueError

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    if filename is None:
        filename = os.path.split(innput)[1][0:-4]
    filename += '.txt'

    if docid is None:
        docid = title

    with open(os.path.join(outdir, filename), 'w', newline='', encoding='utf-8') as txtfile:
        for l in lines:
            location = '-' + '-'.join(l[1:]) if len(l) > 1 else ''
            currdocid = docid + location
            text = l[0]
            text = text.replace('#SEND#', ' ')
            text = text.replace('#CSTART#', ' ')
            text = text.replace('#INSTART#', '')
            text = text.replace('#INEND#', ' ')
            text = text.replace('#lb#', ' ')
            text = text.replace('-', ' ')
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            js = '{"title": "' + title + '", "doc_id": "' + currdocid + '", "text": "' + text.strip() + '"}\n'
            txtfile.write(js)

    if gzipdir is None:
        gzipdir = os.path.join(outdir, 'gzipped')
    if not os.path.exists(gzipdir):
        os.mkdir(gzipdir)
    with open(os.path.join(outdir, filename), 'rb') as f_in:
        with gzip.open(os.path.join(gzipdir, str(filename) + '.gz'), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def passimify(innput, outdir, series, docid=None, filename=None):
    """
    Formats the given input and saves it as a json-file compatible as passim input
    :param innput: A path to a csv- or txt-file containing the input text. If it is a cs-file, each line must
    contain the text as first element. All other elements will be used as location information.
    :param outdir: The path of the directory to which the passimified output should be written.
    :param series: The title of the document.
    :param docid: The prefix used for document ids. Each id consists of the prefix and the location information of the
    document (if any is given). If docid is None, title will be taken as prefix.
    :param filename: The name of the output file. If None, the name of the input file will be used.
    """

    ext = os.path.splitext(innput)[-1]
    if ext == '.txt':
        lines = [[readFromTxt(innput)]]
    elif ext == '.csv':
        lines = readFromCSV(innput)['lines']
    else:
        raise ValueError

    if docid is None:
        docid = series
    if filename is None:
        filename = os.path.split(innput)[1][0:-4]
    filename += '.json'

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    with open(os.path.join(outdir, filename), 'w', newline='', encoding='utf-8') as txtfile:
        for l in lines:
            location = '-' + '-'.join(l[1:]) if len(l) > 1 else ''
            currdocid = docid + location
            text = l[0]
            text = text.replace('#SEND#', ' ')
            text = text.replace('#CSTART#', ' ')
            text = text.replace('#INSTART#', '')
            text = text.replace('#INEND#', ' ')
            text = text.replace('#lb#', ' ')
            text = text.replace('-', ' ')
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            js = '{"id": "' + currdocid + '", "series": "' + series + '", "text": "' + text.strip() + '"}\n'
            txtfile.write(js)


def tpairify(innput, outdir, metadata, name, title, year, author, prefix=None):
    """
    Formats the given input and saves it as a txt-file compatible as BLAST input. In addition, each file is saved as
    gzipped file in the gzipdir.
    :param innput: A path to a csv- or txt-file containing the input text. If it is a cs-file, each line must
    contain the text as first element. All other elements will be used as location information.
    :param outdir: The path of the directory to which the blastified output should be written.
    :param title: The title of the document.
    :param prefix: The prefix used for the names of the new files. Each filename consists of the prefix and the location
    information of the document (if any is given). If prefix is None, the name of the input file will be used as prefix.
    """

    ext = os.path.splitext(innput)[-1]
    if ext == '.txt':
        lines = [[readFromTxt(innput)]]
    elif ext == '.csv':
        lines = readFromCSV(innput)['lines']
    else:
        raise ValueError

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    if prefix is None:
        prefix = os.path.split(innput)[1][0:-4]

    with open(os.path.join(metadata), 'a', encoding='utf-8') as mdfile:
        for l in lines:
            location = '-'.join(l[1:]) if len(l) > 1 else ''
            currdocid = prefix + '-' + location if location != '' else prefix
            filename = currdocid + '.txt'
            text = l[0]
            text = text.replace('#SEND#', ' ')
            text = text.replace('#CSTART#', ' ')
            text = text.replace('#INSTART#', '')
            text = text.replace('#INEND#', ' ')
            text = text.replace('#lb#', ' ')
            text = text.replace('-', ' ')
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            with open(os.path.join(outdir, filename), 'w', newline='', encoding='utf-8') as txtfile:
                txtfile.write(text)
            # Metadata:
            # filename,name,year,author,title,location
            md = ','.join([filename, name, year, author, title, location]) + '\n'
            mdfile.write(md)


def blastifyDir(indir, outdir, filenames, ext='.csv'):
    """
    Performs a blastification for all the files in indir that match the filenames and extension given. The output
    is written to outdir.
    :param indir: The path of the directory containing the input files.
    :param outdir: The path of the directory to which the blastified output should be written.
    :param filenames: A list of filenames (without extension) to be blastified.
    :param ext: The extension of the files.
    """

    if not os.path.exists(outdir):
        os.mkdir(outdir)
    for f in filenames:
        title = f.split('_')[0]
        docid = f.split('.')[0]
        blastify(
            os.path.join(indir, f + ext),
            outdir,
            title,
            docid=docid
        )


def passimifyDir(indir, outdir, filenames, ext='.csv'):
    """
    Performs a passimification for all the files in indir that match the filenames and extension given. The output
    is written to outdir.
    :param indir: The path of the directory containing the input files.
    :param outdir: The path of the directory to which the passimified output should be written.
    :param filenames: A list of filenames (without extension) to be passimified.
    :param ext: The extension of the files.
    """

    if not os.path.exists(outdir):
        os.mkdir(outdir)
    for f in filenames:
        series = f.split('_')[0]
        docid = f.split('.')[0]
        passimify(
            os.path.join(indir, f + ext),
            outdir,
            series,
            docid=docid
        )


def tpairifyDir(indir, outdir, filenames, ext='.csv'):
    """
    Performs a tpairification for all the files in indir that match the filenames and extension given. The output
    is written to outdir.
    :param indir: The path of the directory containing the input files.
    :param outdir: The path of the directory to which the tpairified output should be written.
    :param filenames: A list of filenames (without extension) to be tpairified.
    :param ext: The extension of the files.
    """

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    mdsource = os.path.join(outdir, 'source_md.csv')
    mdtarget = os.path.join(outdir, 'target_md.csv')
    with open(mdsource, 'w', newline='', encoding='utf-8') as mdf:
        mdf.write('filename,name,year,author,title,location\n')
    with open(mdtarget, 'w', newline='', encoding='utf-8') as mdf:
        mdf.write('filename,name,year,author,title,location\n')

    outsrc = os.path.join(outdir, 'source')
    outtrg = os.path.join(outdir, 'target')
    if not os.path.exists(outsrc):
        os.mkdir(outsrc)
    if not os.path.exists(outtrg):
        os.mkdir(outtrg)

    for f in filenames:
        name = f.split('_')[0]
        if name == 'Dk':
            year = '1490'
            author = 'Anonymus'
            title = 'Der Doernenkrantz van Collen'
            outdir = outsrc
            mdfile = mdsource
        else:
            year = '1499'
            author = 'Anonymus'
            title = 'Die Cronica van der hilliger Stat van Collen'
            outdir = outtrg
            mdfile = mdtarget

        tpairify(
            os.path.join(indir, f + ext),
            outdir,
            mdfile,
            name,
            title,
            year,
            author
        )


def filterBlastClusters(innput, outdir, fname=None):
    """
    Filters BLAST output clusters to only contain clusters with hits from different documents (hits within the same
    document are still included, if the hit cluster contains at least one hit of a different document as well). The
    function uses the 'title'-key of the BLAST output to determine, which hit belongs to which document. If the value
    of 'title' contains underscores, only the part before the first underscore is considered as title-string. For
    example: 'Dk_full_pages' would become 'Dk', hence 'Dk_full_pages' and 'Dk_some_pages' would be considered identical.
    The filtered clusters are written to a new file in the outdir.
    :param innput: The path to the cluster to be filtered.
    :param outdir: The folder to where the filtered clusters should be written.
    :param fname: The filename of the new filtered clusters. If None, the original filename will have the suffix
    '-filtered'.
    """

    if fname is None:
        bn = os.path.basename(innput)
        bn = '.'.join(bn.split('.')[0:-1])
        fname = os.path.join(outdir, bn + '-filtered.json')
    cluster = readDictFromJson(innput)
    newcluster = {}
    for k in cluster.keys():
        titles = set()
        for hit in cluster[k]['hits']:
            titles.add(hit['title'].split('_')[0])
        if len(titles) > 1:
            newcluster[k] = cluster[k]
    saveDictAsJson(os.path.join(outdir, fname), newcluster)
