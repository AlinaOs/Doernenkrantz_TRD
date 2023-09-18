import configparser
import gzip
import lz4.frame as lz4
import os.path
import shutil
import sys

from trd.formatting import blastifyDir, passimifyDir, tpairifyDir, filterBlastClusters, reformatPassimOutput

if sys.platform.startswith('linux'):
    sys.path.append('/mnt/c/Projects/BLAST/textreuse-blast')
    blastpath = '/mnt/c/Projects/BLAST/textreuse-blast/run_full.py'
else:
    sys.path.append('C:\\Projects\\BLAST\\textreuse-blast')
    blastpath = 'C:\\Projects\\BLAST\\textreuse-blast\\run_full.py'

import subprocess
import time


def runblastFull(indir, workingdir, outdir=None, e_value=0.001, word_size=6, language='RIP', min_length=0,
                 max_length=100000, renamecf=False):
    """
    Convenience function for running BLAST with preparation and postprocessing. Internally,
    runblastWithPreparation() is called with the given parameters and the resulting json files in
    workingdir/output/clusters/filled are decompressed, filtered to exclude document-internal hits, and saved in
    outdir.

    :param indir: Path to the directory containing the input data.
    :param workingdir: Path to the directory that the tool will use for execution. All formatted input files will be
        created here as well as all output files.
    :param outdir: The directory to where the post-processed output file will be written. Defaults to
        workingdir/output/results.
    :param e_value: The e_value as float. Influences, how long or short parallels must be to be recognized as match.
    :param word_size: The number of characters constituting a 'word' for BLAST. Must be between 2 and 7.
    :param language: The language to be used for data encoding.
    :param min_length: Minimum length of hits to clusterize. Decreasing this wilĺ not make the program a lot faster, as
        BLAST will still find these and they are just ignored in the clusterizer part.
    :param max_length: Maximum length of hits to clusterize.
    :param renamecf: Whether to give the filtered result files a new name. Default: False, so all original
        cluster files will be overwritten by the decompressed, filtered ones.
    :return: A dictionary containing the configuration details ('config') and the time it took the tool to run ('time').
    """

    blastout = os.path.join(workingdir, 'output/clusters/filled')
    if outdir is None:
        outdir = blastout

    rtrn = runblastWithPreparation(indir, workingdir, e_value, word_size, language, min_length, max_length)

    clusterfiles = os.listdir(blastout)
    for cf in clusterfiles:
        # Unzip the cluster files, filter the clusters and save them in the outdir
        cname = os.path.basename(cf)[:-3] + '.json'
        cpath = os.path.join(outdir, cname)
        with gzip.open(os.path.join(blastout, cf), 'rb') as f_in:
            with open(cpath, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        if renamecf:
            filterBlastClusters(cpath, outdir)
        else:
            filterBlastClusters(cpath, outdir, cname)

    return rtrn


def runblastWithPreparation(indir, workingdir, e_value=0.001, word_size=6, language='RIP', min_length=0,
                            max_length=100000):
    """
    Convenience function for running BLAST with preparation. The input data is converted to BLAST's own input
    format and saved in the directory workingdir/injson (uncompressed) workingdir/data (compressed). BLAST's output can
    be found in workingdir/output.

    :param indir: Path to the directory containing the input data.
    :param workingdir: Path to the directory that the tool will use for execution. All formatted input files will be
        created here as well as all output files.
    :param e_value: The e_value as float. Influences, how long or short parallels must be to be recognized as match.
    :param word_size: The number of characters constituting a 'word' for BLAST. Must be between 2 and 7.
    :param language: The language to be used for data encoding.
    :param min_length: Minimum length of hits to clusterize. Decreasing this wilĺ not make the program a lot faster, as
        BLAST will still find these and they are just ignored in the clusterizer part.
    :param max_length: Maximum length of hits to clusterize.
    :return: A dictionary containing the configuration details ('config') and the time it took the tool to run ('time').
    """

    data_folder = os.path.join(workingdir, 'data')
    output_folder = os.path.join(workingdir, 'output')
    injson_folder = os.path.join(workingdir, 'injson')
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if not os.path.exists(injson_folder):
        os.mkdir(injson_folder)

    infiles = os.listdir(indir)
    blastifyDir(indir, injson_folder, infiles, ext='.txt', gzipdir=data_folder)

    return runblast(data_folder, output_folder, e_value, word_size, language, min_length, max_length)


def runblast(data_folder, output_folder, e_value=0.001, word_size=6, language='RIP', min_length=0, max_length=100000):
    """
    Runs BLAST with the given parameters.

    :param data_folder: Path to the directory containing the input formatted according to BLAST's requirements.
    :param output_folder: Path to the directory where BLAST's output will be written.
    :param e_value: The e_value as float. Influences, how long or short parallels must be to be recognized as match.
    :param word_size: The number of characters constituting a 'word' for BLAST. Must be between 2 and 7.
    :param language: The language to be used for data encoding.
    :param min_length: Minimum length of hits to clusterize. Decreasing this wilĺ not make the program a lot faster, as
        BLAST will still find these and they are just ignored in the clusterizer part.
    :param max_length: Maximum length of hits to clusterize.
    :return: A dictionary containing the configuration details ('config') and the time it took the tool to run ('time').
    """

    timestart = time.time()

    cmd = ['python3', blastpath,
           '--data_folder', data_folder,
           '--output_folder', output_folder,
           '--e_value', str(e_value),
           '--word_size', str(word_size),
           '--language', language,
           '--min_length', str(min_length),
           '--max_length', str(max_length)
           ]

    subprocess.run(cmd, check=True)

    timeend = time.time()

    return {'config': {
        'e_value': e_value,
        'word_size': word_size,
        'language': language,
        'min_length': min_length,
        'max_length': max_length,
    }, 'time': timeend - timestart}


def runpassimFull(indir, workingdir, outdir=None, n=25, min_match=5, maxDF=100, floating_ngrams=False,
                  beam=20, min_align=50, pcopy=0.8, all_pairs=False):
    """
    Convenience function for running Passim with preparation and postprocessing. Internally,
    runpassimWithPreparation() is called with the given parameters and the resulting json file in
    workingdir/output/out.json is reformatted and saved as outdir/out.json.

    :param indir: Path to the directory containing the input data.
    :param workingdir: Path to the directory that the tool will use for execution. All formatted input files will be
        created here as well as all output files.
    :param outdir: The directory to where the post-processed output file will be written. Defaults to
        workingdir/output/results.
    :param n: Number of characters (n) in each character n-gram used for TRD.
    :param min_match: Minimum number of n-gram matches between documents for the documents to be compared. Default: 5.
    :param maxDF: Upper limit on ngram document frequency to be considered during detection. Default: 100.
    :param floating_ngrams: Whether ngrams can only begin at word boundaries.
    :param beam: Number of optimal alignments kept in memory during beam search.
    :param min_align: Minimum number of characters in an aligned passage to be considered a match.
    :param pcopy: Probability of a character being copied from the source to the target passage.
    :param all_pairs: Whether to compute alignments for all pairs.
    :return: A dictionary containing the configuration details ('config') and the time it took the tool to run ('time').
    """

    passimout = os.path.join(workingdir, 'output/out.json')
    if outdir is None:
        outdir = passimout

    rtrn = runpassimWithPreparation(indir, workingdir, n, min_match, maxDF, floating_ngrams,
                                    beam, min_align, pcopy, all_pairs)

    cf = [of for of in os.listdir(passimout) if of.endswith('.json')][0]

    # Reformat and save the file in the outdir
    cpath = os.path.join(outdir, 'out.json')
    reformatPassimOutput(os.path.join(passimout, cf), cpath)

    return rtrn


def runpassimWithPreparation(indir, workingdir, n=25, min_match=5, maxDF=100, floating_ngrams=False,
                             beam=20, min_align=50, pcopy=0.8, all_pairs=False):
    """
    Convenience function for running Passim with preparation. The input data is converted to Passim's own input
    format and saved in the directory workingdir/input. Passim's output can be found in workingdir/output.

    :param indir: Path to the directory containing the input data.
    :param workingdir: Path to the directory that the tool will use for execution. All formatted input files will be
        created here as well as all output files.
    :param n: Number of characters (n) in each character n-gram used for TRD.
    :param min_match: Minimum number of n-gram matches between documents for the documents to be compared. Default: 5.
    :param maxDF: Upper limit on ngram document frequency to be considered during detection. Default: 100.
    :param floating_ngrams: Whether ngrams can only begin at word boundaries.
    :param beam: Number of optimal alignments kept in memory during beam search.
    :param min_align: Minimum number of characters in an aligned passage to be considered a match.
    :param pcopy: Probability of a character being copied from the source to the target passage.
    :param all_pairs: Whether to compute alignments for all pairs.
    :return: A dictionary containing the configuration details ('config') and the time it took the tool to run ('time').
    """
    inputfiles = os.path.join(workingdir, 'input')
    outdir = os.path.join(workingdir, 'output')

    if not os.path.exists(inputfiles):
        os.mkdir(inputfiles)
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    infiles = os.listdir(indir)
    passimifyDir(indir, inputfiles, infiles, ext='.txt')

    return runpassim(inputfiles, outdir, n, min_match, maxDF, floating_ngrams, beam, min_align,
                     pcopy, all_pairs)


def runpassim(indir, outdir, n=25, min_match=5, maxDF=100, floating_ngrams=False, beam=20, min_align=50,
              pcopy=0.8, all_pairs=False):
    """
    Runs Passim with the given parameters.

    :param indir: Path to the directory containing the input formatted according to Passim's requirements.
    :param outdir: Path to the directory where Passim's output will be written.
    :param n: Number of characters (n) in each character n-gram used for TRD.
    :param min_match: Minimum number of n-gram matches between documents for the documents to be compared. Default: 5.
    :param maxDF: Upper limit on ngram document frequency to be considered during detection. Default: 100.
    :param floating_ngrams: Whether ngrams can only begin at word boundaries.
    :param beam: Number of optimal alignments kept in memory during beam search.
    :param min_align: Minimum number of characters in an aligned passage to be considered a match.
    :param pcopy: Probability of a character being copied from the source to the target passage.
    :param all_pairs: Whether to compute alignments for all pairs.
    :return: A dictionary containing the configuration details ('config') and the time it took the tool to run ('time').
    """
    
    timestart = time.time()

    cmd = ['passim',
           indir,
           outdir,
           '--n', str(n),
           '-m', str(min_match),
           '--maxDF', str(maxDF),
           '-g', '200',
           '--beam', str(beam),
           '-a', str(min_align),
           '--pcopy', str(pcopy)
           ]

    if floating_ngrams:
        cmd.append('--floating-ngrams')
    if all_pairs:
        cmd.append('--all-pairs')

    subprocess.run(cmd, check=True)

    timeend = time.time()

    return {'config': {
        'n': str(n),
        'min_match': str(min_match),
        'floating_ngrams': str(floating_ngrams),
        'beam': str(beam),
        'min_align': str(min_align),
        'maxDF': maxDF,
        'pcopy': pcopy,
        'all_pairs': all_pairs
    }, 'time': timeend - timestart}


def runtextpairFull(indir, workingdir, outdir=None, ngram=3, gap=0, matching_window_size=30,
                    minimum_matching_ngrams_in_docs=4, minimum_matching_ngrams_in_window=4,
                    max_gap=15, minimum_matching_ngrams=4, store_banalities=False, word_order=False):
    """
    Convenience function for running TextPAIR with preparation and postprocessing. Internally,
    runTextPAIRWithPreparation() is called with the given parameters and the resultfile
    workingdir/output/results/alignments.jsonl.lz4 is decompressed and saved as outdir/alignments.jsonl.

    :param indir: Path to the directory containing the input data.
    :param workingdir: Path to the directory that the tool will use for execution. All formatted input files will be
        created here as well as all output files.
    :param outdir: The directory to where the post-processed output file will be written. Defaults to
        workingdir/output/results.
    :param ngram: Number of tokens in an n-gram.
    :param gap: Size of gap allowed in n-grams.
    :param matching_window_size: Size of ngram window to be initially evaluated in the sequence aligner.
    :param minimum_matching_ngrams_in_docs: Minimum number of shared ngrams between docs to start a comparison.
    :param minimum_matching_ngrams_in_window: Minimum number of matching ngrams in ngram window
    :param max_gap: Maximum gap authorized between matching ngrams
    :param minimum_matching_ngrams: Minimum number of matching ngrams to constitute a match
    :param store_banalities: Whether to store banalities in a different file than all other alignments. Setting
        this to True might impact the evaluation results.
    :param word_order: Whether to respect word order.
    :return: A dictionary containing the configuration details ('config') and the time it took the tool to run ('time').
    """

    tpout = os.path.join(workingdir, 'output/results')
    if outdir is None:
        outdir = tpout

    rtrn = runtextpairWithPreparation(indir, workingdir, ngram, gap, matching_window_size,
                                      minimum_matching_ngrams_in_docs, minimum_matching_ngrams_in_window,
                                      max_gap, minimum_matching_ngrams, store_banalities, word_order)

    # Unzip the result file, save it in the outdir
    rname = 'alignments.jsonl'
    rpath = os.path.join(outdir, rname)
    with lz4.open(os.path.join(tpout, rname + '.lz4'), 'rb') as f_in:
        with open(rpath, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    return rtrn


def runtextpairWithPreparation(indir, workingdir, ngram=3, gap=0, matching_window_size=30,
                               minimum_matching_ngrams_in_docs=4, minimum_matching_ngrams_in_window=4,
                               max_gap=15, minimum_matching_ngrams=4, store_banalities=False, word_order=False):
    """
    Convenience function for running TextPAIR with preparation. The input data is converted to TextPAIR's own input
    format and saved in the directory workingdir/input, which contains the following subfolders and files:
        - workingdir/input/source: Containing all Dk input files.
        - workingdir/input/target: Containing all KCh input files.
        - workingdir/input/source_md.csv: The metadata of the source files.
        - workingdir/input/target_md.csv: The metadata of the target files.
        - workingdir/input/config.ini: The configuration file. The file is created by this function and contains all
            TextPAIR parameters as given in the function call as well as the paths to the source and target files.
    TextPAIR's output can be found in workingdir/output.

    :param indir: Path to the directory containing the input data.
    :param workingdir: Path to the directory that the tool will use for execution. All formatted input files will be
        created here as well as all output files.
    :param ngram: Number of tokens in an n-gram.
    :param gap: Size of gap allowed in n-grams.
    :param matching_window_size: Size of ngram window to be initially evaluated in the sequence aligner.
    :param minimum_matching_ngrams_in_docs: Minimum number of shared ngrams between docs to start a comparison.
    :param minimum_matching_ngrams_in_window: Minimum number of matching ngrams in ngram window
    :param max_gap: Maximum gap authorized between matching ngrams
    :param minimum_matching_ngrams: Minimum number of matching ngrams to constitute a match
    :param store_banalities: Whether to store banalities in a different file than all other alignments. Setting
        this to True might impact the evaluation results.
    :param word_order: Whether to respect word order.
    :return: A dictionary containing the configuration details ('config') and the time it took the tool to run ('time').
    """

    inputdir = os.path.join(workingdir, 'input')
    outdir = os.path.join(workingdir, 'output')

    if not os.path.exists(inputdir):
        os.mkdir(inputdir)
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    dockermount = '/TextPAIR_run'

    infiles = os.listdir(indir)
    infiles = [f for f in infiles if f.endswith('.txt')]
    tpairifyDir(indir, inputdir, infiles, ext='.txt')

    config = configparser.ConfigParser()

    config['TEXT_SOURCES'] = {
        'source_file_path': os.path.join(dockermount, 'input', 'source'),
        'source_metadata': os.path.join(dockermount, 'input', 'source_md.csv'),
        'source_url': '',
        'target_file_path': os.path.join(dockermount, 'input', 'target'),
        'target_metadata': os.path.join(dockermount, 'input', 'target_md.csv'),
        'target_url': ''
    }
    config['TEXT_PARSING'] = {'parse_source_files': 'yes',
                              'source_file_type': 'plain_text',
                              'source_words_to_keep': 'all',
                              'parse_target_files': 'yes',
                              'target_file_type': 'plain_text',
                              'target_words_to_keep': 'all'}

    config['PREPROCESSING'] = {'source_text_object_type': 'doc',
                               'target_text_object_type': 'doc',
                               'ngram': str(ngram),
                               'gap': str(gap),
                               'word_order': 'yes' if word_order else 'no',
                               'language': 'de',
                               'target_language': '',
                               'modernize': 'no',
                               'ascii': 'no',
                               'stemmer': 'yes',
                               'lemmatizer': '',
                               'lowercase': 'yes',
                               'numbers': 'yes',
                               'minimum_word_length': '2',
                               'stopwords': '',
                               'pos_to_keep': '',
                               'text_object_definition': 'n_token',
                               'text_object_type_split': 'doc',
                               'min_text_object_length': '10',
                               'n_chunk': '3',
                               'vectorization': 'tfidf',
                               'min_freq': '0.05',
                               'max_freq': '0.9',
                               'model_name': ''}

    config['MATCHING'] = {'matching_algorithm': 'sa',
                          'sort_by': 'year',
                          'source_batch': '1',
                          'target_batch': '1',
                          'context_size': '100',
                          'matching_window_size': str(matching_window_size),
                          'minimum_matching_ngrams_in_docs': str(minimum_matching_ngrams_in_docs),
                          'duplicate_threshold': '100',
                          'minimum_matching_ngrams_in_window': str(minimum_matching_ngrams_in_window),
                          'max_gap': str(max_gap),
                          'minimum_matching_ngrams': str(minimum_matching_ngrams),
                          'flex_gap': 'true',
                          'min_similarity': '0.5',
                          'similarity_metric': 'cosine',
                          'min_matching_words': '5',
                          'merge_passages_on_byte_distance': 'true',
                          'passage_distance_multiplier': '0.5',
                          'merge_passages_on_ngram_distance': 'true',
                          'banality_auto_detection': 'true',
                          'most_common_ngram_proportion': '0.1',
                          'common_ngram_threshold': '90',
                          'store_banalities': str(store_banalities).lower(),
                          'phrase_filter': ''}

    with open(os.path.join(inputdir, 'config.ini'), 'w') as configfile:
        config.write(configfile)

    return runtextpair(os.path.abspath(workingdir), dockermount, 'output', 'input/config.ini')


def runtextpair(basedir, dockermount, outdir, configpath):
    """
    Runs TextPAIR in a Docker container in the shell. All parameters impacting the text reuse detection must be
    contained in the config.ini indicated by configpath.

    :param basedir: The directory that will be mounted as docker volume, containing all input and output files.
    :param dockermount: The path to which the basedir will be mounted. This path must be absolute.
    :param outdir: The path for the TextPAIR output (relative to basedir).
    :param configpath: The path to the TextPAIR config (relative to basedir). All paths in config must have the form
        [dockermount]/[path relative to basedir].
    :return: A dictionary containing the configuration details ('config') and the time it took the tool to run ('time').
    """

    dockerconf = os.path.join(dockermount, configpath)
    dockerout = os.path.join(dockermount, outdir)

    timestart = time.time()

    cmd = ['docker', 'run',
           '--rm',
           '-v', f'{basedir}:{dockermount}',
           'artfl/TextPAIR',
           'sh', '-c',
           'init_TextPAIR_db; TextPAIR '
           f'--config={dockerconf} '
           f'--output_path={dockerout} '
           '--skip_web_app '
           'TextPAIR'
           ]

    subprocess.run(cmd, check=True)

    timeend = time.time()
    config = configparser.ConfigParser()
    config.read(os.path.join(basedir, configpath))

    return {'config': {
        'ngram': config['PREPROCESSING']['ngram'],
        'gap': config['PREPROCESSING']['gap'],
        'matching_window_size': config['MATCHING']['matching_window_size'],
        'minimum_matching_ngrams_in_docs': config['MATCHING']['minimum_matching_ngrams_in_docs'],
        'minimum_matching_ngrams_in_window': config['MATCHING']['minimum_matching_ngrams_in_window'],
        'max_gap': config['MATCHING']['max_gap'],
        'minimum_matching_ngrams': config['MATCHING']['minimum_matching_ngrams'],
    }, 'time': timeend - timestart}
