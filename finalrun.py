import os
import shutil
from trd.wrapping import runblastFull, runpassimFull, runtextpairFull

if __name__ == '__main__':
    shutil.rmtree('trdoutput/final_results')
    os.makedirs('trdoutput/final_results', exist_ok=True)
    bdir = 'trdoutput/final_results/blast'
    pdir = 'trdoutput/final_results/passim'
    tpdir = 'trdoutput/final_results/textpair'
    os.makedirs(bdir, exist_ok=True)
    os.makedirs(pdir, exist_ok=True)
    os.makedirs(tpdir, exist_ok=True)

    runblastFull('trdinput/unnorm/full', bdir,
                            word_size=7,
                            e_value=0.005,
                            min_length=15,
                            max_length=100000,
                            language='RIP'
                            )
    runpassimFull('trdinput/unnorm-lem/full', pdir,
                             n=9,
                             min_align=20,
                             min_match=2,
                             maxDF=100,
                             floating_ngrams=True,
                             beam=25,
                             pcopy=0.8,
                             all_pairs=True)
    runtextpairFull('trdinput/unnorm-lem/full', tpdir,
                               ngram=3,
                               gap=3,
                               matching_window_size=5,
                               minimum_matching_ngrams_in_docs=3,
                               minimum_matching_ngrams_in_window=3,
                               max_gap=2,
                               minimum_matching_ngrams=3,
                               store_banalities=True
                               )
