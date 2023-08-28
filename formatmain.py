import os

from trd.formatting import blastifyDir, passimifyDir, tpairifyDir

blastdir = 'trdinput/blast'
passimdir = 'trdinput/passim'
tpdir = 'trdinput/textpair'
textbase = 'textbase'
files = ['Dk_Pages', 'Dk_Meta', 'KCh_Pages', 'KCh_Meta']

if __name__ == "__main__":

    blastifyDir(
            os.path.join(textbase, 'unnorm-unlem/full'),
            os.path.join(blastdir, 'unul/full'),
            files,
            ext='.txt'
        )

    blastifyDir(
        os.path.join(textbase, 'norm-unlem/pages'),
        os.path.join(blastdir, 'nnul/pages'),
        files
    )

    passimifyDir(
        os.path.join(textbase, 'unnorm-unlem/full'),
        os.path.join(passimdir, 'unul/full'),
        files,
        ext='.txt'
    )

    passimifyDir(
        os.path.join(textbase, 'norm-unlem/pages'),
        os.path.join(passimdir, 'nnul/pages'),
        files
    )

    tpairifyDir(
        os.path.join(textbase, 'unnorm-unlem/full'),
        os.path.join(tpdir, 'unul/full'),
        files,
        ext='.txt'
    )

    tpairifyDir(
        os.path.join(textbase, 'unnorm-unlem/pages'),
        os.path.join(tpdir, 'unul/pages'),
        files
    )
