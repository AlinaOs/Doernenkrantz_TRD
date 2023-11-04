import os.path
from tools.plots import segmentPlot, highscoreTable, goldmatchPlot, highestRecallRun

if __name__ == '__main__':
    pd = 'trdoutput/evaluation/plots'
    tp_ed = 'trdoutput/evaluation/textpair_evaluation_scores.csv'
    p_ed = 'trdoutput/evaluation/passim_evaluation_scores.csv'
    b_ed = 'trdoutput/evaluation/blast_evaluation_scores.csv'
    '''
    segmentPlot(tp_ed, os.path.join(pd, 'textpair_prep_fm.png'), 'TextPAIR', 'F_m', 0.2)
    segmentPlot(p_ed, os.path.join(pd, 'passim_prep_fm.png'), 'Passim', 'F_m', 0.2)
    segmentPlot(b_ed, os.path.join(pd, 'blast_prep_fm.png'), 'BLAST', 'F_m', 0.2, loc='upper right')
    segmentPlot(tp_ed, os.path.join(pd, 'textpair_prep_fch.png'), 'TextPAIR', 'F_ch', 0.8)
    segmentPlot(p_ed, os.path.join(pd, 'passim_prep_fch.png'), 'Passim', 'F_ch', 0.8)
    segmentPlot(b_ed, os.path.join(pd, 'blast_prep_fch.png'), 'BLAST', 'F_ch', 0.8, loc='upper right')
    segmentPlot(tp_ed, os.path.join(pd, 'textpair_prep_fcl.png'), 'TextPAIR', 'F_cl')
    segmentPlot(p_ed, os.path.join(pd, 'passim_prep_fcl.png'), 'Passim',  'F_cl')
    segmentPlot(b_ed, os.path.join(pd, 'blast_prep_fcl.png'), 'BLAST',  'F_cl', loc='upper right')
    

    highscoreTable(tp_ed, os.path.join(pd, 'textpair_highscores_fm.csv'), 'f_m', 10)
    highscoreTable(p_ed, os.path.join(pd, 'passim_highscores_fm.csv'), 'f_m', 10)
    highscoreTable(b_ed, os.path.join(pd, 'blast_highscores_fm.csv'), 'f_m', 10)
    highscoreTable(tp_ed, os.path.join(pd, 'textpair_highscores_fch.csv'), 'f_ch', 10)
    highscoreTable(p_ed, os.path.join(pd, 'passim_highscores_fch.csv'), 'f_ch', 10)
    highscoreTable(b_ed, os.path.join(pd, 'blast_highscores_fch.csv'), 'f_ch', 10)
    highscoreTable(tp_ed, os.path.join(pd, 'textpair_highscores_fcl.csv'), 'f_cl', 10)
    highscoreTable(p_ed, os.path.join(pd, 'passim_highscores_fcl.csv'), 'f_cl', 10)
    highscoreTable(b_ed, os.path.join(pd, 'blast_highscores_fcl.csv'), 'f_cl', 10)
    highscoreTable(tp_ed, os.path.join(pd, 'textpair_highscores_av.csv'), 'av', 10)
    highscoreTable(p_ed, os.path.join(pd, 'passim_highscores_av.csv'), 'av', 10)
    highscoreTable(b_ed, os.path.join(pd, 'blast_highscores_av.csv'), 'av', 10)

    goldmatchPlot((
        'trdoutput/evaluation/blast', 'trdoutput/evaluation/passim', 'trdoutput/evaluation/textpair',
        ), os.path.join(pd, 'goldmatches.png'),
        ('BLAST', 'Passim', 'TextPAIR'
         ), loc='upper right')
    '''

    highestRecallRun((b_ed, p_ed, tp_ed), os.path.join(pd, 'highestRecall.json'))
