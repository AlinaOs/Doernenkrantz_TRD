import os

import matplotlib.pyplot as plt
import numpy as np

from tools.rw import readFromCSV, writeToCSV, readDictFromJson, saveDictAsJson

gm_names = {
    0: 'alle ouerschafft van gode',
    1: 'alle ouerschaff van gode',
    2: 'krone bouen allen steden',
    3: 'do vnse lieue vrauwe geboren',
    4: 'do alle die werlt vreden',
    5: 'stat des (waren) friddens',
    6: 'stat des friddens',
    7: 'hie in puluer rastet',
    8: 'merteler des alden testamentz',
    9: 'des hemels influsse',  # KCh
    10: 'mynschen gueder complexien',
    11: 'wyrdicheyden der stat collen',
    12: 'wyrdicheyt der hilliger stat collen',
    13: 'schatz verwart wijl syn',
    14: 'eyndrechticheit vnd vreden',
    15: 'vrij sunder laster',
    16: 'nye hilligen gedoit',
    17: 'her gefoeget sijnre hilligen',
    18: 'gebeyne by eyn gefoeget',
}

seg = {
    'full': 'Volltexte',
    'pages': 'Seiten',
    'sentences': 'Sätze'
}

lem = {
    '0': 'Keine',
    '1': 'lem',
    '2': 'lem-ft',
    '3': 'lem-ren',
}


def segmentPlot(scorefile, outfile, toolname, score, ylim=1.0, colors=None, loc='upper left'):
    """
    This tutorial helped constructing the bar charts https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html
    :param scorefile:
    :return:
    """

    if colors is None:
        colors = ['tab:olive', 'tab:gray', 'tab:cyan']

    if score.lower() == 'f_m':
        sc = 6
        type = 'Treffer-basierte'
    elif score.lower() == 'f_ch':
        sc = 9
        type = 'Zeichenweise'
    elif score.lower() == 'f_cl':
        sc = 12
        type = 'Passagen-basierte'
    else:
        raise ValueError

    no_full = [0, 0, 0, 0, 0, 0, 0]
    no_sent = [0, 0, 0, 0, 0, 0, 0]
    no_pags = [0, 0, 0, 0, 0, 0, 0]

    full = [0, 0, 0, 0, 0, 0, 0]
    sent = [0, 0, 0, 0, 0, 0, 0]
    pags = [0, 0, 0, 0, 0, 0, 0]

    runscores = readFromCSV(scorefile)['lines']
    for run in runscores:
        if run[14] == 'full':
            segscores = full
            nos = no_full
        elif run[14] == 'sentences':
            segscores = sent
            nos = no_sent
        else:
            segscores = pags
            nos = no_pags
        if run[15] == '0':
            if run[16] == '1':
                prep_no = 1
            elif run[16] == '2':
                prep_no = 2
            elif run[16] == '3':
                prep_no = 3
            else:
                prep_no = 0
        else:
            if run[16] == '1':
                prep_no = 5
            elif run[16] == '2':
                prep_no = 6
            else:
                prep_no = 4
        segscores[prep_no] += float(run[sc])
        nos[prep_no] += 1

    full = [round(full[i] / no_full[i], 4) if no_full[i] > 0 else 0 for i in range(len(full)) if True]
    sent = [round(sent[i] / no_sent[i], 4) if no_sent[i] > 0 else 0 for i in range(len(sent))]
    pags = [round(pags[i] / no_pags[i], 4) if no_pags[i] > 0 else 0 for  i in range(len(pags))]

    preps = ('unnorm', 'unnorm-lem', 'unnorm-ft', 'unnorm-ren', 'norm', 'norm-lem', 'norm-ft')
    labels = ['Volltexte', 'Seiten','Sätze']
    segment_scores = [full, pags, sent]
    segment_nos = [no_full, no_pags, no_sent]

    for i in reversed(range(len(segment_scores))):
        if sum(segment_nos[i]) == 0:
            del colors[i]
            del labels[i]
            del segment_scores[i]
            del segment_nos[i]

    x = np.arange(len(preps))  # label locations
    width = 0.25  # width of the bars
    multiplier = -0.5 * len(segment_scores) + 1.5

    fig, ax = plt.subplots(layout='constrained')

    ncols = 0
    for i in range(len(segment_scores)):
        measurement = segment_scores[i]
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=labels[i], color=colors[i])
        ax.bar_label(rects, padding=3)
        multiplier += 1
        ncols += 1

    ax.set_ylabel(f'{score} (Durchschnitt)')
    ax.set_title(f'{type} Performanz pro Präprozessierungsart ({toolname})')
    ax.set_xticks(x + width, preps)
    ax.legend(loc=loc, ncols=ncols)
    ax.set_ylim(0, ylim)

    plt.savefig(outfile)


def goldmatchPlot(combievaldirs, outfile, toolnames, colors=None, loc='upper left'):
    """
    This tutorial helped constructing the bar charts https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html
    :param combievaldirs:
    :param outfile:
    :param toolnames:
    :param colors:
    :param loc:
    :return:
    """

    all_found_gms = []
    gms = set()

    if colors is None:
        colors = ['cadetblue', 'navy', 'maroon']
    for ced in combievaldirs:
        allruns = 0
        found_gms = {}
        inputs = os.listdir(ced)
        for inp in inputs:
            inp_path = os.path.join(ced, inp)
            if not os.path.isdir(inp_path):
                continue
            runs = os.listdir(inp_path)
            for r in runs:
                r_path = os.path.join(inp_path, r)
                try:
                    info = readDictFromJson(os.path.join(r_path, 'info.json'))
                    for match in info['goldmatches']:
                        found_gms[match[0]] = found_gms.get(match[0], 0) + 1
                    allruns += 1
                except FileNotFoundError:
                    continue
        for gm in found_gms.keys():
            onepc = allruns / 100
            found_gms[gm] = found_gms[gm] / onepc
        all_found_gms.append(found_gms)
        gms = gms.union(found_gms.keys())

    gms = list(gms)
    gms.sort()

    gm_labels = [f'{gm_names[gm]} ({gm + 1})' for gm in gms]
    run_percentages = []
    for found_gms in all_found_gms:
        rp = [round(found_gms.get(gm, 0), 1) for gm in gms]
        run_percentages.append(rp)

    x = np.arange(len(gm_labels))  # label locations
    width = 0.25  # width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    ncols = 0
    for i in range(len(run_percentages)):
        measurement = run_percentages[i]
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=toolnames[i], color=colors[i])
        multiplier += 1
        ncols += 1

    ax.set_ylabel(f'Anzahl Durchläufe (%)')
    ax.set_title(f'Erkennungsquote einzelner TR-Fälle pro Tool')
    ax.set_xticks(x + width, gm_labels)
    ax.legend(loc=loc, ncols=1)
    ax.set_ylim(0, 100)
    ax.grid(axis='y')
    ax.set_axisbelow(True)

    plt.xticks(rotation=45, ha='right')
    plt.savefig(outfile, dpi=350.0)


def highscoreTable(scorefile, outfile, score, n):
    if score.lower() == 'f_m':
        sc = 6
    elif score.lower() == 'f_ch':
        sc = 9
    elif score.lower() == 'f_cl':
        sc = 12
    elif score.lower() == 'av':
        sc = None
    else:
        raise ValueError

    rundata = readFromCSV(scorefile)
    runscores = rundata['lines']
    if sc is None:
        runscores.sort(reverse=True, key=lambda run: (float(run[5]) + float(run[8]) + float(run[11])) / 3)
        runscores.sort(reverse=True, key=lambda run: (float(run[4]) + float(run[7]) + float(run[10])) / 3)
        runscores.sort(reverse=True, key=lambda run: (float(run[6]) + float(run[9]) + float(run[12])) / 3)
    else:
        runscores.sort(reverse=True, key=lambda run: run[sc-1])
        runscores.sort(reverse=True, key=lambda run: run[sc-2])
        runscores.sort(reverse=True, key=lambda run: run[sc])

    if n > len(runscores):
        n = len(runscores)

    highscores = []
    for i in range(n):
        rs = runscores[i]
        hs = [
            rs[1],
            (float(rs[6]) + float(rs[9]) + float(rs[12])) / 3]
        if sc == 6:
            hs.extend([
                round(float(rs[6]), 4),  # F_m
                round(float(rs[4]), 4),  # R_m
                round(float(rs[5]), 4),  # P_m
                round(float(rs[9]), 4),  # F_ch
                round(float(rs[12]), 4)  # F_cl
            ])
        elif sc == 9:
            hs.extend([
                round(float(rs[6]), 4),  # F_m
                round(float(rs[9]), 4),  # F_ch
                round(float(rs[7]), 4),  # R_ch
                round(float(rs[8]), 4),  # P_ch
                round(float(rs[12]), 4)  # F_cl
            ])
        elif sc == 12:
            hs.extend([
                round(float(rs[6]), 4),  # F_m
                round(float(rs[9]), 4),  # F_ch
                round(float(rs[12]), 4),  # F_cl
                round(float(rs[10]), 4),  # R_cl
                round(float(rs[11]), 4)  # P_cl
            ])
        else:
            hs.extend([
                round(float(rs[6]), 4),  # F_m
                round(float(rs[9]), 4),  # F_ch
                round(float(rs[12]), 4)  # F_cl
            ])
        hs.extend([
            rs[2],  # detected test matches
            rs[3],  # detected gold matches
            rs[13],  # runtime
            seg[rs[14]],  # seg
            'Ja' if rs[15] == 1 else 'Nein',  # norm
            lem[rs[16]]  # lem
        ])
        hs.extend(rs[17:])
        highscores.append(hs)

    head = rundata['header']
    newheader = [
        head[1],
        'av']
    if sc == 6:
        newheader.extend([
            head[6],
            head[4],
            head[5],
            head[9],
            head[12]
        ])
    elif sc == 9:
        newheader.extend([
            head[6],
            head[9],
            head[7],
            head[8],
            head[12]
        ])
    elif sc == 12:
        newheader.extend([
            head[6],
            head[9],
            head[12],
            head[10],
            head[11],
        ])
    else:
        newheader.extend([
            head[6],
            head[9],
            head[12]
        ])
    newheader.extend([
        head[2],
        head[3],
        head[13],
        head[14],
        head[15],
        head[16]
    ])
    newheader.extend(head[17:])

    writeToCSV(outfile, highscores, newheader)


def highestRecallRun(scorefiles, outfile):
    hr_data = {}

    for scorefile in scorefiles:
        rundata = readFromCSV(scorefile)
        runscores = rundata['lines']
        header = rundata['header']
        runscores.sort(reverse=True, key=lambda run: run[5])
        runscores.sort(reverse=True, key=lambda run: run[4])
        rs = runscores[0]

        hr = {
            header[1]: rs[1],
            'av': (float(rs[6]) + float(rs[9]) + float(rs[12])) / 3,
            header[6]: round(float(rs[6]), 4),  # F_m
            header[9]: round(float(rs[9]), 4),  # F_ch
            header[12]: round(float(rs[12]), 4),  # F_cl
            header[2]: rs[2],  # detected test matches
            header[3]: rs[3],  # detected gold matches
            header[13]: rs[13],  # runtime
            header[14]: seg[rs[14]],  # seg
            header[15]: 'Ja' if rs[15] == 1 else 'Nein',  # norm
            header[16]: lem[rs[16]]  # lem
        }
        for i in range(17, len(rs)):
            hr[header[i]] = rs[i]
        hr_data[rs[0].lower()] = hr

        saveDictAsJson(outfile, hr_data)
