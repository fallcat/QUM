"""
Analyze macrof1 and bleu

Requirement:

You need to install sacrebleu from https://github.com/fallcat/sacre-BLEU/tree/bleu-recall first.

Usage:

Example:
1) get delta stats of SNMT and UNMT system:
python analysis/macrof1_analysis.py \
  --sys1-path ../../translations/snmt/de-en/newstest2019.1m64k.out.txt.detok \
  --sys2-path ../../translations/unmt/de-en/test.translation.en.debpe.detok \
  --ref-path ../../data/refs/newstest2019-ende-src.en \
  --src-path ../../data/refs/newstest2019-ende-ref.de \
  --save-stats-prefix stats/deen \
  --stats-type delta \
  --max-order 1

2) load delta stats and get a full report
python analysis/macrof1_analysis.py \
  --load-stats-path stats/deen.stats.pkl \
  --stats-type apply_delta \
  --report-type top10_mf1 top10_bleu median10_mf1 median10_bleu same_bleu_diff_mf1 \
  --report-path-prefix reports/deen \
  --max-order 1

3) just get bleu and mf1
python analysis/macrof1_analysis.py \
  --sys1-path ../../translations/snmt/de-en/newstest2019.1m64k.out.txt.detok \
  --sys2-path ../../translations/unmt/de-en/test.translation.en.debpe.detok \
  --ref-path ../../data/refs/newstest2019-ende-src.en \
  --src-path ../../data/refs/newstest2019-ende-ref.de \
  --report-type total_metrics \
  --report-path-prefix reports/deen \
  --max-order 1

4) print all sentences base on a metric
python analysis/macrof1_analysis.py \
  --load-stats-path stats/deen.delta.stats.pkl \
  --stats-type apply_delta \
  --save-stats-prefix stats/deen \
  --report-type all_mf1 \
  --report-path-prefix reports/deen \
  --max-order 1
"""


import pickle
import argparse
import sacrebleu
import numpy as np
from tqdm import tqdm
# from bleurt import score
import tensorflow as tf
from collections import Counter
from nltk.corpus import words
from nltk.corpus import wordnet as wn
import nltk
nltk.download('words')
nltk.download('wordnet')

tf.compat.v1.flags.DEFINE_string('sys1-path','','')
tf.compat.v1.flags.DEFINE_string('sys2-path', '', '')
tf.compat.v1.flags.DEFINE_string('ref-path', '', '')
tf.compat.v1.flags.DEFINE_string('src-path', '', '')
tf.compat.v1.flags.DEFINE_string('sys1-name', '', '')
tf.compat.v1.flags.DEFINE_string('sys2-name', '', '')
tf.compat.v1.flags.DEFINE_string('lc', '', '')
tf.compat.v1.flags.DEFINE_string('max-order', '', '')
tf.compat.v1.flags.DEFINE_string('load-stats-path', '', '')
tf.compat.v1.flags.DEFINE_string('save-stats-prefix', '', '')
tf.compat.v1.flags.DEFINE_string('stats-type', '', '')
tf.compat.v1.flags.DEFINE_string('report-type', '', '')
tf.compat.v1.flags.DEFINE_string('print-report', '', '')
tf.compat.v1.flags.DEFINE_string('report-path-prefix', '', '')



def get_parser():
    parser = argparse.ArgumentParser()

    # model args
    # vocab_size, d_embed, output_size, num_heads, dropout_p=0.1, padding_idx=0
    parser.add_argument('--sys1-path', type=str, default='/path/to/sys1/output', help='path of output from system 1')

    parser.add_argument('--sys2-path', type=str, default='/path/to/sys2/output', help='path of output from system 2')

    parser.add_argument('--ref-path', type=str, default='/path/to/reference/output', help='path of reference')

    parser.add_argument('--src-path', type=str, default='/path/to/source/output', help='path of source')

    parser.add_argument('--sys1-name', type=str, default='SNMT', help='name of sys 1')

    parser.add_argument('--sys2-name', type=str, default='UNMT', help='name of sys 2')

    parser.add_argument('--lc', default=False, action='store_true', help='lowercase sentences')

    parser.add_argument('--max-order', type=int, default=1, help='ngram')

    parser.add_argument('--load-stats-path', type=str, default=None, help='path to load stats')

    parser.add_argument('--save-stats-prefix', type=str, default=None, help='prefix to store stats')

    parser.add_argument('--stats-type', type=str, default=None, choices=['single', 'delta', 'apply_delta'],
                        help='type of stats to use')

    parser.add_argument('--report-type', type=str, nargs='*', default='top10', choices=['all_mf1', 'all_bleu',
                                                                                        'all_sys1', 'all_sys2',
                                                                                        'top10_mf1', 'top10_bleu',
                                                                                        'median10_mf1', 'median10_bleu',
                                                                                        'same_bleu_diff_mf1', 'total_metrics',
                                                                                        'improved_translation'])
    parser.add_argument('--print-report', default=False, action='store_true',
                        help='whether or not to print the report')

    parser.add_argument('--report-path-prefix', type=str, default=None, help='path to store the reports')

    return parser


def get_percent_en(sent_list):
    tokenizer = nltk.RegexpTokenizer(r"\w+")

    total = Counter()
    en = Counter()

    print("Accumulating stats for percent en ...")
    for sent in tqdm(sent_list):
        w_list = tokenizer.tokenize(sent)
        for w in w_list:
            total[w.lower()] += 1
    for w in tqdm(total):
        if len(wn.synsets(w.lower())) > 0 or w in words.words():
            en[w] = total[w]

    micro_perc = sum(en.values()) / sum(total.values())

    total_list = []
    en_list = []
    for sent in tqdm(sent_list):
        w_list = tokenizer.tokenize(sent)
        total_list.append(len(w_list))
        en_list.append(sum([1 if w.lower() in en else 0 for w in w_list]))

    macro_perc = sum([en_list[i] / total_list[i] for i in range(len(total_list))]) / len(total_list)

    return micro_perc, macro_perc, total_list, en_list, total, en


def get_improved_translation(sys1_path, sys2_path, ref_path, src_path, sys1_name='sys1', sys2_name='sys2', print_report=False, filepath=None):
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    with open(sys1_path, 'rt') as sys1_file:
        with open(sys2_path, 'rt') as sys2_file:
            with open(ref_path, 'rt') as ref_file:
                with open(src_path, 'rt') as src_file:
                    sys1_list = [line.strip() for line in sys1_file.readlines()]
                    sys2_list = [line.strip() for line in sys2_file.readlines()]
                    ref_list = [line.strip() for line in ref_file.readlines()]
                    src_list = [line.strip() for line in src_file.readlines()]
    micro_perc_sys1, macro_perc_sys1, total_list_sys1, en_list_sys1, total_sys1, en_sys1 = get_percent_en(sys1_list)
    micro_perc_sys2, macro_perc_sys2, total_list_sys2, en_list_sys2, total_sys2, en_sys2 = get_percent_en(sys2_list)
    report = ''
    better = 0
    worse = 0
    print("total_list_sys1", total_list_sys1)
    print("en_list_sys1", en_list_sys1)
    print("total_list_sys2", total_list_sys2)
    print("en_list_sys2", en_list_sys2)
    for i in range(len(en_list_sys1)):
        if total_list_sys2[i] - en_list_sys2[i] < total_list_sys1[i] - en_list_sys1[i]:
            report += f'{i}, # untranslations: {total_list_sys1[i] - en_list_sys1[i]} => {total_list_sys2[i] - en_list_sys2[i]} \n'
            report += f"Src : {src_list[i]}\n"
            report += f"Ref : {ref_list[i]}\n"
            report += f"{sys1_name}: {sys1_list[i]}\n"
            report += f"{sys2_name}: {sys2_list[i]}\n"
            report += f"Old Untranslation: {','.join([w for w in tokenizer.tokenize(sys1_list[i]) if w.lower() not in en_sys1])}\n"
            report += f"New Untranslation: {','.join([w for w in tokenizer.tokenize(sys2_list[i]) if w.lower() not in en_sys2])}\n"
            report += "------------------\n"
            better += 1
        elif total_list_sys2[i] - en_list_sys2[i] > total_list_sys1[i] - en_list_sys1[i]:
            worse += 1
    print("improved #: ", better)
    print("degraded #: ", worse)

    if print_report:
        print(report)
    if filepath is not None:
        print(filepath)
        with open(filepath, 'wt') as output_file:
            output_file.write(report)


def get_total_metrics(sys1_path, sys2_path, ref_path, lowercase=False, max_order=1, sys1_name='sys1', sys2_name='sys2', filepath=None):
    with open(sys1_path, 'rt') as sys1_file:
        with open(sys2_path, 'rt') as sys2_file:
            with open(ref_path, 'rt') as ref_file:
                sys1_list = [line.strip() for line in sys1_file.readlines()]
                sys2_list = [line.strip() for line in sys2_file.readlines()]
                ref_list = [line.strip() for line in ref_file.readlines()]

    mf1_sys1 = sacrebleu.corpus_rebleu2(sys1_list, [ref_list], lowercase=lowercase, average='macro', max_order=max_order)
    mf1_sys2 = sacrebleu.corpus_rebleu2(sys2_list, [ref_list], lowercase=lowercase, average='macro', max_order=max_order)
    mf1_sys1_f1 = sacrebleu.corpus_rebleu2(sys1_list, [ref_list], lowercase=lowercase, average='macro', max_order=max_order, measure_name='f1')
    mf1_sys2_f1 = sacrebleu.corpus_rebleu2(sys2_list, [ref_list], lowercase=lowercase, average='macro', max_order=max_order, measure_name='f1')
    mf1_sys1_new = sacrebleu.corpus_rebleu2(sys1_list, [ref_list], average='macro', word_class=True, max_order=max_order)
    mf1_sys2_new = sacrebleu.corpus_rebleu2(sys2_list, [ref_list], average='macro', word_class=True, max_order=max_order)
    mf1_sys1_new_f1 = sacrebleu.corpus_rebleu2(sys1_list, [ref_list], average='macro', word_class=True, max_order=max_order, measure_name='f1')
    mf1_sys2_new_f1 = sacrebleu.corpus_rebleu2(sys2_list, [ref_list], average='macro', word_class=True, max_order=max_order, measure_name='f1')
    bleu_sys1 = sacrebleu.corpus_bleu(sys1_list, [ref_list], lowercase=lowercase)
    bleu_sys2 = sacrebleu.corpus_bleu(sys2_list, [ref_list], lowercase=lowercase)
    chrf_sys1 = sacrebleu.corpus_chrf(sys1_list, ref_list)
    chrf_sys2 = sacrebleu.corpus_chrf(sys2_list, ref_list)
    micro_perc_sys1, macro_perc_sys1, total_list_sys1, en_list_sys1, total_sys1, en_sys1 = get_percent_en(sys1_list)
    micro_perc_sys2, macro_perc_sys2, total_list_sys2, en_list_sys2, total_sys2, en_sys2 = get_percent_en(sys2_list)


    # bleurt_checkpoint = "/Users/weiqiuyou/Documents/USC_ISI/QUM/tools/bleurt/bleurt/bleurt-base-128"
    # scorer = score.BleurtScorer(bleurt_checkpoint)
    # bleurt_sys1 = np.mean(scorer.score(ref_list, sys1_list))
    # bleurt_sys2 = np.mean(scorer.score(ref_list, sys2_list))

    report = ''
    report += f'mf1_{sys1_name}: {mf1_sys1}\n'
    report += f'mf1_{sys2_name}: {mf1_sys2}\n'
    report += f'mf1_{sys1_name}_f1: {mf1_sys1_f1}\n'
    report += f'mf1_{sys2_name}_f1: {mf1_sys2_f1}\n'
    report += f'mf1_new_{sys1_name}: {mf1_sys1_new}\n'
    report += f'mf1_new_{sys2_name}: {mf1_sys2_new}\n'
    report += f'mf1_new_{sys1_name}_f1: {mf1_sys1_new_f1}\n'
    report += f'mf1_new_{sys2_name}_f1: {mf1_sys2_new_f1}\n'
    report += f'bleu_{sys1_name}: {bleu_sys1}\n'
    report += f'bleu_{sys2_name}: {bleu_sys2}\n'
    report += f'chrf_{sys1_name}: {chrf_sys1}\n'
    report += f'chrf_{sys2_name}: {chrf_sys2}\n'
    report += f'micro_perc_{sys1_name}: {micro_perc_sys1}\tmacro_perc_{sys1_name}: {macro_perc_sys1}\n'
    report += f'micro_perc_{sys2_name}: {micro_perc_sys2}\tmacro_perc_{sys2_name}: {macro_perc_sys2}\n'
    # report += f'bleurt_{sys1_name}: {bleurt_sys1}\n'
    # report += f'bleurt_{sys2_name}: {bleurt_sys2}\n'

    print(report)
    if filepath is not None:
        with open(filepath, 'wt') as output_file:
            output_file.write(report)


def get_stats(sys1_path, sys2_path, ref_path, src_path, lowercase=False, max_order=1):
    with open(sys1_path, 'rt') as sys1_file:
        with open(sys2_path, 'rt') as sys2_file:
            with open(ref_path, 'rt') as ref_file:
                with open(src_path, 'rt') as src_file:
                    sys1_list = [line.strip() for line in sys1_file.readlines()]
                    sys2_list = [line.strip() for line in sys2_file.readlines()]
                    ref_list = [line.strip() for line in ref_file.readlines()]
                    src_list = [line.strip() for line in src_file.readlines()]

    mf1s = []
    mf1_diff = []
    bleus = []
    bleu_diff = []
    for i in range(len(sys1_list)):
        # get macro f1
        mf1_sys1 = sacrebleu.corpus_rebleu([sys1_list[i]], [[ref_list[i]]], lowercase=lowercase, average='macro', max_order=max_order)
        mf1_sys2 = sacrebleu.corpus_rebleu([sys2_list[i]], [[ref_list[i]]], lowercase=lowercase, average='macro', max_order=max_order)
        mf1s.append([mf1_sys1.score, mf1_sys2.score])
        mf1_diff.append(mf1_sys1.score - mf1_sys2.score)

        # get bleu
        bleu_sys1 = sacrebleu.corpus_bleu([sys1_list[i]], [[ref_list[i]]], lowercase=lowercase)
        bleu_sys2 = sacrebleu.corpus_bleu([sys2_list[i]], [[ref_list[i]]], lowercase=lowercase)
        bleus.append([bleu_sys1.score, bleu_sys2.score])
        bleu_diff.append(bleu_sys1.score - bleu_sys2.score)

    mf1_ranked_indices = np.flip(np.argsort(mf1_diff), 0)  # indices of sentences ranked by macro f1 of sys1 - of sys2
    bleu_ranked_indices = np.flip(np.argsort(bleu_diff), 0)  # indices of sentences ranked by bleu of sys1 - of sys2

    return {'sys1': sys1_list,
            'sys2': sys2_list,
            'ref': ref_list,
            'src': src_list,
            'mf1s': mf1s,
            'mf1_diff': mf1_diff,
            'mf1_ranked_indices': mf1_ranked_indices,
            'bleus': bleus,
            'bleu_diff': bleu_diff,
            'bleu_ranked_indices': bleu_ranked_indices}


def get_delta_stats(sys1_path, sys2_path, ref_path, src_path, lowercase=False, max_order=1):
    with open(sys1_path, 'rt') as sys1_file:
        with open(sys2_path, 'rt') as sys2_file:
            with open(ref_path, 'rt') as ref_file:
                with open(src_path, 'rt') as src_file:
                    sys1_list = [line.strip() for line in sys1_file.readlines()]
                    sys2_list = [line.strip() for line in sys2_file.readlines()]
                    ref_list = [line.strip() for line in ref_file.readlines()]
                    src_list = [line.strip() for line in src_file.readlines()]

    mf1s = []
    bleus = []
    bp_sys1 = []
    bp_sys2 = []

    for i in tqdm(range(len(sys1_list))):
        # get macro f1
        sys1_sublist = sys1_list[:i] + sys1_list[i + 1:]
        sys2_sublist = sys2_list[:i] + sys2_list[i + 1:]
        ref_sublist = ref_list[:i] + ref_list[i + 1:]
        mf1_sys1 = sacrebleu.corpus_rebleu(sys1_sublist, [ref_sublist], lowercase=lowercase, average='macro', max_order=max_order)
        mf1_sys2 = sacrebleu.corpus_rebleu(sys2_sublist, [ref_sublist], lowercase=lowercase, average='macro', max_order=max_order)
        mf1s.append([mf1_sys1.score, mf1_sys2.score])

        # get bleu
        bleu_sys1 = sacrebleu.corpus_bleu(sys1_sublist, [ref_sublist], lowercase=lowercase)
        bleu_sys2 = sacrebleu.corpus_bleu(sys2_sublist, [ref_sublist], lowercase=lowercase)
        bleus.append([bleu_sys1.score, bleu_sys2.score])
        bp_sys1.append(bleu_sys1.bp)
        bp_sys2.append(bleu_sys2.bp)

    return {'sys1': sys1_list,
            'sys2': sys2_list,
            'ref': ref_list,
            'src': src_list,
            'mf1s': mf1s,
            'bleus': bleus,
            'bp_sys1': bp_sys1,
            'bp_sys2': bp_sys2
            }


def apply_delta(stats, lowercase=False, max_order=1):
    sys1_list = stats['sys1']
    sys2_list = stats['sys2']
    ref_list = stats['ref']
    src_list = stats['src']

    mf1_sys1 = sacrebleu.corpus_rebleu(sys1_list, [ref_list], lowercase=lowercase, average='macro', max_order=max_order)
    mf1_sys2 = sacrebleu.corpus_rebleu(sys2_list, [ref_list], lowercase=lowercase, average='macro', max_order=max_order)
    bleu_sys1 = sacrebleu.corpus_bleu(sys1_list, [ref_list], lowercase=lowercase)
    bleu_sys2 = sacrebleu.corpus_bleu(sys2_list, [ref_list], lowercase=lowercase)
    print('mf1_sys1', mf1_sys1)
    print('mf1_sys2', mf1_sys2)
    print('bleu_sys1', bleu_sys1)
    print('bleu_sys2', bleu_sys2)

    mf1s = []
    mf1_diff = []
    bleus = []
    bleu_diff = []

    sys1s = []
    sys1_diff = []
    sys2s = []
    sys2_diff = []

    for i in tqdm(range(len(sys1_list))):
        # get macro f1
        new_mf1_sys1_score = mf1_sys1.score - stats['mf1s'][i][0]  # old(all) - new(-1)
        new_mf1_sys2_score = mf1_sys2.score - stats['mf1s'][i][1]
        mf1s.append([new_mf1_sys1_score, new_mf1_sys2_score])
        mf1_diff.append(new_mf1_sys1_score - new_mf1_sys2_score)

        # get bleu
        new_bleu_sys1_score = bleu_sys1.score - stats['bleus'][i][0]
        new_bleu_sys2_score = bleu_sys2.score - stats['bleus'][i][1]
        bleus.append([new_bleu_sys1_score, new_bleu_sys2_score])
        bleu_diff.append(new_bleu_sys1_score - new_bleu_sys2_score)

        # get sys1
        sys1s.append([new_mf1_sys1_score, new_bleu_sys1_score])
        sys1_diff.append(new_mf1_sys1_score - new_bleu_sys1_score)
        sys2s.append([new_mf1_sys2_score, new_bleu_sys2_score])
        sys2_diff.append(new_mf1_sys2_score - new_bleu_sys2_score)

    mf1_ranked_indices = np.flip(np.argsort(mf1_diff), 0)  # indices of sentences ranked by macro f1 of sys1 - of sys2
    bleu_ranked_indices = np.flip(np.argsort(bleu_diff), 0)  # indices of sentences ranked by bleu of sys1 - of sys2
    sys1_ranked_indices = np.flip(np.argsort(sys1_diff), 0)
    sys2_ranked_indices = np.flip(np.argsort(sys2_diff), 0)

    return {'sys1': sys1_list,
            'sys2': sys2_list,
            'ref': ref_list,
            'src': src_list,
            'mf1s': mf1s,
            'mf1_diff': mf1_diff,
            'mf1_ranked_indices': mf1_ranked_indices,
            'bleus': bleus,
            'bleu_diff': bleu_diff,
            'bleu_ranked_indices': bleu_ranked_indices,
            'sys1s': sys1s,
            'sys1_diff': sys1_diff,
            'sys1_ranked_indices': sys1_ranked_indices,
            'sys2s': sys2s,
            'sys2_diff': sys2_diff,
            'sys2_ranked_indices': sys2_ranked_indices,
            'bp_sys1': stats['bp_sys1'] if 'bp_sys1' in stats else None,
            'bp_sys2': stats['bp_sys2'] if 'bp_sys1' in stats else None
            }


def get_top10_report(stats, filepath=None, print_report=True, metric='mf1', sys1_name='sys1', sys2_name='sys2'):
    if metric == 'mf1':
        metrics = stats['mf1s']
        metric_diff = stats['mf1_diff']
        ranked_indices = stats['mf1_ranked_indices']
    else:
        metrics = stats['bleus']
        metric_diff = stats['bleu_diff']
        ranked_indices = stats['bleu_ranked_indices']
    src_list = stats['src']
    ref_list = stats['ref']
    sys1_list = stats['sys1']
    sys2_list = stats['sys2']

    report = f'Metric: {metric}\n'
    report += f'{sys1_name} > {sys2_name} top 10\n'
    report += "==================\n"
    for i in range(10):
        idx = ranked_indices[i]
        report += f'{idx}, {sys1_name}: {metrics[idx][0]}, {sys2_name}: {metrics[idx][1]}, diff: {metric_diff[idx]}\n'
        report += f"Src : {src_list[idx]}\n"
        report += f"Ref : {ref_list[idx]}\n"
        report += f"sys1: {sys1_list[idx]}\n"
        report += f"sys2: {sys2_list[idx]}\n"
        report += "------------------\n"

    report += f'{sys2_name} > {sys1_name} top 10\n'
    report += "==================\n"
    for i in range(1, 11):
        idx = ranked_indices[-i]
        report += f'{idx}, {sys1_name}: {metrics[idx][0]}, {sys2_name}: {metrics[idx][1]}, diff: {metric_diff[idx]}\n'
        report += f"Src : {src_list[idx]}\n"
        report += f"Ref : {ref_list[idx]}\n"
        report += f"sys1: {sys1_list[idx]}\n"
        report += f"sys2: {sys2_list[idx]}\n"
        report += "------------------\n"

    if filepath is not None:
        with open(filepath, 'wt') as output_file:
            output_file.write(report)

    if print_report:
        print(report)


def get_median10_report(stats, filepath=None, print_report=True, metric='mf1', sys1_name='sys1', sys2_name='sys2'):
    metrics = stats[f'{metric}s']
    metric_diff = stats[f'{metric}_diff']
    src_list = stats['src']
    ref_list = stats['ref']
    sys1_list = stats['sys1']
    sys2_list = stats['sys2']
    ranked_indices = stats[f'{metric}_ranked_indices']

    sys1_better = []
    sys2_better = []
    equal = []
    for idx in ranked_indices:
        if metric_diff[idx] > 0:
            sys1_better.append(idx)
        elif metric_diff[idx] < 0:
            sys2_better.append(idx)
        else:
            equal.append(idx)
    sys2_better = np.flip(sys2_better)

    def add_report(metrics, src_list, ref_list, sys1_list, sys2_list, idx):
        subreport = ''
        subreport += f'{idx}, {sys1_name}: {metrics[idx][0]}, {sys2_name}: {metrics[idx][1]}, diff: {metric_diff[idx]}\n'
        subreport += f"Src : {src_list[idx]}\n"
        subreport += f"Ref : {ref_list[idx]}\n"
        subreport += f"sys1: {sys1_list[idx]}\n"
        subreport += f"sys2: {sys2_list[idx]}\n"
        subreport += "------------------\n"
        return subreport

    def add_median10_reports(idx_list, metrics, src_list, ref_list, sys1_list, sys2_list):
        subreport = ''
        if len(idx_list) > 10:
            for i in range(-5, 5):
                idx = idx_list[int(len(idx_list) / 2) + i]
                subreport += add_report(metrics, src_list, ref_list, sys1_list, sys2_list, idx)
        else:
            for idx in idx_list:
                subreport += add_report(metrics, src_list, ref_list, sys1_list, sys2_list, idx)
        return subreport

    report = ''
    report += f'{sys1_name} better: {len(sys1_better)}, {sys2_name} better: {len(sys2_better)}, Equal: {len(equal)}\n\n'

    report += f"{sys1_name} > {sys2_name} median 10\n"
    report += "==================\n"
    report += add_median10_reports(sys1_better, metrics, src_list, ref_list, sys1_list, sys2_list)

    report += f"{sys2_name} > {sys1_name} median 10\n"
    report += "==================\n"
    report += add_median10_reports(sys2_better, metrics, src_list, ref_list, sys1_list, sys2_list)

    report += f"{sys2_name} = {sys1_name} median 10\n"
    report += "==================\n"
    report += add_median10_reports(equal, metrics, src_list, ref_list, sys1_list, sys2_list)

    if filepath is not None:
        with open(filepath, 'wt') as output_file:
            output_file.write(report)
    if print_report:
        print(report)


def get_same_bleu_diff_mf1_report(stats, filepath=None, print_report=True, sys1_name='sys1', sys2_name='sys2'):
    mf1s = stats['mf1s']
    mf1_diff = stats['mf1_diff']
    bleus = stats['bleus']
    bleu_diff = stats['bleu_diff']
    src_list = stats['src']
    ref_list = stats['ref']
    sys1_list = stats['sys1']
    sys2_list = stats['sys2']
    mf1_ranked_indices = stats['mf1_ranked_indices']

    sys1_better = []
    sys2_better = []
    equal = []
    for idx in mf1_ranked_indices:
        if bleu_diff[idx] == 0:
            if mf1_diff[idx] > 0:
                sys1_better.append(idx)
            elif mf1_diff[idx] < 0:
                sys2_better.append(idx)
            else:
                equal.append(idx)
    sys2_better = np.flip(sys2_better)

    report = ''
    report += f'{sys1_name} better: {len(sys1_better)}, {sys2_name} better: {len(sys2_better)}, Equal: {len(equal)}\n\n'

    def add_mf1_report(mf1s, src_list, ref_list, sys1_list, sys2_list, bleus, idx):
        subreport = ''
        subreport += f'{idx}, {sys1_name}: {mf1s[idx][0]}, {sys2_name}: {mf1s[idx][1]}, mf1 diff: {mf1_diff[idx]}, bleu {bleus[idx][0]}\n'
        subreport += f"Src : {src_list[idx]}\n"
        subreport += f"Ref : {ref_list[idx]}\n"
        subreport += f"sys1: {sys1_list[idx]}\n"
        subreport += f"sys2: {sys2_list[idx]}\n"
        subreport += "------------------\n"
        return subreport

    def add_median10_reports(idx_list, mf1s, src_list, ref_list, sys1_list, sys2_list, bleus):
        subreport = ''
        if len(idx_list) > 10:
            for i in range(-5, 5):
                idx = idx_list[int(len(idx_list) / 2) + i]
                subreport += add_mf1_report(mf1s, src_list, ref_list, sys1_list, sys2_list, bleus, idx)
        else:
            for idx in idx_list:
                subreport += add_mf1_report(mf1s, src_list, ref_list, sys1_list, sys2_list, bleus, idx)
        return subreport

    report += f"{sys1_name} > {sys2_name} median 10\n"
    report += "==================\n"
    report += add_median10_reports(sys1_better, mf1s, src_list, ref_list, sys1_list, sys2_list, bleus)

    report += f"{sys2_name} > {sys1_name} median 10\n"
    report += "==================\n"
    report += add_median10_reports(sys2_better, mf1s, src_list, ref_list, sys1_list, sys2_list, bleus)

    report += f"{sys2_name} = {sys1_name} median 10\n"
    report += "==================\n"
    report += add_median10_reports(equal, mf1s, src_list, ref_list, sys1_list, sys2_list, bleus)

    if filepath is not None:
        with open(filepath, 'wt') as output_file:
            output_file.write(report)
    if print_report:
        print(report)

def get_all_report(stats, fileprefix=None, print_report=True, metric='mf1', sys1_name='sys1', sys2_name='sys2'):
    sys1_list = stats['sys1']
    sys2_list = stats['sys2']
    ref_list = stats['ref']
    src_list = stats['src']
    mf1s = stats['mf1s']
    mf1_diff = stats['mf1_diff']
    mf1_ranked_indices = stats['mf1_ranked_indices']
    bleus = stats['bleus']
    bleu_diff = stats['bleu_diff']
    bleu_ranked_indices = stats['bleu_ranked_indices']
    sys1s = stats['sys1s']
    sys1_diff = stats['sys1_diff']
    sys1_ranked_indices = stats['sys1_ranked_indices']
    sys2s = stats['sys2s']
    sys2_diff = stats['sys2_diff']
    sys2_ranked_indices = stats['sys2_ranked_indices']
    bp_sys1 = stats['bp_sys1']
    bp_sys2 = stats['bp_sys2']
    n = len(ref_list)

    # bleurt_checkpoint = "/Users/weiqiuyou/Documents/USC_ISI/QUM/tools/bleurt/bleurt/bleurt-base-128"
    # scorer = score.BleurtScorer(bleurt_checkpoint)
    # bleurt_sys1 = scorer.score(ref_list, sys1_list)
    # bleurt_sys2 = scorer.score(ref_list, sys2_list)

    def add_report(metrics, metric_diff, idx, src_list, ref_list, sys1_list, sys2_list):
        subreport = ''
        subreport += f'{idx}, {sys1_name}: {metrics[idx][0]}, {sys2_name}: {metrics[idx][1]}, diff: {metric_diff[idx]}\n'
        subreport += f"Src : {src_list[idx]}\n"
        subreport += f"Ref : {ref_list[idx]}\n"
        subreport += f"sys1: {sys1_list[idx]}\n"
        subreport += f"sys2: {sys2_list[idx]}\n"
        subreport += "------------------\n"
        return subreport

    report = []
    for i in range(n):
        if metric == 'mf1':
            idx = mf1_ranked_indices[i]
        elif metric == 'bleu':
            idx = bleu_ranked_indices[i]
        elif metric == 'sys1':
            idx = sys1_ranked_indices[i]
        else:
            idx = sys2_ranked_indices[i]
        report.append('\t'.join([str(idx), src_list[idx], ref_list[idx], sys1_list[idx], sys2_list[idx],
                                 *[str(x) for x in mf1s[idx]], str(mf1_diff[idx]), *[str(x) for x in bleus[idx]],
                                 str(bleu_diff[idx]), *[str(x) for x in sys1s[idx]],
                                 str(sys1_diff[idx]), *[str(x) for x in sys2s[idx]],
                                 str(sys2_diff[idx]), str(bp_sys1[idx]), str(bp_sys2[idx])]) + '\n')
        # report.append('\t'.join([str(idx), src_list[idx], ref_list[idx], sys1_list[idx], sys2_list[idx],
        #               *[str(x) for x in mf1s[idx]], str(mf1_diff[idx]), *[str(x) for x in bleus[idx]], str(bleu_diff[idx]),
        #                          str(bleurt_sys1[idx]), str(bleurt_sys2[idx])]) + '\n')

    if fileprefix is not None:
        with open(fileprefix + '.tsv', 'wt') as output_file:
            output_file.writelines(report)
        with open(fileprefix + '.txt', 'wt') as output_file:
            if metric == 'mf1':
                metrics = mf1s
                metric_diff = mf1_diff
            elif metric == 'bleu':
                metrics = bleus
                metric_diff = bleu_diff
            elif metric == 'sys1':
                metrics = sys1s
                metric_diff = sys1_diff
            else:
                metrics = sys2s
                metric_diff = sys2_diff
            for i in range(n):
                if metric == 'mf1':
                    idx = mf1_ranked_indices[i]
                elif metric == 'bleu':
                    idx = bleu_ranked_indices[i]
                elif metric == 'sys1':
                    idx = sys1_ranked_indices[i]
                else:
                    idx = sys2_ranked_indices[i]
                output_file.write(add_report(metrics, metric_diff, idx, src_list, ref_list, sys1_list, sys2_list))

    if print_report:
        print(report)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    # get stats, either load or produce now
    if args.stats_type is not None:
        if args.load_stats_path is not None:
            with open(args.load_stats_path, 'rb') as input_file:
                stats = pickle.load(input_file)
            assert args.stats_type == 'apply_delta'
            stats = apply_delta(stats, lowercase=args.lc, max_order=args.max_order)
        else:
            if args.stats_type == 'single':
                stats = get_stats(args.sys1_path, args.sys2_path, args.ref_path, args.src_path, lowercase=args.lc, max_order=args.max_order)
            else:
                stats = get_delta_stats(args.sys1_path, args.sys2_path, args.ref_path, args.src_path, lowercase=args.lc, max_order=args.max_order)
                if args.stats_type == 'apply_delta':
                    stats = apply_delta(stats, lowercase=args.lc, max_order=args.max_order)
        if args.save_stats_prefix is not None:
            with open(f'{args.save_stats_prefix}.{args.stats_type}.{args.max_order}gram.stats.pkl', 'wb') as output_file:
                pickle.dump(stats, output_file)
    else:
        if args.load_stats_path is not None:
            with open(args.load_stats_path, 'rb') as input_file:
                stats = pickle.load(input_file)
        else:
            assert args.report_type == ['total_metrics'] or args.report_type == ['improved_translation']

    # get reports
    # ['top10_mf1', 'top10_bleu', 'median10_mf1', 'median10_bleu', 'same_bleu_diff_mf1']
    if 'total_metrics' in args.report_type:
        get_total_metrics(args.sys1_path, args.sys2_path, args.ref_path, lowercase=args.lc, max_order=args.max_order,
                          sys1_name=args.sys1_name, sys2_name=args.sys2_name,
                          filepath=f'{args.report_path_prefix}.{args.stats_type}.total_metrics.{args.max_order}gram')

    if 'improved_translation' in args.report_type:
        get_improved_translation(args.sys1_path, args.sys2_path, args.ref_path, args.src_path,
                                 sys1_name=args.sys1_name, sys2_name=args.sys2_name, print_report=args.print_report,
                                 filepath=f'{args.report_path_prefix}.{args.stats_type}.improve_translation.{args.max_order}gram')

    if 'all_mf1' in args.report_type:
        get_all_report(stats, fileprefix=f'{args.report_path_prefix}.{args.stats_type}.all.{args.max_order}gram',
                       print_report=args.print_report, metric='mf1')

    if 'all_bleu' in args.report_type:
        get_all_report(stats, fileprefix=f'{args.report_path_prefix}.{args.stats_type}.all.{args.max_order}gram',
                       print_report=args.print_report, metric='bleu')

    if 'all_sys1' in args.report_type:
        get_all_report(stats, fileprefix=f'{args.report_path_prefix}.{args.stats_type}.all.{args.max_order}gram',
                       print_report=args.print_report, metric='sys1')

    if 'all_sys2' in args.report_type:
        get_all_report(stats, fileprefix=f'{args.report_path_prefix}.{args.stats_type}.all.{args.max_order}gram',
                       print_report=args.print_report, metric='sys2')

    if 'top10_mf1' in args.report_type:
        get_top10_report(stats, filepath=f'{args.report_path_prefix}.{args.stats_type}.top10_mf1.{args.max_order}gram',
                         print_report=args.print_report, metric='mf1',
                         sys1_name=args.sys1_name, sys2_name=args.sys2_name)
    if 'top10_bleu' in args.report_type:
        get_top10_report(stats, filepath=f'{args.report_path_prefix}.{args.stats_type}.top10_bleu.{args.max_order}gram',
                         print_report=args.print_report, metric='bleu',
                         sys1_name=args.sys1_name, sys2_name=args.sys2_name)
    if 'median10_mf1' in args.report_type:
        get_median10_report(stats, filepath=f'{args.report_path_prefix}.{args.stats_type}.median10_mf1.{args.max_order}gram',
                            print_report=args.print_report, metric='mf1',
                            sys1_name=args.sys1_name, sys2_name=args.sys2_name)
    if 'median10_bleu' in args.report_type:
        get_median10_report(stats, filepath=f'{args.report_path_prefix}.{args.stats_type}.median10_bleu.{args.max_order}gram',
                            print_report=args.print_report, metric='bleu',
                            sys1_name=args.sys1_name, sys2_name=args.sys2_name)
    if 'same_bleu_diff_mf1' in args.report_type:
        get_same_bleu_diff_mf1_report(stats,
                                      f'{args.report_path_prefix}.{args.stats_type}.same_bleu_diff_mf1.{args.max_order}gram',
                                      print_report=args.print_report,
                                      sys1_name=args.sys1_name, sys2_name=args.sys2_name)
