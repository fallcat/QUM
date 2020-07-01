"""
Analyze macrof1 and bleu

Usage:

Example:
1) get delta stats of SNMT and UNMT system:
python analysis/macrof1_analysis.py \
  --sys1-path ../../translations/snmt/de-en/newstest2019.1m64k.out.txt.detok \
  --sys2-path ../../translations/unmt/de-en/test.translation.en.debpe.detok \
  --ref-path ../../data/refs/newstest2019-ende-src.en \
  --src-path ../../data/refs/newstest2019-ende-ref.de \
  --save-stats-path stats/deen.delta.stats.pkl \
  --stats-type delta

2) load delta stats and get a full report
python analysis/macrof1_analysis.py \
  --load-stats-path stats/deen.stats.pkl \
  --stats-type apply_delta \
  --report-type top10_mf1 top10_bleu median10_mf1 median10_bleu same_bleu_diff_mf1 \
  --report-path-prefix reports/deen

3) just get bleu and mf1
python analysis/macrof1_analysis.py \
  --sys1-path ../../translations/snmt/de-en/newstest2019.1m64k.out.txt.detok \
  --sys2-path ../../translations/unmt/de-en/test.translation.en.debpe.detok \
  --ref-path ../../data/refs/newstest2019-ende-src.en \
  --report-type total_metrics \
  --report-path-prefix reports/deen

4) print all sentences base on a metric
python analysis/macrof1_analysis.py \
  --load-stats-path stats/deen.delta.stats.pkl \
  --stats-type apply_delta \
  --save-stats-path stats/deen.apply_delta.stats.pkl \
  --report-type all_mf1 \
  --report-path-prefix reports/deen
"""


import pickle
import argparse
import sacrebleu
import numpy as np
from tqdm import tqdm


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
                                                                                        'top10_mf1', 'top10_bleu',
                                                                                        'median10_mf1', 'median10_bleu',
                                                                                        'same_bleu_diff_mf1', 'total_metrics'])
    parser.add_argument('--print-report', default=False, action='store_true',
                        help='whether or not to print the report')

    parser.add_argument('--report-path-prefix', type=str, default=None, help='path to store the reports')

    return parser


def get_total_metrics(sys1_path, sys2_path, ref_path, lowercase=False, max_order=1, sys1_name='sys1', sys2_name='sys2', filepath=None):
    with open(sys1_path, 'rt') as sys1_file:
        with open(sys2_path, 'rt') as sys2_file:
            with open(ref_path, 'rt') as ref_file:
                sys1_list = [line.strip() for line in sys1_file.readlines()]
                sys2_list = [line.strip() for line in sys2_file.readlines()]
                ref_list = [line.strip() for line in ref_file.readlines()]

    mf1_sys1 = sacrebleu.corpus_rebleu(sys1_list, [ref_list], lowercase=lowercase, average='macro', max_order=max_order)
    mf1_sys2 = sacrebleu.corpus_rebleu(sys2_list, [ref_list], lowercase=lowercase, average='macro', max_order=max_order)
    bleu_sys1 = sacrebleu.corpus_bleu(sys1_list, [ref_list], lowercase=lowercase)
    bleu_sys2 = sacrebleu.corpus_bleu(sys2_list, [ref_list], lowercase=lowercase)

    report = ''
    report += f'mf1_{sys1_name}: {mf1_sys1}\n'
    report += f'mf1_{sys2_name}: {mf1_sys2}\n'
    report += f'bleu_{sys1_name}: {bleu_sys1}\n'
    report += f'bleu_{sys2_name}: {bleu_sys2}\n'

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
    mf1_diff = []
    bleus = []
    bleu_diff = []

    for i in tqdm(range(len(sys1_list))):
        # get macro f1
        sys1_sublist = sys1_list[:i] + sys1_list[i + 1:]
        sys2_sublist = sys2_list[:i] + sys2_list[i + 1:]
        ref_sublist = ref_list[:i] + ref_list[i + 1:]
        mf1_sys1 = sacrebleu.corpus_rebleu(sys1_sublist, [ref_sublist], lowercase=lowercase, average='macro', max_order=max_order)
        mf1_sys2 = sacrebleu.corpus_rebleu(sys2_sublist, [ref_sublist], lowercase=lowercase, average='macro', max_order=max_order)
        mf1s.append([mf1_sys1.score, mf1_sys2.score])
        mf1_diff.append(mf1_sys1.score - mf1_sys2.score)

        # get bleu
        bleu_sys1 = sacrebleu.corpus_bleu(sys1_sublist, [ref_sublist], lowercase=lowercase)
        bleu_sys2 = sacrebleu.corpus_bleu(sys2_sublist, [ref_sublist], lowercase=lowercase)
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

# return {'sys1': sys1_list,
#             'sys2': sys2_list,
#             'ref': ref_list,
#             'src': src_list,
#             'mf1s': mf1s,
#             'mf1_diff': mf1_diff,
#             'mf1_ranked_indices': mf1_ranked_indices,
#             'bleus': bleus,
#             'bleu_diff': bleu_diff,
#             'bleu_ranked_indices': bleu_ranked_indices}


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
    n = len(ref_list)

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
        else:
            idx = bleu_ranked_indices[i]
        report.append('\t'.join([str(idx), src_list[idx], ref_list[idx], sys1_list[idx], sys2_list[idx],
                      *[str(x) for x in mf1s[idx]], str(mf1_diff[idx]), *[str(x) for x in bleus[idx]], str(bleu_diff[idx])]) + '\n')

    if fileprefix is not None:
        with open(fileprefix + '.tsv', 'wt') as output_file:
            output_file.writelines(report)
        with open(fileprefix + '.txt', 'wt') as output_file:
            if metric == 'mf1':
                metrics = mf1s
                metric_diff = mf1_diff
            else:
                metrics = bleus
                metric_diff = bleu_diff
            for i in range(n):
                if metric == 'mf1':
                    idx = mf1_ranked_indices[i]
                else:
                    idx = bleu_ranked_indices[i]
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
            stats = apply_delta(stats, lowercase=args.lc)
        else:
            if args.stats_type == 'single':
                stats = get_stats(args.sys1_path, args.sys2_path, args.ref_path, args.src_path, lowercase=args.lc, max_order=args.max_order)
            else:
                stats = get_delta_stats(args.sys1_path, args.sys2_path, args.ref_path, args.src_path, lowercase=args.lc, max_order=args.max_order)
                if args.stats_type == 'apply_delta':
                    stats = apply_delta(stats, lowercase=args.lc)
        if args.save_stats_prefix is not None:
            with open(f'{args.save_stats_prefix}.{args.stats_type}.{args.max_order}gram.stats.pkl', 'wb') as output_file:
                pickle.dump(stats, output_file)
    else:
        if args.load_stats_path is not None:
            with open(args.load_stats_path, 'rb') as input_file:
                stats = pickle.load(input_file)
        else:
            assert args.report_type == ['total_metrics']

    # get reports
    # ['top10_mf1', 'top10_bleu', 'median10_mf1', 'median10_bleu', 'same_bleu_diff_mf1']
    if 'total_metrics' in args.report_type:
        get_total_metrics(args.sys1_path, args.sys2_path, args.ref_path, lowercase=args.lc, max_order=args.max_order,
                          sys1_name=args.sys1_name, sys2_name=args.sys2_name,
                          filepath=f'{args.report_path_prefix}.{args.stats_type}.total_metrics.{args.max_order}gram')

    if 'all_mf1' in args.report_type:
        get_all_report(stats, fileprefix=f'{args.report_path_prefix}.{args.stats_type}.all.{args.max_order}gram',
                       print_report=args.print_report, metric='mf1')

    if 'all_bleu' in args.report_type:
        get_all_report(stats, fileprefix=f'{args.report_path_prefix}.{args.stats_type}.all.{args.max_order}gram',
                       print_report=args.print_report, metric='bleu')

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
