# QUM
Qualitative Unsupervised NMT


Analyze macrof1 and bleu

Requirements:

Python 3.6

```
pip install -r requirements.txt
```

You need to also install sacrebleu from https://github.com/fallcat/sacre-BLEU/tree/bleu-recall.

Usage:

Example:
1) get delta stats of SNMT and UNMT system:
```
python analysis/macrof1_analysis.py \
  --sys1-path sys1/test.translated.en \
  --sys2-path sys2/test.translated.en \
  --sys1-name sys1 \
  --sys2-name sys2 \
  --ref-path refs/newstest2019-ende-src.en \
  --src-path refs/newstest2019-ende-ref.de \
  --save-stats-prefix stats/deen \
  --stats-type delta \
  --max-order 1
```

2) load delta stats and get a full report
```
python analysis/macrof1_analysis.py \
  --sys1-name sys1 \
  --sys2-name sys2 \
  --load-stats-path stats/deen.stats.pkl \
  --stats-type apply_delta \
  --report-type top10_mf1 top10_bleu median10_mf1 median10_bleu same_bleu_diff_mf1 \
  --report-path-prefix reports/deen \
  --max-order 1
```

3) just get bleu and mf1
```
python analysis/macrof1_analysis.py \
  --sys1-name sys1 \
  --sys2-name sys2 \
  --sys1-path sys1/test.translated.en \
  --sys2-path sys2/test.translated.en \
  --ref-path refs/newstest2019-ende-src.en \
  --src-path refs/newstest2019-ende-ref.de \
  --report-type total_metrics \
  --report-path-prefix reports/deen \
  --max-order 1
```

4) print all sentences base on a metric
```
python analysis/macrof1_analysis.py \
  --sys1-name sys1 \
  --sys2-name sys2 \
  --load-stats-path stats/deen.delta.stats.pkl \
  --stats-type apply_delta \
  --save-stats-prefix stats/deen \
  --report-type all_mf1 \
  --report-path-prefix reports/deen \
  --max-order 1
```