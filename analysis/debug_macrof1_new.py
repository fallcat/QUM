import sacrebleu
import numpy as np
import pickle
from tqdm import tqdm

# snmt_path = '/Users/weiqiuyou/Documents/USC_ISI/translations/snmt/de-en/newstest2019.1m64k.out.txt.detok'
# unmt_path = '/Users/weiqiuyou/Documents/USC_ISI/translations/unmt/de-en/test.translation.en.debpe.detok'
# ref_path = '/Users/weiqiuyou/Documents/USC_ISI/data/refs/newstest2019-ende-src.en'
# src_path = '/Users/weiqiuyou/Documents/USC_ISI/data/refs/newstest2019-ende-ref.de'
#
# with open(snmt_path, 'rt') as snmt_file:
#     snmt_list = [line.strip() for line in snmt_file.readlines()]
#
# with open(ref_path, 'rt') as ref_file:
#     ref_list = [line.strip() for line in ref_file.readlines()]
#
# mf1 = sacrebleu.corpus_rebleu2(snmt_list, [ref_list], average='macro', word_class=True)
# print(mf1.score)
#
# sys = ['The dog runs home hi.', 'the dog runs home hi.']
# ref = ['The dog ran home hi.', 'the dog runs home hello.']
# mf1 = sacrebleu.corpus_rebleu2(sys, [ref], average='macro', word_class=True)
# mf1_old = sacrebleu.corpus_rebleu2(sys, [ref], average='macro', word_class=False)
# print(mf1.score)
# print(mf1_old.score)


sys = ['The dog runs home now.']
ref = ['The dog ran home now.']
mf1 = sacrebleu.corpus_rebleu2(sys, [ref], average='macro', word_class=True)
mf1_f1 = sacrebleu.corpus_rebleu2(sys, [ref], average='macro', word_class=True, measure_name='f1')
mf1_old = sacrebleu.corpus_rebleu2(sys, [ref], average='macro', word_class=False)
mf1_old_f1 = sacrebleu.corpus_rebleu2(sys, [ref], average='macro', word_class=False, measure_name='f1')
chrf = sacrebleu.corpus_chrf(sys, ref)
print("sys:", sys)
print("ref:", ref)
print("mf1 new:", mf1.score)
print("mf1 f1 new:", mf1_f1.score)
print("mf1 old:", mf1_old.score)
print("mf1 f1 old:", mf1_old_f1.score)
print("chrf:", chrf)

print("-------------")

sys = ['The dog runs home now.']
ref = ['The dog runs home later.']
mf1 = sacrebleu.corpus_rebleu2(sys, [ref], average='macro', word_class=True)
mf1_f1 = sacrebleu.corpus_rebleu2(sys, [ref], average='macro', word_class=True, measure_name='f1')
mf1_old = sacrebleu.corpus_rebleu2(sys, [ref], average='macro', word_class=False)
mf1_old_f1 = sacrebleu.corpus_rebleu2(sys, [ref], average='macro', word_class=False, measure_name='f1')

print("mf1 new:", mf1.score)
print("mf1 f1 new:", mf1_f1.score)
print("mf1 old:", mf1_old.score)
print("mf1 f1 old:", mf1_old_f1.score)