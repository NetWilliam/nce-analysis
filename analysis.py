#!/usr/bin/env python
#! coding: utf-8

import io
import re
import nltk
import string
import itertools

import numpy as np
import matplotlib.pyplot as plt


from textstat.textstat import textstat
from ispassive.ispassive import Tagger
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def get_words_of_passage_senteneces(passage):
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sent_tokenizer.tokenize(passage)
    words_vec = []
    for s in sentences:
        words = [w for w in nltk.word_tokenize(s) if not re.match(
            '^['+string.punctuation+']+$', w)]
        words_vec.append(words)
    return words_vec


word_rank = {}
line_cnt = 1
with io.open('uniq_coca.csv') as f:
    for line in f.readlines():
        word_rank[line.strip()] = line_cnt
        line_cnt += 1
    print("get coca.csv done, coca size: {}".format(len(word_rank)))


lemmatizer = WordNetLemmatizer()


def extract_from_pos_tags(tags, calc_word_freq=False):
    def extractor1(types, count=True):
        if count:
            return sum([
                len(filter(lambda x: x[1] in types, k)) for k in tags
            ])
        else:
            return sum([
                [y[0]] for k in tags for y in filter(lambda x: x[1] in types, k)
            ], [])

    def get_word_freq_rank(word_tuple):
        word = word_tuple[0]
        if word_tuple[1].startswith('NN'):
            return -1
        pos = get_wordnet_pos(word_tuple[1])

        if pos is not None:
            word = lemmatizer.lemmatize(word, pos)
        try:
            return word_rank[word]
        except KeyError as e:
            print("word: {} lemma: {} rank -1".format(word_tuple, word))
            return -1

    def extractor2(freq, count=True):
        if count:
            return sum([
                len(filter(lambda x: get_word_freq_rank(x) > freq, k)) for k in tags
            ])
        else:
            return sum([
                [y[0]] for k in rags for y in filter(lambda x: get_word_freq_rank(x) > freq, k)
            ], [])
    if calc_word_freq == False:
        return extractor1
    else:
        return extractor2


def get_coca_rank(word):
    try:
        return word_rank[word]
    except KeyError as e:
        return -1


t = Tagger()

LEXICAL_ATTR = [
    "UNKNOWN",  # 未知词
    "DT",  # 限定词
    "QT",  # 量词
    "CD",  # 基数
    "NN",  # 名词（单数）
    "NNS",  # 名词（复数）
    "NNP",  # 专有名词（单数）
    "NNPS",  # 专有名词（复数）
    "EX",  # 表示存在性的 there，例如在 There was a party 句子中。
    "PRP",  # 人称代词 (PP)
    "PRP$",  # 物主代词 (PP$)
    "POS",  # 所有格结束词
    "RBS",  # 副词（最高级）
    "RBR",  # 副词（比较级）
    "RB",  # 副词
    "JJS",  # 形容词（最高级）
    "JJR",  # 形容词（比较级）
    "JJ",  # 形容词
    "MD",  # 情态动词
    "VB",  # 动词（基本形式）
    "VBP",  # 动词（现在时态，非第三人称单数）
    "VBZ",  # 动词（现在时态，第三人称单数）
    "VBD",  # 动词（过去时态）
    "VBN",  # 动词（过去分词）
    "VBG",  # 动词（动名词或现在分词）
    "WDT",  # Wh 限定词，例如 Which book do you like better 句子中的 which
    "WP",  # Wh 代词，例如用作关系代词的 which 和 that
    "WP$",  # wh 物主代词，例如 whose
    "WRB",  # Wh 副词，例如 I like it when you make dinner for me 句子中的 when
    "TO",  # 介词 to
    "IN",  # 介词或从属连词
    "CC",  # 并列连词
    "UH",  # 感叹词
    "RP",  # 小品词
    "SYM",  # 符号
    "$",  # 货币符号
    "''",  # 双引号或单引号
    "(",  # 左圆括号、左方括号、左尖括号或左花括号
    ")",  # 右圆括号、右方括号、右尖括号或右花括号
    ",",  # 逗号
    ".",  # 句末标点符号 (. ! ?)
    ":",  # 句中标点符号 (: ; ... -- -)
]

chinese_translations = []

with io.open('chinese_translation', encoding='utf-8') as f:
    all_text = f.read()
    chinese_translations = filter(lambda x: x, all_text.split(u"参考译文"))
    print("len(chinese_translations): {}".format(len(chinese_translations)))
    assert(len(chinese_translations) == 60)


with io.open('titles_and_texts_revised') as f:
    all_text = f.read()
    splitted_text = all_text.split('\n\n')
    #print "len(splitted_text):", len(splitted_text)
    assert(len(splitted_text) == 60)
    assert(len(splitted_text) == len(chinese_translations))
    #u1 = []
    #u2 = []
    #u3 = []
    #u = [[], [], []]
    u = [{} for i in xrange(60)]
    #for lesson in splitted_text:
    for i in xrange(len(splitted_text)):
        lesson = splitted_text[i]
        lesson = lesson.split('\n')
        title = lesson[0]
        text = '\n'.join(lesson[1:])
        #print "title:", title.encode('utf-8')
        #print "word count:", get_word_counts_of_passage_sentences(text)
        sentence = get_words_of_passage_senteneces(text)
        words_cnt = [len(s) for s in sentence]
        #pos_tags = filter(lambda x: nltk.pos_tag(x), sentence)
        pos_tags = [nltk.pos_tag(x) for x in sentence]
        extractor = extract_from_pos_tags(pos_tags)
        word_extractor = extract_from_pos_tags(pos_tags, calc_word_freq=True)

        u_n = i / 20 + 1
        lexical_attr = {key: extractor([key]) for key in LEXICAL_ATTR}
        u[i] = {
            'unit': u_n,
            'lesson': i + 1,
            #'title': title.encode('utf-8'),
            'sentences': len(sentence),
            'translation': len(chinese_translations[i]),
            'passive_cnt': len(filter(lambda x: t.is_passive(" ".join(x)), sentence)),
            # 以下几个词根的定义请参照:
            # https://www.ibm.com/support/knowledgecenter/zh/SS5RWK_3.5.0/com.ibm.discovery.es.ta.doc/iiysspostagset.htm
            # 连词. 表因果的 Though, 和表并列的 and, or 的数量
            #'conjunction': extractor(['IN', 'CC']),
            # 'WH-word' When, Why, What, Who, Which ...
            #'WDTc': extractor(['WDT'], False),
            # Wh词
            'lexical_attr': lexical_attr,
            'beyound8000': word_extractor(8000),
            'beyound10000': word_extractor(10000),
            'beyound12000': word_extractor(12000),


            'words': sum(words_cnt),
            'avg': round(np.average(words_cnt), 2),
            'median': round(np.median(words_cnt), 2),
            'std': round(np.std(words_cnt), 2),
            #'flesch_kincaid_score': textstat.flesch_reading_ease(text),
            'FKS': textstat.flesch_reading_ease(text),
            #'SMOG': textstat.smog_index(text),
            #'automated_readability_index': textstat.automated_readability_index(text),
            #'ARI': textstat.automated_readability_index(text),

        }
        u[i]['translate_per_word'] = round(
            float(u[i]['translation']) / u[i]['words'], 4)
        u[i]['translate_per_sentence'] = u[i]['translation'] / u[i]['sentences']
        u[i]['lexical_attr']['nonfinite_verb'] = u[i]['lexical_attr']['VBD'] + \
            u[i]['lexical_attr']['VBN'] + u[i]['lexical_attr']['VBG']
        u[i]['lexical_attr']['wh_word'] = u[i]['lexical_attr']['WDT'] + \
            u[i]['lexical_attr']['WP'] + u[i]['lexical_attr']['WP$'] + \
            u[i]['lexical_attr']['WRB']
        u[i]['lexical_attr']['jj_total'] = u[i]['lexical_attr']['JJ'] + \
            u[i]['lexical_attr']['JJR'] + u[i]['lexical_attr']['JJS']
        u[i]['lexical_attr']['rb_total'] = u[i]['lexical_attr']['RB'] + \
            u[i]['lexical_attr']['RBR'] + u[i]['lexical_attr']['RBS']
        u[i]['lexical_attr']['jjrb'] = u[i]['lexical_attr']['jj_total'] + \
            u[i]['lexical_attr']['rb_total']
        u[i]['lexical_attr']['jjrb_percentage'] = round(
            float(u[i]['lexical_attr']['jjrb']) / u[i]['words'], 4)
        u[i]['lexical_attr']['vb_dng'] = u[i]['lexical_attr']['VBD'] + \
            u[i]['lexical_attr']['VBN'] + u[i]['lexical_attr']['VBG']
        u[i]['lexical_attr']['v_adj_adv'] = u[i]['lexical_attr']['vb_dng'] + \
            u[i]['lexical_attr']['jj_total'] + u[i]['lexical_attr']['rb_total']
        u[i]['lexical_attr']['v_adj_adv_percentage'] = round(
            float(u[i]['lexical_attr']['v_adj_adv']) / u[i]['words'], 4)


for i in xrange(len(splitted_text)):
    print(u[i])

fig, axe = plt.subplots(1, 1)
plt.plot([x['words'] for x in u], 'ko--')
axe.set_xticks([x for x in xrange(-1, 61)])
axe.set_xticklabels([str(x) for x in xrange(0, 62)],
                    rotation=-30, fontsize='small')
print(plt.xlim())


def draw_pics(row, col, names, getter=None):
    n_i = 0
    fig, axe = plt.subplots(row, col)
    if row == 1 and col == 1:
        label_name = names[n_i]
        if getter is None:
            axe.plot([x[label_name] for x in u], 'ko--')
        else:
            axe.plot([getter(x, label_name) for x in u], 'ko--')
        axe.set_xlabel(label_name)
        return

    for i in xrange(row):
        for j in xrange(col):
            if n_i >= len(names):
                return
            label_name = names[n_i]
            if getter is None:
                axe[i, j].plot([x[label_name] for x in u], 'ko--')
            else:
                axe[i, j].plot([getter(x, label_name) for x in u], 'ko--')
            axe[i, j].set_xlabel(label_name)
            n_i += 1


fig, axe = plt.subplots(2, 2)
axe[0, 0].plot([x['avg'] for x in u], 'ko--')
axe[0, 0].set_xlabel('avg')
axe[0, 1].plot([x['median'] for x in u], 'ko--')
axe[0, 1].set_xlabel('median')
axe[1, 0].plot([x['std'] for x in u], 'ko--')
axe[1, 0].set_xlabel('std')
axe[1, 1].plot([x['passive_cnt']
                for x in u], 'ko--')
axe[1, 1].set_xlabel('passive_cnt')
#for i in xrange(2):
#    for j in xrange(2):
#        axe[i, j].set_xticks([x for x in xrange(-1, 61)])
#        axe[i, j].set_xticklabels([str(x) for x in xrange(0, 62)],
#                        rotation=-30, fontsize='small')

#draw_pics(2, 2, ['FKS', 'conjunction', 'wh_word', 'nonfinite_verb'])


def lexical_attr_getter(x, label): return x['lexical_attr'][label]


draw_pics(2, 2, ['WDT', 'WP', 'WP$', 'WRB'], lexical_attr_getter)

draw_pics(2, 2, ['VBD', 'VBN', 'VBG', 'vb_dng'], lexical_attr_getter)

draw_pics(2, 2, ['DT', 'QT', 'CD'], lexical_attr_getter)

draw_pics(2, 2, ['JJ', 'JJR', 'JJS', 'jj_total'], lexical_attr_getter)

draw_pics(2, 2, ['RB', 'RBR', 'RBS', 'rb_total'], lexical_attr_getter)

draw_pics(1, 1, ['jjrb'], lexical_attr_getter)

draw_pics(1, 1, ['jjrb_percentage'], lexical_attr_getter)

draw_pics(1, 1, ['v_adj_adv'], lexical_attr_getter)

draw_pics(1, 1, ['v_adj_adv_percentage'], lexical_attr_getter)

draw_pics(2, 2, ['beyound8000', 'beyound10000', 'beyound12000'])

draw_pics(2, 2, ['translation', 'translate_per_word',
                 'translate_per_sentence'])

plt.pause(10000)
