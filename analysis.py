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


def get_word_counts_of_passage_sentences(passage):
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sent_tokenizer.tokenize(passage)
    length = []
    for s in sentences:
        words = [w for w in nltk.word_tokenize(s) if not re.match(
            '^['+string.punctuation+']+$', w)]
        length.append(len(words))
    return length


def get_words_of_passage_senteneces(passage):
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sent_tokenizer.tokenize(passage)
    words_vec = []
    for s in sentences:
        words = [w for w in nltk.word_tokenize(s) if not re.match(
            '^['+string.punctuation+']+$', w)]
        words_vec.append(words)
    return words_vec


def extract_from_pos_tags(tags):
    def extractor(types, count=True):
        if count:
            return sum([
                len(filter(lambda x: x[1] in types, k)) for k in tags
            ])
        else:
            return sum([
                [y[0]] for k in tags for y in filter(lambda x: x[1] in types, k)
            ], [])
    return extractor

word_rank = {}
line_cnt = 1
with io.open('uniq_coca.csv') as f:
    for line in f.readlines():
        word_rank[line.strip()] = line_cnt
        line_cnt += 1
    print("get coca.csv done, coca size: {}".format(len(word_rank)))


def get_coca_rank(word):
    try:
        return word_rank[word]
    except KeyError as e:
        return -1


t = Tagger()
with io.open('titles_and_texts_revised') as f:
    all_text = f.read()
    splitted_text = all_text.split('\n\n')
    #print "len(splitted_text):", len(splitted_text)
    assert(len(splitted_text) == 60)
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

        u_n = i / 20 + 1
        u[i] = {
            'unit': u_n,
            'lesson': i + 1,
            #'title': title.encode('utf-8'),
            'setences': len(sentence),
            'passive_cnt': len(filter(lambda x: t.is_passive(" ".join(x)), sentence)),
            # 以下几个词根的定义请参照:
            # https://www.ibm.com/support/knowledgecenter/zh/SS5RWK_3.5.0/com.ibm.discovery.es.ta.doc/iiysspostagset.htm
            # 连词. 表因果的 Though, 和表并列的 and, or 的数量
            'conjunction': extractor(['IN', 'CC']),
            # 'WH-word' When, Why, What, Who, Which ...
            #'WDTc': extractor(['WDT'], False),
            'WDT': extractor(['WDT']),
            'WP': extractor(['WP']),
            'WP$': extractor(['WP$']),
            'WRB': extractor(['WRB']),
            'VBD': extractor(['VBD']),
            'VBN': extractor(['VBN']),
            'VBG': extractor(['VBG']),

            'DT': extractor(['DT']),
            'QT': extractor(['QT']),
            'CD': extractor(['CD']),
            #'CDc': extractor(['CD'], False),
            'JJ': extractor(['JJ']),
            'JJR': extractor(['JJR']),
            'JJS': extractor(['JJS']),


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
        u[i]['nonfinite_verb'] = u[i]['VBD'] + u[i]['VBN'] + u[i]['VBG']
        u[i]['wh_word'] = u[i]['WDT'] + u[i]['WP'] + u[i]['WP$'] + u[i]['WRB']
        u[i]['jj_total'] = u[i]['JJ'] + u[i]['JJR'] + u[i]['JJS']


for i in xrange(len(splitted_text)):
    print(u[i])

fig, axe = plt.subplots(1, 1)
plt.plot([x['words'] for x in u], 'ko--')
axe.set_xticks([x for x in xrange(-1, 61)])
axe.set_xticklabels([str(x) for x in xrange(0, 62)],
                    rotation=-30, fontsize='small')
print(plt.xlim())


def draw_pics(row, col, names):
    n_i = 0
    fig, axe = plt.subplots(row, col)
    for i in xrange(row):
        for j in xrange(col):
            if n_i >= len(names):
                return
            label_name = names[n_i]
            axe[i, j].plot([x[label_name] for x in u], 'ko--')
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

draw_pics(2, 2, ['FKS', 'conjunction', 'wh_word', 'nonfinite_verb'])

draw_pics(2, 2, ['WDT', 'WP', 'WP$', 'WRB'])

draw_pics(2, 2, ['VBD', 'VBN', 'VBG'])

draw_pics(2, 2, ['DT', 'QT', 'CD'])

draw_pics(2, 2, ['JJ', 'JJR', 'JJS', 'jj_total'])

plt.pause(10000)
