#!/usr/bin/env python
#! coding: utf-8

import io

titles = []
texts = []


def is_ascii(s):
    arr = [ord(c) < 128 for c in s]
    ascii_size = len([x for x in arr if x])
    all_size = len(arr)
    #all_size = len(s)
    threshold = 0.9
    #return all(ord(c) < 128 for c in s)
    return (float(ascii_size) / float(all_size)) > threshold, (float(ascii_size) / float(all_size))


#print "is_ascii('abc'):", is_ascii('abc')
#print "is_ascii('你好')", is_ascii('你好')
#exit(0)


with io.open("filtered_nce", encoding='utf-8') as f:
    all_text = f.read()
    splitted_text = all_text.encode('utf-8').split('\n\n')
    for i in xrange(len(splitted_text)):
        setences = splitted_text[i]
        if setences.find("Lesson") != -1:
            setence = setences.split('\n')
            for s_i in xrange(len(setence)):
                if setence[s_i].find("Lesson") != -1:
                    titles.append(
                        [setence[s_i], setence[s_i + 1], setence[s_i + 2]])
                    #titles.append(setence[s_i])
                    #titles.append(setence[s_i + 1])
                    #titles.append(setence[s_i + 2])
                    #print setence[s_i]
                    #print setence[s_i + 1]
                    #print setence[s_i + 2]
#            p_i = i - 1
#            if p_i >= 0:
#                prev_setences = splitted_text[p_i]
#                if len(next_setences) > 240:
#                    print next_setences
            #print "xxxxxxxxxxxxxxxxxxxxxxxxx"
        else:
            #print "len:", len(setences), " is_ascii:",  is_ascii(
            #    setences), setences
            if len(setences) > 180 and is_ascii(setences)[0]:
                texts.append(setences)
                #print setences
                pass

assert(len(titles) == len(texts))

#exit(0)
for i in xrange(len(titles)):
    print " ".join([s.strip() for s in titles[i]])
    print texts[i]
    print ""
