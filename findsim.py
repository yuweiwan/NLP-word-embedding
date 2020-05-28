#!/usr/bin/env python3
import argparse
import heapq
import math
import numpy as np
from collections import defaultdict

from numpy import dot
from numpy.linalg import norm


# transform string in file to dictonary: key is word and value is its vector
def getVector(text_dir):
    with open(text_dir, 'r') as mf:
        data = mf.readlines()
    # delete the first line which is not word + vector
    del data[0]
    dic = {}
    for line in data:
        word = line.split('\t', 1)[0]
        string_of_num = line.split('\t', 1)[1]
        # convert string list to float list
        floats = np.array(string_of_num.split('\t'), dtype=float)
        dic.setdefault(word, floats)
    return dic


def cosine_similarity(v1, v2):
    # compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i];
        y = v2[i]
        sumxx += x * x
        sumyy += y * y
        sumxy += x * y
    return sumxy / math.sqrt(sumxx * sumyy)


# find vector of target word and compare the cosine similarity of other vectors
# then return the most similar one
def compare(text_dir, word1, word2, word3):
    vectors = getVector(text_dir)
    if word2 == 'NA' and word3 == 'NA':
        word2 = word1
        word3 = word1
    word_data1 = vectors[word1]
    word_data2 = vectors[word2]
    word_data3 = vectors[word3]
    cal_data = word_data1 - word_data2 + word_data3
    # create a dict to store word and its cos_sim with target word
    dic = {}
    # key is cos, value is word
    for k in vectors.keys():
        if k != word1 and k != word2 and k != word3:
            cos_sim = cosine_similarity(cal_data, vectors[k])
            dic.setdefault(cos_sim, k)
    words = []
    for i in range(10):
        max_cos = max(dic.keys())
        words.append(dic[max_cos])
        dic.pop(max_cos)
    print(" ".join(str(w) for w in words))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process filepath and integer.')
    parser.add_argument('filepath', type=str, help='a string of filepath')
    parser.add_argument('word1', type=str, help='the target word')
    parser.add_argument('--minus', type=str, help='the target word', default='NA')
    parser.add_argument('--plus', type=str, help='the target word', default='NA')
    args = parser.parse_args()
    getVector(args.filepath)
    compare(args.filepath, args.word1, args.minus, args.plus)
