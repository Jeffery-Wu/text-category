# -*- coding:utf8 -*-
import sys

# reload(sys)
# sys.setdefaultencoding('utf8')
sys.path.append('..')
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import math
import numpy as np


def tf_mrf(corpus, test, package):
    """
    dictlist: 目录名——文档——词表 的三层词典嵌套， 以label为KEY索引到文本列表，每个文本是词语列表，词语列表包含该词在这篇文本中的频次
    doclist: 目录名——词表——文档 的三层词典嵌套， 以label为KEY索引到词表，每个词对应包含的文档名集合
    worddict : 词——文档 词典， 每个词对应 包含该词语的文档数目（不看label）
    """
    labelset = package["labelset"]
    weights = package["weights"]
    doclist = package["doclist"]

    # 字典要声明
    doclen = {}
    dictlist = {}
    doclist = {}
    worddict = {}
    totaldoc = len(corpus)

    for i in corpus:
        labell = i["label"]
        docl = i["document"]
        doclen[docl] = i["length"]
        type(doclen)
        if labell not in doclen:
            doclen[labell] = {}
        if docl not in doclen[labell]:
            doclen[labell][docl] = 0
        doclen[labell][docl] += i["length"]

        for j in i["split_sentence"]:
            # 计算 dictlist : label —— doc —— word —— frequency
            if labell not in dictlist:
                dictlist[labell] = {}
            if docl not in dictlist[labell]:
                dictlist[labell][docl] = {}
            if j not in dictlist[labell][docl]:
                dictlist[labell][docl][j] = 0
            dictlist[labell][docl][j] += 1
            if test == 0:
                # 计算 doclist : label —— word ——　doc set
                if labell not in doclist:
                    doclist[labell] = {}
                if j not in doclist[labell]:
                    doclist[labell][j] = set()
                doclist[labell][j].add(docl)

    if test == 0:

        # 获取每个词出现的文档数目（不看labell）
        # a = len(docllist[labell][word])
        # b = word出现的文档数目 - a
        # c = 用for求sum(doclist[label][!word])
        # rf = a/b

        # 按类别统计文档数量
        for labell in labelset:
            weights[labell] = {}
            for word in doclist[labell]:
                if word not in worddict:
                    worddict[word] = 0
                worddict[word] += len(doclist[labell][word])

        for labell in labelset:
            weights[labell] = {}
            for word in doclist[labell]:
                a_b = worddict[word]
                a = len(doclist[labell][word])
                b = a_b - a
                # c = sum([len(doclist[labell][x]) for x in (doclist[labell]) if x != word])
                weights[labell][word] = math.log(2 + a * 1.0 / max(1, b * 1.0), 2)

    # 计算 tf-rf 值
    tf_rf = {}
    for labell in labelset:
        tf_rf[labell] = {}
        for doc in dictlist[labell]:
            tf_rf[labell][doc] = {}
            for word in dictlist[labell][doc]:
                # print doc + word
                tf_rf[labell][doc][word] = dictlist[labell][doc][word] * 1.0 / (doclen[labell][doc] * 1.0)
                if word in weights[labell]:
                    tf_rf[labell][doc][word] *= weights[labell][word]
                else:
                    tf_rf[labell][doc][word] *= max([weights[x][word] for x in weights if word in weights[x]])
    package["labelset"] = labelset
    package["weights"] = weights
    package["doclist"] = doclist
    return tf_rf
