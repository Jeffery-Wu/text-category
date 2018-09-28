# -*- coding:utf8 -*-
import sys

# reload(sys)
# sys.setdefaultencoding('utf8')
sys.path.append('..')
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import math
import numpy as np


def selection_chi(corpus, test, package):
    labelset = package["labelset"]
    chi_value = package["chi_value"]
    doclist = package["doclist"]

    dictlist = {}
    doclen = {}
    doclist = {}
    totaldoc = len(corpus)
    worddict = {}

    # labell : 目录名
    # label: 目录列表
    # dictlist: 目录名——文档——词表 的三层词典嵌套， 以label为KEY索引到文本列表，每个文本是词语列表，词语列表包含该词在这篇文本中的频次
    # doclist: 目录名——词表——文档 的三层词典嵌套， 以label为KEY索引到词表，每个词对应包含的文档名集合
    # worddict : 词——文档 词典， 每个词对应 包含该词语的文档数目（不看label）
    print()
    for i in corpus:
        # print("corpus内容：")
        # print(i)
        labell = i["label"]
        docl = i["document"]
        doclen[docl] = i["length"]
        for j in i["split_sentence"]:

            # 计算 dictlist : label —— doc —— word —— frequency
            if labell not in dictlist:
                dictlist[labell] = {}
            if docl not in dictlist[labell]:
                dictlist[labell][docl] = {}
            if j not in dictlist[labell][docl]:
                dictlist[labell][docl][j] = 1
            else:
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
        # d = sum( !word 出现的文档数目 ) - c
        # chi = totaldoc * sqrt(ad-bc) / ( (a+b)*(a+c)*(d+b)*(c+d)  )

        # 按类别统计文档数量
        for labell in labelset:
            chi_value[labell] = {}
            for word in doclist[labell]:
                if word not in worddict:
                    worddict[word] = 0
                worddict[word] += len(doclist[labell][word])

        # 计算 chi-square
        for labell in labelset:
            for word in doclist[labell]:
                a_b = worddict[word]
                a = len(doclist[labell][word])
                b = a_b - a
                c_d = sum([(worddict[x]) for x in worddict.keys() if x != word])
                c = sum([len(doclist[labell][x]) for x in (doclist[labell]) if x != word])
                d = c_d - c
                chi_value[labell][word] = totaldoc * 1.0 * (a * d - b * c) * (a * d - b * c) / (
                        a_b * c_d * (b + d) * (a + c))

            # print chi

    package["chi_value"] = chi_value

    return chi_value
