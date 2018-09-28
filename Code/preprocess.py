# -*- coding:utf8 -*-
import sys

# reload(sys)
# sys.setdefaultencoding('utf8')
from feature_selection import selection_chi

sys.path.append('..')
import os
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def preprocess(input, package, test=0, k=500):
    vocafreq = package["vocafreq"]
    voca = package["voca"]
    labelset = package["labelset"]
    chi_value = package["chi_value"]


    input = np.array(open(input).read().split("\n"))
    corpus = []

    doccount = {}  # 计数
    for i in input:
        document = {}
        sp = i.split("\t")  # \t--tab 下一制表符  0为标题 1为正文
        # print("sp1:\n")
        # print(sp)
        label = sp[0]
        document["label"] = label
        if label not in doccount:
            doccount[label] = 0
        doccount[label] += 1
        docname = label + str(doccount[label])
        document["document"] = docname  # 对label计数  将document名改为label+“数”
        sp = sp[1].split(" ")  # 将正文按空格分割为单词
        # print("sp2:\n")
        # print(sp)
        while " " in sp:
            sp.remove(" ")
        while "" in sp:
            sp.remove("")
        document["split_sentence"] = sp
        if test == 0:
            if label not in labelset:
                labelset.append(label)  # 得到label集合
            for word in sp:
                if word not in vocafreq:
                    vocafreq[word] = 0
                vocafreq[word] += 1  # 遍历  对各个word计数
        document["length"] = len(sp)  # document里为label name & content length
        # print("document:\n")
        # print(document)
        corpus.append(document)

    if test == 0:

        """
        特征选择
        """
        print("feature select")

        # test = 0
        # k = 10

        if chi_value == {}:
            chi_value = selection_chi(corpus, test, package)

        vocabularyfreq = {}

        for label in labelset:

            sorted_chi_value = sorted(chi_value[label], key=lambda x: chi_value[label][x], reverse=True)
            # sorted_chi_value = sorted(chi_value[label].items(), key=lambda x: x[1])
            sorted_chi_selection = sorted_chi_value[:k]

            for word in sorted_chi_selection:
                if word not in vocabularyfreq:
                    vocabularyfreq[word] = vocafreq[word]

        vocafreq = vocabularyfreq
        voca = vocabularyfreq.keys()

        # vocafreq = {x: vocafreq[x] for x in vocafreq if 5 < vocafreq[x] < 2000}

        # print("\nvocafreq:")
        # print(vocafreq)
        # print("\nvoca:")
        # print(voca)
        # print(len(voca))

        """
        ll = dict(Counter(vocafreq.values()))
        plt.hist(ll.values())
        plt.show()
        """

    for i in corpus:
        i["split_sentence"] = [x for x in i["split_sentence"] if x in vocafreq]

    # 为保证全局一致？  供测试集使用
    package["vocafreq"] = vocafreq
    package["voca"] = voca
    package["labelset"] = labelset
    package["chi_value"] = chi_value

    return corpus
