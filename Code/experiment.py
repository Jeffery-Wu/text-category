# -*- coding:utf8 -*-
# from blist._blist import *
import sys
import re
# sys.path.append('..')
import os
import math
# import memory_profiler

import psutil as psutil

from tf_vc import tf_vc

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import pandas as pd
from liblinearutil import *
from one_hot import one_hot
from collections import Counter
import matplotlib.pyplot as plt
from preprocess import preprocess
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier as KNN

from tf_idf import tf_idf
from tf_dc import tf_dc
from tf_bdc import tf_bdc
from tf_ig import tf_ig
from tf_eccd import tf_eccd
from tf_chi import tf_chi
from tf_rf import tf_rf
from iqf_qf_icf import iqf_qf_icf
from tf_mrf import tf_mrf
from tf_nrf import tf_nrf


"""
microaverage = accuracy = correct / total

macroaverage = avg(F1)

F1(C) = 2*precision*recall / (precision + recall)
precision = TP / (TP + FN)	： TP/预测为C的数目
recall = TP / (TP + TN)		： TP/真实为C的数目

TP: true positive  属于C被分到C（正确）
TN: true nagative  属于C被分到其它（错误）

FP: false positive 不属于C被正确分类（正确）
FN: false nagative 不属于C被分到C（错误）

"""


def init(package):
    package["voca"] = []
    package["labelset"] = []
    package["vocafreq"] = {}
    package["weights"] = {}

    package["doclist"] = []
    package["docname"] = set()


def getXY(input, algo, model, test=0, k=25):

    """
    input: 预处理过的语料库
    algo: 使用的特征权重计算方法名
    model: 使用的模型名

    test = 0 : 记录文件中出现的词汇并构造词汇表(训练集)
    test = 1 : 不构造词汇表，用已经构造好的(测试集)

    """
    global package

    # global voca_list
    # global labelset_list
    # global vocafreq_list
    # global weights_list
    # global doclist
    # global docname

    corpus = preprocess(input, package, test, k)
    labelset = package["labelset"]  # 获得preprocess确定的package
    voca = package["voca"]

    level = 2
    mod = 0
    if algo == "tf_idf":
        weights = tf_idf(corpus, test, package)
        mod = 1
    elif algo == "tf_dc":
        weights = tf_dc(corpus, test, package)
    elif algo == "tf_bdc":
        weights = tf_bdc(corpus, test, package)
    elif algo == "iqf_qf_icf":
        weights = iqf_qf_icf(corpus, test, package)
    elif algo == "tf_eccd":
        weights = tf_eccd(corpus, test, package)
    elif algo == "tf_ig":
        weights = tf_ig(corpus, test, package)
    elif algo == "tf_rf":
        weights = tf_rf(corpus, test, package)
        level = 3
    elif algo == "tf_chi":
        weights = tf_chi(corpus, test, package)
        level = 3
    elif algo == "tf_mrf":
        weights = tf_mrf(corpus, test, package)
        level = 3
    elif algo == "tf_nrf":
        weights = tf_nrf(corpus, test, package)
        level = 3
    elif algo == "tf_vc":
        weights = tf_vc(corpus, test, package)

    # print weights
    X = []
    Y = []  # 标签集
    count = 0
    vocalen = len(voca)
    for doc in corpus:
        if count % 1000 == 0:
            print(str(count) + "/" + str(len(corpus)))
            # print('weights\'s size:')
            # print(sys.getsizeof(weights))
            # print(sys.getsizeof(X))
            # process = psutil.Process(os.getpid())
            # print('Used Memory:', process.memory_info().rss / 1024 / 1024, 'MB')
            # print(memory_usage_psutil())
        count += 1

        # process label
        labelset.append(doc["label"])
        Y.append(int(np.argmax(one_hot(labelset)[-1])))  # 在确定的labelset中添加label，以保证label的位置一致，再进行截取
        # np.argmax返回最大数的索引
        labelset = labelset[:-1]  # 重置labelset

        # process word
        temvocalist = list(voca) + list(doc["split_sentence"])  # 与label同理  voca用以确定位置

        tem_one_hot = one_hot(temvocalist)[vocalen:]  # 截取

        # for word in range(len(tem_one_hot)):  # .shape[0]
        for word in range(tem_one_hot.shape[0]):
            temlabel = doc["label"]  # earn
            temword = doc["split_sentence"][word]
            temdoc = doc["document"]  # earn638

            # print("\ntem_one_hot:")
            # print(tem_one_hot)
            # print("\n")

            # weights--词频*权重
            if level == 2:
                if mod == 0:  # 有监督
                    tem_one_hot[word] *= weights[temlabel][temword]
                else:  # 无监督
                    tem_one_hot[word] *= weights[temdoc][temword]
            else:
                tem_one_hot[word] *= weights[temlabel][temdoc][temword]


        # 空array
        try:
            tem_one_hot = np.max(tem_one_hot, axis=0)  # 去除多余行  每列只保留最大数
        except ValueError:
            # tem_one_hot = tem_one_hot[0]
            # print(tem_one_hot)
            pass

        if model.lower() == "knn":
            tem_one_hot = preprocessing.normalize(np.array(tem_one_hot).reshape(1, -1), norm='l2')  # 转变为矩阵

        # print(tem_one_hot.toarray())
        # tem_one_hot = np.full(tem_one_hot)
        # print(tem_one_hot)  # 稀疏矩阵转化回原矩阵！
        # print(type(tem_one_hot.toarray()))

        X.append(np.squeeze(tem_one_hot.toarray().tolist()))
        # print(tem_one_hot.toarray().tolist())

        # X.append(tem_one_hot)

    # print(np.array(X))
    # print(Y)

    return X, Y  # squeeze压缩维度  如将二维转变为一维


def main(trainf, testf, algo="tf_idf", model="knn", knn_neighbour=0):
    """
    algo: 可选 tf_idf, tf_dc, tf_bdc, tf_ig, tf_chi, tf_eccd, tf_rf, iqf_qf_icf
    model: 可选 svm, knn
    knn_neighbour:
        为0：测试模式，选用 [1,5,10,15,20,25,30,35] 作为邻居数分别进行训练，文件输出正确率（可plot或导入evaluate程序）
        为n：使用邻居数为n进行训练，文件输出所有预测标签和正确标签，中间以\t分离
    """
    global package

    init(package)

    for k in [25, 50, 75, 150, 300, 600, 900, 1200, 1800, 2400]:

        package["weights"] = {}

        print("Training " + model + " with " + algo + " by " + str(k) + " features")
        print("Processing Training Set... ")
        train_x, train_y = getXY(trainf, algo, model, test=0, k=k)
        print("Finished! ")
        print("Processing Test Set... ")
        test_x, test_y = getXY(testf, algo, model, test=1, k=k)
        print("Finished! ")

        if model == "svm":
            prob = problem(train_y, train_x)
            param = parameter('-c 4 -B 1')  # c-代价参数 B-bias 偏差
            mmodel = train(prob, param)
            p_labels, p_acc, p_vals = predict(test_y, test_x, mmodel)
            print(str(p_acc))
            resultfile = open(model + "_" + algo + "_" + str(k) + "f.txt", "w")
            for i in range(len(p_labels)):
                resultfile.write(str(p_labels[i]) + " " + str(test_y[i]) + "\n")
            resultfile.close()

        elif model.lower() == "knn":
            if knn_neighbour != 0:
                Y = []
                clf = KNN(n_neighbors=knn_neighbour, weights='uniform')
                clf.fit(train_x, train_y)
                result = clf.predict(np.array(test_x))

                accuracy = sum(result == test_y) * 1.0 / len(test_y)
                Y.append(accuracy)  # delete

                print(str(accuracy))

                resultfile = open(model + "_" + algo + ".txt", "w")

                for i in range(len(result)):
                    resultfile.write(str(result[i]) + " " + str(test_y[i]) + "\n")
                resultfile.close()

            else:
                X = [1] + [i for i in range(5, 36, 5)]
                Y = []
                resultfile = open(model + "_" + algo + ".txt", "w")
                for i in X:
                    clf = KNN(n_neighbors=i, weights='uniform')
                    clf.fit(train_x, train_y)
                    result = clf.predict(np.array(test_x))
                    accuracy = sum(result == test_y) * 1.0 / len(test_y)
                    Y.append(accuracy)

                    str_nei = str(i)
                    print(str_nei)
                    print(accuracy)
                    resultfile.write(str_nei + "\t" + str(accuracy) + "\n")
                resultfile.close()
                plt.plot(X, Y)
                plt.show()


if __name__ == "__main__":
    # list = []
    # package_list = blist(list)
    package = {}
    package["chi_value"] = {}
    # np.set_printoptions(threshold = np.NaN)
    # trainf = "../Corpus/Reuters_train_1.txt"

    # 路透
    trainf = "../Corpus/Reuters_train.txt"
    testf = "../Corpus/Reuters_test.txt"

    # trainf = "../Corpus/Reuters_test.txt"


    # 20 newsgroup
    # trainf = "../Corpus/20_newsgroup_train.txt"
    # testf = "../Corpus/20_newsgroup_test.txt"

    # main(trainf, testf, algo="tf_idf", model="svm")
    # main(trainf, testf, algo="tf_dc", model="svm")
    # main(trainf, testf, algo="tf_bdc", model="svm")
    # main(trainf, testf, algo="iqf_qf_icf", model="svm")
    # main(trainf, testf, algo="tf_rf", model="svm")
    main(trainf, testf, algo="tf_chi", model="svm")
    # main(trainf, testf, algo="tf_eccd", model="svm")
    # main(trainf, testf, algo="tf_mrf", model="svm")
    # main(trainf, testf, algo="tf_nrf", model="svm")
    # main(trainf, testf, algo="tf_vc", model="svm")

    # knn_neighbour 选用 1-35之间MicroF1即准确率最高的
    # main(trainf, testf, algo="tf_idf", model="knn", knn_neighbour=30)
    # main(trainf, testf, algo="tf_dc", model="knn", knn_neighbour=5)
    # main(trainf, testf, algo="tf_bdc", model="knn", knn_neighbour=1)
    # main(trainf, testf, algo="iqf_qf_icf", model="knn", knn_neighbour=35)
    # main(trainf, testf, algo="tf_rf", model="knn", knn_neighbour=10)
    # main(trainf, testf, algo="tf_chi", model="knn", knn_neighbour=10)
    # main(trainf, testf, algo="tf_eccd", model="knn", knn_neighbour=5)
    # main(trainf, testf, algo="tf_mrf", model="knn", knn_neighbour=25)
    # main(trainf, testf, algo="tf_nrf", model="knn", knn_neighbour=25)
