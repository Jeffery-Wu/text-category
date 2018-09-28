import operator
import unittest

# import memory_profiler
# from blist._blist import *

from preprocess import preprocess
from feature_selection import selection_chi
from one_hot import one_hot
from experiment import getXY


class prepocessTest(unittest.TestCase):

    def test_preprocess(self):
        package = {"voca": [], "labelset": [], "vocafreq": {}, "chi_value": {}, "doclist": {}}
        input = "../Corpus/Reuters_train_1.txt"

        corpus = preprocess(input, package)
        print("\npreprocess运行成功")
        # print(corpus)
        test = 0
        k = 10
        chi_value = selection_chi(corpus, test, package)
        print(chi_value)

        labelset = package["labelset"]
        vocafreq = package["vocafreq"]
        voca = package["voca"]

        vocabularyfreq = {}

        for label in labelset:

            sorted_chi_value = sorted(chi_value[label], key=lambda x: chi_value[label][x], reverse=True)
            # sorted_chi_value = sorted(chi_value[label].items(), key=lambda x: x[1])
            sorted_chi_selection = sorted_chi_value[:10]

            for word in sorted_chi_selection:
                if word not in vocabularyfreq:
                    vocabularyfreq[word] = vocafreq[word]


        vocafreq = vocabularyfreq
        voca = vocabularyfreq.keys()

        package["vocafreq"] = vocafreq
        package["voca"] = voca

        for i in corpus:
            i["split_sentence"] = [x for x in i["split_sentence"] if x in vocafreq]

        print(corpus)


    # def test_one_hot(self):
    #     labelset = ["a", "a", "b", "a", "c", "d", "a", "e", "a", "f", "g", "h"]
    #     one_hot(labelset)
    #
    # def test_getXY(self):
    #     trainf = "../Corpus/Reuters_train_1.txt"
    #     algo = "tf_idf"
    #     model = "svm"
    #     getXY(trainf, algo, model, test=0)


if __name__ == '__main__':
    unittest.main()
