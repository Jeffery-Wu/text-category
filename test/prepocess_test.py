import operator
import unittest

# import memory_profiler
# from blist._blist import *

from preprocess import preprocess
from feature_selection import selection_chi
from one_hot import one_hot
from experiment import getXY
from tf_vc import tf_vc


class prepocessTest(unittest.TestCase):

    def test_preprocess(self):
        package = {"voca": [], "labelset": [], "vocafreq": {}, "weights": {}, "doclist": {}, "chi_value": {}}
        input = "../Corpus/Reuters_train_1.txt"

        corpus = preprocess(input, package)
        print("\npreprocess运行成功")

        print(corpus)

        test = 0

        tf_vc(corpus, test, package)


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
