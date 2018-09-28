# -*- coding:utf8 -*-
import sys

# reload(sys)
# sys.setdefaultencoding('utf8')
sys.path.append('..')
import os
from numpy import *

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# 转变为稀疏矩阵
def one_hot(data):
    values = array(data)
    # print("\nvalues:")
    # print(values)

    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # print("\ninteger_encoded:")
    # print(integer_encoded)

    # binary encode

    onehot_encoder = OneHotEncoder(sparse=True)
    integer_encoded = integer_encoded.reshape(integer_encoded.shape[0], 1)

    # 非稀疏
    # onehot_encoder = OneHotEncoder(sparse=False)
    # integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # print("\nonehot_encoded:")
    # print(onehot_encoded)

    return onehot_encoded
