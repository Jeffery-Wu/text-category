# -*- coding: utf-8 -*-
# 主成分分析 降维
import pandas as pd
import numpy as np

# 参数初始化
inputfile = 'C:\\Users\\LENOVO\\Desktop\\武汉市住房保障2.xls'
outputfile = '../dimention_reducted.xls'  # 降维后的数据

X = pd.read_excel(inputfile, header=None)  # 读入数据
print(X.shape)
X_mean = X.mean()

num = 4

E = np.mat(np.zeros((num, num)))
for i in range(len(X)):
    E += (X.iloc[i, :].values.reshape(num, 1) - X_mean.values.reshape(num, 1)) * (
            X.iloc[i, :].values.reshape(1, num) - X_mean.values.reshape(1, num))

R = np.mat(np.zeros((num, num)))
for i in range(num):
    for j in range(num):
        R[i, j] = E[i, j] / np.sqrt(E[i, i] * E[j, j])

import numpy.linalg as nlg

print(R)

# eig_value, eig_vector = nlg.eig(R)
#
# eig = pd.DataFrame()
# eig['names'] = X.columns
# eig['eig_value'] = eig_value
# print(eig)
#
# eig.sort_values('eig_value', ascending=False, inplace=True)
# print(eig)
# print(eig_vector)
#
# for m in range(1, 10):
#     if eig['eig_value'][:m].sum() / eig['eig_value'].sum() >= 0.8:
#         print(m)
#         break
#
# A = np.mat(np.zeros((num, 9)))
# A[:, 0] = np.sqrt(eig_value[0]) * eig_vector[:, 0]
# A[:, 1] = np.sqrt(eig_value[1]) * eig_vector[:, 1]
# A[:, 2] = np.sqrt(eig_value[3]) * eig_vector[:, 3]
# A[:, 3] = np.sqrt(eig_value[2]) * eig_vector[:, 2]
#
# print('A:')
# print(pd.DataFrame(A))  # 因子载荷阵
#
# h = np.zeros(num)  # 共同度
# D = np.mat(np.eye(num))  # 特殊因子方差
# for i in range(num):
#     a = A[i, :] * A[i, :].T
#     h[i] = a[0, 0]
#     D[i, i] = 1 - a[0, 0]
#
# print(h)
# print(pd.DataFrame(D))
#
# for i in range(num):
#     a = A[i, :] / np.sqrt(D[i, i])

# a = pd.DataFrame(A)
# a.columns = ['factor1', 'factor2', 'factor3', 'factor4']
# a.index = ['x1', 'x2', 'x3', 'x4']
# # a.index = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9']
# print(a)

# from sklearn.decomposition import PCA
#
# pca = PCA()   #保留所有成分
#
# pca.fit(X)
# # pca.components_ #返回模型的各个特征向量
# # pca.explained_variance_ratio_ #返回各个成分各自的方差百分比(也称贡献率）
# print(pca.components_)
# print(pca.explained_variance_ratio_)


# pca.fit(data)

# low_d = pca.transform(data)   #降低维度
# pd.DataFrame(low_d).to_excel(outputfile)  #保存结果
