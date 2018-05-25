#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# doc1：路 损坏 火灾 金 中国 地 货物
# doc2：路 到达 运输 中国 地 银 卡车
# doc3：路 到达 金 中国 地 货物 卡车
a = np.array([[1, 1, 1], # 路
              [0, 1, 1], # 到达
              [1, 0, 0], # 损坏
              [0, 1, 0], # 运输
              [1, 0, 0], # 火灾
              [1, 0, 1], # 金
              [1, 1, 1], # 中国
              [1, 1, 1], # 地
              [1, 0, 1], # 货物
              [0, 2, 0], # 银
              [0, 1, 1]])# 卡车

# doc_test：金 银 卡车
q = np.array([[0], # 路
              [0], # 到达
              [0], # 损坏
              [0], # 运输
              [0], # 火灾
              [1], # 金
              [0], # 中国
              [0], # 地
              [0], # 货物
              [1], # 银
              [1]])# 卡车

# SVD
u, s, vh = np.linalg.svd(a, full_matrices=False)
s = s * np.eye(len(s))
print('u\n', u)
print('s\n', s)
print('vh\n', vh)
print('q\n', q)

# reduce feature size
feature_size = 2
u_k = u[:, :feature_size]
s_k = s[:feature_size, :feature_size]
vh_k = vh[:feature_size, :]
print('u_k\n', u_k)
print('s_k\n', s_k)
print('vh_k\n', vh_k)
# compute q vec
s_k_inv = np.linalg.inv(s_k)
q_t = q.T
q_test = np.dot(np.dot(q_t, u_k), s_k_inv)
print('s_k_inv\n', s_k_inv)
print('q_t\n', q_t)
print('q_test\n', q_test)

# Visualizing
ax = plt.axes()
for i in range(len(vh_k[0])):
    x = vh_k[0][i]
    y = vh_k[1][i]
    ax.text(x, y, 'doc' + str(i + 1) + '(' + "{:.2f}".format(x)  + ',' + "{:.2f}".format(y) + ')')
    ax.arrow(0, 0, x, y,
             head_width=0.05, head_length=0.05, fc='g', ec='g')
x = q_test[0][0]
y = q_test[0][1]
ax.text(x, y, 'doc_test' + '(' + "{:.2f}".format(x)  + ',' + "{:.2f}".format(y) + ')')
ax.arrow(0, 0, x, y,
         head_width=0.05, head_length=0.05, fc='r', ec='r')
plt.axis([-1, 1, -1, 1])
plt.show()
