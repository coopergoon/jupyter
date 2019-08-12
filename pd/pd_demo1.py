# coding=utf-8

import pandas as pd
import numpy as np


a1 = [1, 2, 3, 4]
a2 = ['a', 'b', 'c', 'd']
a3 = pd.Series(data=a1, index=a2, name='chen')
# index 是索引列
# data 是数据列   date和index 长度必须相同
print(a3)

print(a3.index)  # 取出索引列
print(a3.name)  # Series对象的名字
print(a3.values) # 取出数据列
print(a3.dtype)  # 元素类型


# index 要么不提供，要么提供的时候必须和data长度一致
a5 = np.array([1, 2, 4, 4]) # 创建一个简单的相当于py中的list
a4 = pd.Series(data=a5)

print(a4)



# 如果传的数据是已经自带索引了 就不需要再传递索引 比如字典，或者已经建立的Series
a6 = {"a": 1, "b": 2, "c": 3, 'd': 44}
a7 = pd.Series(data=[1,2,3,5,5], index=['z1', 'z2', 'z3', 'z4', 'z1'])
f4 = [11, 22, 33, 44, 55]
# q1 = pd.Series(data=a7, index=f4)
q2 = pd.Series(data=a6, index=f4)
print(q2)

