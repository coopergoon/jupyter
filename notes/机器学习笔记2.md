归一化公式计算 

视频缺失  以后补上 



sklearn归一化api

> from sklearn.preprocessing import MinMaxScaler
>
> Scaler是缩放的意思
>
>  MinMaxScaler(feature_range=(0,1).....) 
>
> feature_range是每个特征值缩放的范围， 可以指定，默认为0到1

>  MinMaxScaler.fit_transform(X) 方法 
>
> X 是numpy array格式的数据   [n_samples, n_features]



归一化的步骤：

1. 实例化MinMaxScaler

2. 通过fit_transform 转换

   

问: 为什么要使用归一化？ 

使得某一个特征不会对最终结果造成更大的影响。



问：什么时候使用归一化？

当多个特征同等重要的时候需要进行归一化。



问： 如果数据中异常点较多， 会有什么影响？

如果异常点是最大致或者最小值是 影响很大。



归一化总结：

> 注意在特点场景下最大值最小是值是变化的，最大值和最小值最容易受异常点的影响，所以这个方法的鲁棒性（即稳定性）很差，只适合传统的精确小数据场景

代码：

```python

from sklearn.preprocessing import MinMaxScaler

def min_max():
   """
   归一化处理
   :return:
   """
   # 二维数组
   data = [
      [100, 203, 45656],
      [233, 203, 12344],
      [100, 234, 5554],
   ]

   mm = MinMaxScaler(feature_range=(3,4))

   response = mm.fit_transform(data)
   print(response)
  
  
  输出：
    [[3.         3.         4.        ]
     [4.         3.         3.16931824]
     [3.         4.         3.        ]]
```



#### 标准化 

问: 为什么要使用标准化？ 

使得某一个特征不会对最终结果造成更大的影响。



问： 方差反应了什么？

方差反应了数控的稳定性。 



<img src="https://tva1.sinaimg.cn/large/006y8mN6ly1g6b91jee7aj30to0iowli.jpg">



