归一化公式计算 

视频缺失  以后补上 

<img src="https://tva1.sinaimg.cn/large/006y8mN6ly1g6da005getj30ui0gsjxa.jpg">



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

标准化是最常使用的

问: 为什么要使用标准化？ 

使得某一个特征不会对最终结果造成更大的影响。



问： 方差反应了什么？

方差反应了数控的稳定性。 



<img src="https://tva1.sinaimg.cn/large/006y8mN6ly1g6dalfelitj30ty0jwjxg.jpg" width=500 height=400>



作用于每一列的意思： 

<img src="https://tva1.sinaimg.cn/large/006y8mN6ly1g6da8e3ic8j30co074myk.jpg">

对特征1这一列进行计算，也要对特征2这一列进行计算. 是按列来的。 

异常点对平均值的影响不大， 因为即使加进来一个异常，样本数才是增加1， 不会有很大的波动。 



为什么方差能反应数据的稳定性？ 

根据公式， mean是平均值， 当数据集中的时候， 每个特征值和平均值的差都是比较靠近的。这样的话，点也是集中的。方差就会变小；

但是特征值比较分散的时候，点也是分散的， 方差就会变大。 

极端情况是，当方差为0， 则该列的特征值都是一样的数字。

如图:

<img src="https://tva1.sinaimg.cn/large/006y8mN6ly1g6daesshm9j30oq0eq0vy.jpg"  width=300 height=300>







总结：

对于归一化来说，如果出现异常点，影响了最大值最小值，那么结果显然会发生改变；

对于标准化来说，如果出现异常点，由于其有一定的数据量少量的异常点对于平均值的影响并不大，从而方差改变小。

所以大部分使用的是标准化。 

标准化的API：

> ```python
> from sklearn.preprocessing import StandardScaler
> ```



处理之后每列所有的数据都聚集在均值为0附近标准差为1 





标准化步骤：

1. 实例化StandardScaler
2. 调用fit_transform 



```python
def stand():
   """
   标准化处理
   :return:
   """
   # 二维数组
   data = [
      [100, 203, 45656],
      [233, 203, 12344],
      [100, 234, 5554],
   ]

   std = StandardScaler()

   response = std.fit_transform(data)
   print(response)
  
  
  标准化的结果：
[[-0.70710678 -0.70710678  1.39640922]
 [ 1.41421356 -0.70710678 -0.50447551]
 [-0.70710678  1.41421356 -0.89193371]]
```



如果要对数据进行缩放， 首要的想到的要是标准化； 

在样本足够多的情况下比较稳定，适合现代的嘈杂的大数据场景。



数值型数据，在进行标准缩放时候：

1. 归一化
2. 标准化
3. 缺失值处理（pandas处理）

类别型数据： One-hot编码

时间类型数据： 时间切分





##### 缺失值处理方法

| 删除 | 如果数据缺失到一定比例， 直接删除整行或者整列                |
| ---- | ------------------------------------------------------------ |
| 插补 | 可以通过缺失值的每行或者每列的平均值，中位数填充， 一般都会选择按列，因为同一列才是同一个特征的数据。 |

sklearn的API：

> from sklearn.preprocessing import Imputer

```python
def __init__(self, missing_values="NaN", strategy="mean",
             axis=0, verbose=0, copy=True):
    self.missing_values = missing_values
    self.strategy = strategy
    self.axis = axis
    self.verbose = verbose
    self.copy = copy
    
    
    missing_values 
    strategy 平均值还是中位数等  填补的策略
    axis  0是列， 1是行
    
```



```python
def imputer():
   """
   处理缺失值
   :return:
   """

   data = [
      [100, 203, 45656],
      [np.nan, 203, 12344],
      [100, 234, 5554],
   ]
   im = Imputer(missing_values='NaN', axis=0, strategy='mean')
   response = im.fit_transform(data)
   print(response)
    
    
    
   输出：
[[  100.   203. 45656.]
 [  100.   203. 12344.]
 [  100.   234.  5554.]]


在写的np.nan的位置被替换为了100,   因为使用这个api，在计算的时候是将该的第一个和第三个详想加在除以2，不是除以3的。
```



注意在转换之前需要将原始数据中的缺失值写为np.nan或者np.NaN,(这是float类型)， 否则无法处理。 

