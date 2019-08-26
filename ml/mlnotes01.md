#### 机器学习  

1. 机器学习数据格式 csv   
2. 不使用mysql等， 因为性能问题，数据格式也不
3. python是动态语言， 性能不好
4. numpy释放了GIL解释器锁 ， 运行的时候线程是真的多线程，而不是轮训的的线程   （cpython特有的）
5. jpython是没有GIL锁

#### 常用数据集的结构组成

**结构： 特征值 +  目标值**      

pandns的数据结构即使dataframe

|      | 身高 | 体重 | 头发长度 | 肤色 | 性别 |
| ---- | ---- | ---- | -------- | ---- | ---- |
| 1    | 160  | 60   | 长       | 黄色 | 女   |
| 2    | 170  | 60   | 短       | 黄色 | 男   |
| 3    | 180  | 60   | 短       | 黄色 | 男   |

其中 身高， 体重， 头长度，肤色， 这些都是特征 ， 即列索引 

而 最终想要什么，就是目标值， 比如， 要通过特征值判断性别， 性别就是目标值 

行索引123则是样本1，样本2， 样本3.。。。。。   

有些数据集是可以没有目标值的（这和算法有关）



#### 数据中对特征的处理 

pandas读取方便， 可以对缺失值处理， 数据转换 ， 合并 等。 比如说在肤色里面， 如果是黄色，白色， 这些字符串是无法处理的， 需要转换为数字。 

机器学习中重复值需不需要进行处理： 看情况， 有些需要处理， 有些需要， 即使重复了， 也不会对学习产生影响。

sklearn： 对特征处理提供了强大的接口



以上对特征的处理， 成为特征工程。 这个概念比较广， 就是对一个个的特征处理。 

算法的选择都是差不多的， 数据也都是差不多的时候， 特征的处理就是最关键的， 差别也就体现了。所以要注意特征的处理。

拿到数据--> 数据预处理--> 特征工程-->  机器学习-->  模型评估 --> 数据预处理--> 。。。。。。这是一个循环的过程 





#### 特征工程   

1. 数据的特征抽取
2. 数据的特征预处理
3. 数据的降维
4. 特征工程的意义

特征工程的意义 ： 特征工程是将原始数据转换为更好的代表预测模型的潜在问题的特征的过程，从而提高对未知数据的预测的准确性。 

比如一文章， 输入进去后， 会被转化为代表这篇文章的特征，也就是数组， 然后再针对不一样的文章进行处理或者分类。  

**特征工程会直接影响预测结果。** 





#### 特征抽取  

将一个文章转换为一数组就是特征抽取。 

针对的是数字， 文本。特征抽取是对文本等数据进行**特征值化**. 就是转变为数字，是为了让机器更好的理解数据。 



sklearn特征抽取的API： 

> sklearn.feature_extraction
>
> feature是特征， extraction是抽取的意思



##### 字典特征抽取 

对字典数据进行特征值化 

> sklearn.feature_extraction.DictVectorizer
>
> Vectorizer是数字化意思



```python
# coding=utf-8

from sklearn.feature_extraction import DictVectorizer

data = [
   {"city": '北京', 'tempera': 30},
   {"city": '上海', 'tempera': 60},
   {"city": '深圳', 'tempera': 90},
]

obj = DictVectorizer()
d = obj.fit_transform(data)
print(obj.get_feature_names()) # 输出的是 ['city=上海', 'city=北京', 'city=深圳', 'tempera']
print(d)

输出结果 sparse矩阵： 
  (0, 1)	1.0
  (0, 3)	30.0
  (1, 0)	1.0
  (1, 3)	60.0
  (2, 2)	1.0
  (2, 3)	90.0
前面是坐标，后面内容

在实例化话的时候可以指定参数sparse=False
obj = DictVectorizer(sparse=False)
print(obj.get_feature_names()) # 输出的是 ['city=上海', 'city=北京', 'city=深圳', 'tempera']
输出结果是 ndarray类型 二维数组：
 ['city=上海', 'city=北京', 'city=深圳', 'tempera']
[[ 0.  1.  0. 30.]
 [ 1.  0.  0. 60.]
 [ 0.  0.  1. 90.]
 ]

可以看出， sparse矩阵是将这个二维矩阵转化了， 如果二维矩阵中不为0的，会用左边表示出来；其余都是为0。
对此原始数据data， 第一个是上海的数据， 所以，二维数组中第一个被置为0， 


```

sparse矩阵：可以节约内存方便读取处理 ，但是实际用的不是很多。 

数组形式， 有类别的的这些特征，先转为字典数据

字典数据抽取： 把字典中一些类别数据分别进行转换成特征。如果是字符的， 会转为数值型，如果是数字的， 还是数字

**One-hot编码**

> [[ 0.  1.  0. 30.]
>  [ 1.  0.  0. 60.]
>  [ 0.  0.  1. 90.]
>  ] 
>
> 这成为one-hot编码





#### 文本特征抽取 

作用： 对文本进行特征值化

```python

from sklearn.feature_extraction.text import CountVectorizer

def count_vectorizer():
	"""
	文本特征值化
	:return:
	"""
	# 对文本进行特征值化
	# 文本数据， 数组形式
	data = ['hello i`m python user', 'i know this is cool, do you know that?']

	cv = CountVectorizer()

	# 调用fit_transform 输入数据并转换
	response = cv.fit_transform(data)

	# 文本特征值化里面没有inverse_transform 方法 但是通过 toarray 方法转为数组形式

	print(response)    # 转为sparse矩阵形式
	print('*' * 15)
	print(cv.get_feature_names())  # 统计文章中没有重复出现的词
	print('*' * 15)
	print(response.toarray())  # 转为数组形式

  
  输出结果：
   (0, 2)	1
  (0, 5)	1
  (0, 8)	1
  (1, 4)	2
  (1, 7)	1
  (1, 3)	1
  (1, 0)	1
  (1, 1)	1
  (1, 9)	1
  (1, 6)	1
***************
['cool', 'do', 'hello', 'is', 'know', 'python', 'that', 'this', 'user', 'you']
***************
[[0 0 1 0 0 1 0 0 1 0]
 [1 1 0 1 2 0 1 1 0 1]]

此处也是one-hot编码
**注意： 单个字母不统计**，因为单个字母没有分类的依据



如果输入的data数据是中文的，返回的结果是： 
	data = ['十三陵是明朝的皇帝墓', '我明天要去天津']

  结果：
  (0, 0)	1
  (1, 1)	1
***************
['十三陵是明朝的皇帝墓', '我明天要去天津']
***************
[[1 0]
 [0 1]]
```





```python

from sklearn.feature_extraction.text import CountVectorizer
import jieba

def cut_words():
   """
   jieba分词
   :return:
   """
   c1 = jieba.cut('十三陵是明朝的皇帝墓')
   c2 = jieba.cut('我明天要去天津')

   print(c1, c2)  # 返回是的生成器

   # 转为list

   con1 = list(c1)
   con2 = list(c2)

   print(con1, con2)


   # 组合成字符串形式返回

   return ' '.join(con1), ' '.join(con2)


def hanzi_vectorizer():
   """
   汉语句子文本特征值化
   :return:
   """
   # 切分汉语句子， 得出分词结果
   c1, c2 = cut_words()

   cv = CountVectorizer()

   # 调用 fit_transform 输入数据并转换   列表形式
   response = cv.fit_transform([c1, c2])

   # 文本特征值化里面没有inverse_transform 方法 但是通过 toarray 方法转为数组形式

   print(response)    # 转为sparse矩阵形式
   print('*' * 15)
   print(cv.get_feature_names())  # 统计文章中没有重复出现的词
   print('*' * 15)
   print(response.toarray())  # 转为数组形式
```



文本特抽取方式：

1. 使用CountVectorizer 方法 统计次数  【这样的方式用的很少，因为不实用】
2. 使用tfidf 

文本特征应用场景:

> 1. 文本分类
> 2. 情感分析 

汉字是不被支持进行文本特征抽取的。 只有先将其进行分词之后，才能让特征值化



##### 以下 是tfidf 的内容

<hr>

1. tf:  term frequencty 词的频率       即出现的次数
2. idf ： 逆文档频率   inverse document frequency   log(zong文章数量/该词出现的文档数量)
3. log(x) 函数： x的值越小， log(x) 的值越小  
4. 最后将 tf 和idf 相乘 ， 得出的结果为重要性程度 

代码示例：

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import jieba


def tfidf_vectorizer():
	"""
	汉语句子文本特征值化
	使用tf-idf方法
	:return:
	"""
	# 切分汉语句子， 得出分词结果
	c1, c2 = cut_words()

	tf = TfidfVectorizer()

	# 调用 fit_transform 输入数据并转换   列表形式
	response = tf.fit_transform([c1, c2])

	print(response)    # 转为sparse矩阵形式
	print('*' * 15)
	print(tf.get_feature_names())  # 统计文章中没有重复出现的词
	print('*' * 15)
	print(response.toarray())  # 转为数组形式
	print('*' * 15)
	print(tf.inverse_transform([c1, c2]))

  
  结果：
  
  (0, 4)	0.5773502691896257
  (0, 3)	0.5773502691896257
  (0, 0)	0.5773502691896257
  (1, 1)	0.7071067811865476
  (1, 2)	0.7071067811865476
***************
['十三陵', '天津', '明天', '明朝', '皇帝']
***************
[[0.57735027 0.         0.         0.57735027 0.57735027]
 [0.         0.70710678 0.70710678 0.         0.        ]]

# 数值越大， 说明其重要性程度越高  
***************   
[array(['十三陵', '天津'], dtype='<U3')]

Process finished with exit code 0

    
```



为什么会用到TfidfVectorizer？？

答： 因为分类机器学习算法的重要依据 



