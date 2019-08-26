# coding=utf-8

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import jieba
from sklearn.preprocessing import MinMaxScaler

def dict_vectorizer():
	"""
	字典特征化
	:return:
	"""
	data = [
		{"city": '北京', 'tempera': 30},
		{"city": '上海', 'tempera': 60},
		{"city": '深圳', 'tempera': 90},
	]

	dv = DictVectorizer(sparse=False)
	response = dv.fit_transform(data)
	response_inverse = dv.inverse_transform(data)
	print(response.get_feature_names())
	print('*' * 15)
	print(response_inverse)
	print('*' * 15)
	print(response)


def count_vectorizer():
	"""
	文本特征值化
	:return:
	"""
	# 对文本进行特征值化
	# 文本数据， 数组形式
	# data = ['hello i`m python user', 'i know this is cool, do you know that?']
	data = ['十三陵是明朝的皇帝墓', '我明天要去天津']

	cv = CountVectorizer()

	# 调用fit_transform 输入数据并转换
	response = cv.fit_transform(data)

	# 文本特征值化里面没有inverse_transform 方法 但是通过 toarray 方法转为数组形式

	print(response)    # 转为sparse矩阵形式
	print('*' * 15)
	print(cv.get_feature_names())  # 统计文章中没有重复出现的词
	print('*' * 15)
	print(response.toarray())  # 转为数组形式


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



if __name__ == '__main__':
	# dict_vectorizer()

	# count_vectorizer()

	# hanzi_vectorizer()

	# tfidf_vectorizer()

	min_max()
