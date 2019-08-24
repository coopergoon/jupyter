# coding=utf-8

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer


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


if __name__ == '__main__':
	# dict_vectorizer()

	count_vectorizer()
