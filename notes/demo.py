# coding=utf-8


from sklearn.feature_extraction import DictVectorizer


data = [
	{"city": '北京', 'tempera': 30},
	{"city": '上海', 'tempera': 60},
	{"city": '深圳', 'tempera': 90},
]


content = DictVectorizer()
