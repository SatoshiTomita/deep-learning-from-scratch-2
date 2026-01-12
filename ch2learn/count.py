#pythonの標準ライブラリ
import sys 
# pythonがモジュールを探しに行くディレクトリのリスト
sys.path.append('..')
import numpy as np
from commonLearn.util import preprocess

text='you say goodbye and I say hello.'
corpus,word_to_id,id_to_word=preprocess(text)

print(corpus)

print(id_to_word)
