import sys 
sys.path.append('..')
import numpy as np
from commonLearn.util import preprocess,create_co_matrix,cos_similarity,ppmi

text='You say goodbye and I say hello.'
corpus,word_to_id,id_to_word=preprocess(text)
vocab_size=len(word_to_id)
C=create_co_matrix(corpus,vocab_size,window_size=4)
W=ppmi(C)

# 有効数字3桁で表示させる
np.set_printoptions(precision=3) 
print('covariance matrix')
print(C)
print('-'*50)
print('PPMI')
print(W)