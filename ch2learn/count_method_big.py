import sys
sys.path.append('..')
import numpy as np
from commonLearn.util import most_similar,create_co_matrix,ppmi
from dataset import ptb

window_size=2
wordvec_size=100

corpus,word_to_id,id_to_word=ptb.load_data('train')
vocab_size=len(word_to_id)
print('counting co-occurrence...')
C=create_co_matrix(corpus,vocab_size,window_size)
print('calculating PPMI...')
W=ppmi(C,verbose=True)

print('calculating SVD')

try:
    # 乱数を使ったSVDで徳一の大きいものだけに限定して計算することで高速に計算できる
    from sklearn.utils.extmath import randmized_svd
    U,S,V=randmized_svd(W,n_components=wordvec_size,n_iter=5,random_state=None)

except ImportError:
    U,S,V=np.linalg.svd(W)
    word_vecs=U[:,:wordvec_size]
    querys=['you','year','car','toyota']
    for query in querys:
        most_similar(query,word_to_id,id_to_word,word_vecs,top=5)