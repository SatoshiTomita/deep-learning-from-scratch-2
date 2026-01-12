import sys
sys.path.append('..')
from commonLearn.util import preprocess,create_co_matrix,most_similar

text='You say goodbye and I say hello.'
corpus,word_to_id,id_to_word=preprocess(text)
vocab_size=len(word_to_id)
C=create_co_matrix(corpus,vocab_size,window_size=5)

# youという単語をクエリとして類似する単語を表示
most_similar('you',word_to_id,id_to_word,C,top=5)
