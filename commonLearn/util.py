from curses import window
from importlib.resources import contents
from multiprocessing import context
import numpy as np
def preprocess(text):
    text=text.lower()
    # コンマの前にスペースが必要
    text=text.replace('.',' .')
    # スペースで単語を区切って格納する
    words=text.split(' ')

    word_to_id={}
    id_to_word={}

    for word in words:
        if word not in word_to_id:
            new_id=len(word_to_id)
            word_to_id[word]=new_id
            id_to_word[new_id]=word

    corpus=np.array([word_to_id[w] for w in words])
    return corpus,word_to_id,id_to_word

# corpusを単語IDのリスト(corpus=[0,1,2,3]など)、vocab_sizeを語彙数、window_sizeをウィンドウサイズ
def create_co_matrix(corpus,vocab_size,window_size):
    corpus_size=len(corpus)
    # 単語×単語の表
    # ゼロを入れて初期化
    co_matrix=np.zeros((vocab_size,vocab_size),dtype=np.int32)

    # enumerateでidとword_idを取り出している
    for idx,word_id in enumerate(corpus):
        # 左にi個、右にi個
        for i in range(1,window_size+1):
            # 左側の単語
            left_idx=idx-i
            # 右側の単語
            right_idx=idx+i
            # 範囲内ならば、word_id行、left_id_word列を+1する
            if left_idx>=0:
                left_word_id=corpus[left_idx]
                co_matrix[word_id,left_word_id]+=1
            
            #右側も同様に処理
            if right_idx<corpus_size:
                right_word_id=corpus[right_idx]
                co_matrix[word_id,right_word_id]+=1

    return co_matrix

def cos_similarity(x,y,eps=1e-8):
    nx=x/(np.sqrt(np.sum(x**2))+eps)
    ny=y/(np.sqrt(np.sum(y**2))+eps)
    return np.dot(nx,ny)

# ある単語がクエリとして与えられたとき、そのクエリに対して類似した単語を上位から順に表示する関数
def most_similar(query,word_to_id,id_to_word,word_matrix,top=5):
    if query not in word_to_id:
        print('%s is not found'%query)
        return 
    
    print('\n[query]'+query)
    # クエリの単語ベクトルを取り出す
    query_id=word_to_id[query]
    query_vec=word_matrix[query_id]
    vocab_size=len(id_to_word)
    # vocab_size個の要素をもつ一次元配列
    similarity=np.zeros(vocab_size)
    # クエリの単語ベクトルと、他の全ての単語ベクトルについてコサイン類似度を求める
    for i in range(vocab_size):
        similarity[i]=cos_similarity(word_matrix[i],query_vec)
    
    count=0
    # argsortは昇順でソートしたインデックスを返す
    # コサイン類似度の高い順に返したいからマイナスする
    for i in (-1*similarity).argsort():
        # クエリ事態はスキップする
        if id_to_word[i]==query:
            continue
        print('%s:%s'%(id_to_word[i],similarity[i]))
        # top件まで出力する
        count+=1
        if count>=top:
            return


# 共起行列をPPMI行列に変換する関数
def ppmi(C,verbose=True,eps=1e-8):
    # ppmi行列をCと同じ形でゼロで初期化
    M=np.zeros_like(C,dtype=np.float32)
    # 共起行列の全要素の合計
    N=np.sum(C)
    # 各列の合計
    # 単語の出現する回数
    S=np.sum(C,axis=0)
    total=C.shape[0]*C.shape[1]
    cnt=0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi=np.log2(C[i,j]*N/(S[j]*S[i])+eps)
            # i行j列についてpmi行列を作成
            M[i,j]=max(0,pmi)
            # verboseがtrueだったら進捗を表示させる
            if verbose:
                cnt+=1
                if cnt%(total//100+1)==0:
                 print('%lf%%done'%(100*cnt/total))
    return M


# corpusからコンテキストとターゲットを作成する関数
def create_contexts_target(corpus,window_size=1):
    # コーパスの両端をのぞいた部分をターゲットとして抽出
    target=corpus[window_size:-window_size]
    contexts=[]

    # ターゲット位置を順に処理
    # idxは現在のターゲット位置
    for idx in range(window_size,len(corpus)-window_size):
        cs=[]
        # ターゲット前後のwindow_sizeこの単語をコンテキストとして収集
        for t in range(-window_size,window_size+1):
            if t==0:
                continue
            cs.append(corpus[idx+t])
        contexts.append(cs)
    return np.array(contexts),np.array(target)






