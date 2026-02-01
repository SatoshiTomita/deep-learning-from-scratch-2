import numpy as np
class Embedding:
    def __init__(self,W):
        self.params=[W]
        # Wと同じ形のゼロで初期化した勾配を入れる箱を用意
        self.grads=[np.zeros_like(W)]
        self.idx=None

    def forward(self,idx):
        W,=self.params
        self.idx=idx
        # idxの行のベクトルを取り出す
        out=W[idx]
        return out
    
    def backward(self,dout):
        # 重みの勾配をdWとして取り出す
        dW=self.grads
        # 勾配をゼロで初期化
        dW[...]=0
        # dWのidxの行にdoutを加える
        # dW[self.idx]=dout

        # enumerateでインデックスとword_idを取り出す
        # iはインデックス、word_idはword_id
        # 重複問題に対応するために代入ではなく加算を行う
        for i,word_id in enumerate(self.idx):
            dW[word_id]+=dout[i]
        return None