from commonLearn.layers import Embedding
import numpy as np
class EmbeddingDot:
    def __init__(self,W):
        # embedにembedding layerを入れる
        self.embed=Embedding(W)
        self.params=self.embed.params
        self.grads=self.embed.grads
        # 順伝播の際に計算した結果を一次的に保存する変数
        self.cache=None
    
    #中間層のニューロンと、ミニバッチ学習のため、単語Idのnumpy配列を受け取る 
    def forward(self,h,idx):

        target_W=self.embed.forward(idx)
        # 内積を計算する
        out=np.sum(target_W*h,axis=1)
        self.cache=(h,target_W)
        return out
    
    def backward(self,dout):
        h,target_W=self.cache
        dout=dout.reshape(dout.shape[0],1)
        dtarget_W=dout*h
        self.embed.backward(dtarget_W)
        dh=dout*target_W
        return dh
