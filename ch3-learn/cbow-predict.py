import sys
sys.path.append('..')
from common.layers import MatMul
import numpy as np
c0=np.array([1,0,0,0,0,0,0])
c1=np.array([0,0,1,0,0,0,0])

# 入力層→隠れ層の重み行列
W_in=np.random.randn(7,3) 
# 隠れ層→入力層の重み行列
W_out=np.random.randn(3,7)

# MatMulレイヤーの作成
in_layer0=MatMul(W_in)
in_layer1=MatMul(W_in)
out_layer=MatMul(W_out)

# c0を隠れ層ベクトルに変換
h0=in_layer0.forward(c0)
# c1を隠れ層ベクトルに変換
h1=in_layer1.forward(c1)
h=0.5*(h1+h0)
s=out_layer.forward(h)

print(s)
