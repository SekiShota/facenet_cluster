# 主成分分析
# 512次元から2次元に変換する

# from sklearn.decomposition import PCA
from cropping import img2vec
import glob
import numpy as np

# 入力画像と出力先の指定
filepath=sorted(glob.glob("./Images/*.jpg"))
op="./face/"

feature=[]
for i,f in enumerate(filepath):
    v=img2vec(file_path=f, idx=i+1, out_path=op)
    feature.append(v)


# リスト型からnumpy型に変換,サイズは"画像枚数x512"
# 特徴量は512
features=np.array(feature).reshape(30,512)
print("type:", type(features))
print("result:", features.shape)


from sklearn.decomposition import PCA
# 主成分分析
def PCA_2d():
    pca=PCA(n_components=2)
    pca.fit(features)
    reduced=pca.fit_transform(features)

    return reduced

print("pca:",PCA_2d().shape)
