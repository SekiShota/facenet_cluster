from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# 160x160に切り抜き
mtcnn=MTCNN(image_size=160, margin=10)
# 512次元配列に変換
resnet=InceptionResnetV1(pretrained="vggface2").eval()

# 画像から顔部分抽出と512次元配列に変換する関数
def img2vec(file_path,idx,out_path):
    img=Image.open(file_path)
    # 顔部分の抽出
    img_cropped=mtcnn(img, out_path+str(idx)+".jpg")
    # img_cropped=mtcnn(img)

    img_enbedding=resnet(img_cropped.unsqueeze(0))
    # 1x512のリストに変換
    v=img_enbedding.tolist()

    # v=img_enbedding.squeeze().to('cpu').detach().numpy().copy()
    
    return v








