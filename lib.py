import numpy as np 

images_= {
    # 5x5の1~5の数字の画像、バイナリデータ
    0: np.array([
        [0,0,1,0,0],
        [0,1,0,1,0],
        [1,0,0,0,1],
        [0,1,0,1,0],
        [0,0,1,0,0],
    ]),
    1: np.array([
        [0,0,1,0,0],
        [0,1,1,0,0],
        [0,0,1,0,0],
        [0,0,1,0,0],
        [0,1,1,1,0],
        ]),
    2: np.array([
        [0,1,1,1,0],
        [0,0,0,1,0],
        [0,1,1,1,0],
        [0,1,0,0,0],
        [1,1,1,1,1],
        ]),
    3: np.array([
        [0,1,1,1,0],
        [0,0,0,1,0],
        [0,1,1,1,0],
        [0,0,0,1,0],
        [0,1,1,1,0],
        ]),
    4: np.array([
        [0,1,0,1,0],
        [0,1,0,1,0],
        [0,1,1,1,0],
        [0,0,0,1,0],
        [0,0,0,1,0],
        ]),
    5: np.array([
        [1,1,1,1,1],
        [1,0,0,0,0],
        [1,1,1,1,1],
        [0,0,0,0,1],
        [1,1,1,1,1],
        ]),
    6: np.array([
        [0,1,1,1,0],
        [1,0,0,0,0],
        [1,1,1,1,1],
        [1,0,0,0,1],
        [0,1,1,1,0],
        ]),
    7: np.array([
        [1,1,1,1,1],
        [0,0,0,1,0],
        [0,0,1,0,0],
        [0,1,0,0,0],
        [1,0,0,0,0],
        ]),
}

IMAGES= {}
for i in range(len(images_.keys())):
    key = list(images_.keys())[i]
    IMAGES[i] = np.reshape(images_[key]*2-1, (25,))

# add noise to images
def add_noise(img: np.ndarray, rate=0.1):
    img2 = np.copy(img)
    img2 = np.reshape(img2, (5,5))
    # deepcopy 
    # 画像のサイズを取得
    h, w = img2.shape
    assert h == w == 5, '画像のサイズは5x5である必要があります'
    # ノイズを追加
    for _ in range(int(h*w*rate)):
        y = np.random.randint(0, h)
        x = np.random.randint(0, w)
        img2[y][x] *= -1
    return img2.reshape(25,)

def print_image(img: np.ndarray):
    img = np.reshape(img, (5,5))
    for y in range(5):
        for x in range(5):
            print("#" if img[y][x] > 0 else " ", end='')
        print()
    print()

