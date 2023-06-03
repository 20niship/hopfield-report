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
}

IMAGES= {}
for i in range(5):
    IMAGES[i] = np.reshape(images_[i]*2-1, (25,))

# add noise to images
def add_noise(img: np.ndarray, rate=0.1):
    img = np.reshape(img, (5,5))
    # 画像のサイズを取得
    h, w = img.shape
    assert h == w == 5, '画像のサイズは5x5である必要があります'
    # ノイズを追加
    for _ in range(int(h*w*rate)):
        y = np.random.randint(0, h)
        x = np.random.randint(0, w)
        img[y][x] = 1 if img[y][x] == 0 else 0
    return img

def print_image(img: np.ndarray):
    img = np.reshape(img, (5,5))
    for y in range(5):
        for x in range(5):
            print("#" if img[y][x] > 0 else " ", end='')
        print()
    print()

