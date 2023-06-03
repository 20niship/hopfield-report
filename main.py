from lib import IMAGES, add_noise, print_image
import numpy as np
from matplotlib.animation import ArtistAnimation
import matplotlib.pyplot as plt
from typing import List

def animate_image(y: List[np.ndarray]):
    # animate image list y
    fig = plt.figure()
    ims = []
    plt.cla()
    IMG_SIZE=30
    for i in range(IMG_SIZE):
        idx = int(i*len(y)/IMG_SIZE)
        t = plt.text(0, 0, "iter: {}".format(str(idx)))
        t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='white'))
        im = plt.imshow(y[idx], animated=True)
        ims.append([im])
    ani = ArtistAnimation(fig, ims, interval=1500/IMG_SIZE)
    ani.save("result.gif", writer="pillow")

class HopfieldNetwork:
    N = 25
    def __init__(self):
        self.init_weight()
        
    def init_weight(self):
        self.weights = np.random.randn(self.N, self.N)
        np.fill_diagonal(self.weights, 0)
        self.weights = (self.weights + self.weights.T)/2

    def _animate_weight(self, weight_list: List[np.ndarray]):
        print(len(weight_list))
        fig = plt.figure()
        ims = []
        IMG_SIZE=10
        for i in range(IMG_SIZE):
            idx = int(i*len(weight_list)/IMG_SIZE)
            im = plt.imshow(weight_list[idx], animated=True)
            ims.append([im])
        ani = ArtistAnimation(fig, ims, interval=int(1500 / IMG_SIZE))
        ani.save("weight.gif")

    def train(self, image: np.ndarray,iter=100, print_weight=False):
        assert image.shape == (self.N,), "image shape must be (25,)"
        self.iter = iter
        self.init_weight()

        # Hebbian learning (初期化)
        self.weights = np.dot(image.reshape(self.N, 1), image.reshape(1, self.N))

        n = 1
        weight_list = []
        for _ in range(self.iter):
            # 非同期更新
            i = np.random.randint(0, self.N)
            j = np.random.randint(0, self.N)

            if i == j:
                continue

            x_new = self.predict(image)
            assert x_new.shape == (self.N,), "image shape must be (25,)"
            
            self.weights[i][j] = (n-1)/n *self.weights[i][j] +  (1/n)*x_new[i]*x_new[j]
            self.weights[j][i] = self.weights[i][j]

            weight_list.append(self.weights.reshape(self.N, self.N))
        if print_weight:
            self._animate_weight(weight_list)

    def predict(self, image: np.ndarray):
        # sign function -1 or 1
        sign = np.vectorize(lambda x: -1 if x < 0 else 1)
        y = sign(image)
        y = sign(self.weights @ y)
        return y

def main1():
    print("hello world")
    net = HopfieldNetwork()
    img = IMAGES[1]
    print(img)
    net.train(img,1000, True)

    print_image(img)
    y= net.predict(img)
    res = []
    for _ in range(100):
        y= net.predict(y)
        res.append(y.reshape(5,5))
    print_image(y)
    animate_image(res)

    # add noise
    for noise_range in range(5, 20, 5):
        print("noise range: ", noise_range)
        img_noised = add_noise(img, noise_range/100)
        print_image(img_noised)
        y = net.predict(img_noised)
        for _ in range(100):
            y= net.predict(y)
        print_image(y)
        print()

def test():
    index = 0
    for i in IMAGES.values():
        print(" -----   ", index, "   ----- ")
        print_image(i)
        print_image(add_noise(i))
        index+=1

if __name__ == "__main__":
    main1()
