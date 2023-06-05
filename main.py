from lib import IMAGES, add_noise, print_image
import sys 
import numpy as np
import os
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

        plt.clf()
        # save last image
        plt.imshow(weight_list[-1])
        plt.savefig("weight.png")

    def train(self, image: List[np.ndarray],iter=100, print_weight=False):
        for i in image:
            assert i.shape == (self.N,), "image shape must be (25,)"
        self.iter = iter
        self.init_weight()

        Q = len(image)
        assert Q > 0, "image must be more than 1"

        # Hebbian learning (初期化)
        self.weights = np.sum([np.dot(i.reshape(self.N, 1), i.reshape(1, self.N)) for i in image], axis=0, dtype=np.float64) 
        self.weights /= Q
        np.fill_diagonal(self.weights, 0)
        self.weights = (self.weights + self.weights.T)/2

        # return

        n = 1
        weight_list = []
        for _ in range(self.iter):
            # 非同期更新
            image_index = np.random.randint(0, Q)
            i = np.random.randint(0, self.N)
            j = np.random.randint(0, self.N)

            if i == j:
                continue

            x_new = self.predict(image[image_index])
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

    def predict_n(self, image:np.ndarray, n:int=100):
        y = image
        for _ in range(n):
            y = self.predict(y)
        return y

    def check_acc(self,img:np.ndarray, random_rate:float, iteration=100):
        acc = 0
        for _ in range(iteration):
            x = add_noise(img, random_rate)
            y = self.predict_n(x)
            if np.all(y == img):
                acc+=1
        print("acc: ", acc/100, "at", random_rate)
        return acc

def problem1():
    print("--- problem 1 ---")
    net = HopfieldNetwork()
    img = IMAGES[1]
    print(img)
    net.train([img],1000, True)

    print_image(img)
    y =net.predict_n(img) 
    print_image(y)

    acc_list = []
    print_image(img)
    for noise_range in range(5, 50):
        acc = net.check_acc(img, noise_range/100)
        acc_list.append(acc)

    # plot it 
    plt.plot(range(5, 50), acc_list)
    plt.xlabel("noise range")
    plt.ylabel("accuracy")
    plt.savefig("result.png")


def problem2():
    print("--- problem 2 ---")
    net = HopfieldNetwork()
    img_list= list(IMAGES.values())[0:6]
    print(img_list.__len__)
    for i in img_list:
        print_image(i)

    net.train(img_list,1000, True)

    print(img_list[0])
    y = net.predict_n(img_list[0])
    print_image(y)

    plt.clf()
    acc_list = {}
    noise_rate_list = {}
    for key in range(len(img_list)):
        for p in range(0,101, 5):
            img = img_list[key]
            acc = net.check_acc(img, p/100)
            if key not in acc_list:
                acc_list[key] = []
            acc_list[key].append(acc)

            if key not in noise_rate_list:
                noise_rate_list[key] = []
            noise_rate_list[key].append(p/100)

    # plot it
    for key in acc_list:
        plt.plot(noise_rate_list[key], acc_list[key], label="image {}".format(key))
    plt.xlabel("noise range")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig("result2.png")
    plt.show()

def tmp():
    img_list= list(IMAGES.values())[0:6]
    # subplot images
    fig, ax = plt.subplots(2, 3)
    for i in range(2):
        for j in range(3):
            ax[i, j].imshow(img_list[i*3+j].reshape(5,5))

    plt.show()

    sys.exit()

if __name__ == "__main__":
    tmp()
    # problem1()
    problem2()
