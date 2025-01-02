import os

import torch
import matplotlib.pyplot as plt
import numpy as np

# 뭔가 오류있는거같음 일단은 save_image만 사용가능
# 나중에 tensorboard나 matplotlib으로 그래프 그리는것도 추가하면 될듯
class Visualizer:
    def __init__(self, path='result/'):
        # plt.ion()
        self.path = path
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def show_images(self, img1, img2=None, title1="Image 1", title2="Image 2"):
        """
        두 이미지를 나란히 표시하거나 한 개의 이미지를 표시합니다.
        """
        plt.close('all')
        if img2 is not None:
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            self._imshow(img1, axs[0], title1)
            self._imshow(img2, axs[1], title2)
        else:
            fig, ax = plt.subplots()
            self._imshow(img1, ax, title1)
        plt.show(block=True)
        plt.pause(0.1)

    def save_image(self, img, filename):
        """
        이미지를 파일로 저장합니다.
        """
        file_path = os.path.join(self.path, filename)
        fig, ax = plt.subplots()
        self._imshow(img, ax, filename)
        plt.savefig(file_path) 
        plt.close(fig) 

    def _imshow(self, tensor, ax, title=None):
        """
        텐서를 이미지로 변환하여 하나의 축에 표시합니다.
        """
        tensor = tensor.permute(1, 2, 0).detach()
        tensor = tensor.cpu().numpy()
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        ax.imshow(tensor)
        if title:
            ax.set_title(title)
        ax.axis('off') 

