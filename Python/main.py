# optical tweezer model by Sanjay Khadka

import models.imageprocessing as IP
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# main class 
class Optical_Tweezer(IP.ImageProcessing):
    def __init__(self):
        super().__init__()

    # def img_processing(self):
    #     img = self.IP.load_img()


if __name__ == '__main__':

    OP = Optical_Tweezer()

    okt11 = False

    if okt11:
        OP.test_denoise("canvas.png", 15, 7, 11)
        print("UV test")

    arr, avg = OP.load_tga_folder()

    plt.imshow(avg[20:90, 35:105])
    plt.show()




