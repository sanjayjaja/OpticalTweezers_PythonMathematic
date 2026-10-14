"""
Name: Sanjay Khadka
Date: 11-10-2026

Optical tweezer models that loads and processes images
and processess them to determine if an atom has been loaded into them

"""


import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time 
from PIL import Image
from pathlib import Path

# Class that loads in images and has image processing functions

class ImageProcessing():
    def __init__(self):
        self.running = True
        
    # Reading and converting image into a colormap
    def load_img(self, name, color = False):
        # Defaults to a RGB map that has a red, green and blue value between 0-255 for each pixel
        if color:
            return cv.imread(f'data/{name}', cv.IMREAD_COLOR) 
        else:   
            return cv.imread(f'data/{name}', cv.IMREAD_GRAYSCALE)

    def load_tga(self, name):
        img = Image.open(f"S41/{name}.tga")

        arr = np.array(img.convert("L"))

        return arr


    def load_tga_folder(self):

        folder = Path("S41/")
        img_array = np.zeros((210, 140, 300))
        avg = np.zeros((210, 140))
        i = 0

        for file in folder.glob("*.tga"):
            img = Image.open(file)
            arr_test = np.array(img)
            #print(np.max(arr_test[:, 1]))

            arr = arr_test[:, :, 1]

            img_d, dst = self.nlm_denoise(arr, 9, 7, 11)
                      
            img_array[:, :, i] = arr[10:]
            i += 1

            avg += arr[10:] / 300

        return img_array, avg

    # def load_tga_folder(self):
    
    #         folder_path = r"S41"
    #         filenames = sorted(glob(os.path.join(folder_path, "*.tga")))
    #         print(filenames)
    #         matrixlist = []
    
    #         for filename in filenames[:300]:
    #             img = cv.imread(f'S41/{filename[4:]}', cv.IMREAD_GRAYSCALE)
    #             print(img)
    
    #             # Select elements starting from index 10
    #             data = img.flatten()[10:]
    
    #             # Partition into rows of 140 elements
    #             matrix = data.reshape(-1, 140)
    
    #             matrixlist.append(matrix)
    
    #         avgmatrix = np.mean(matrixlist, axis=0)
    
    #         plt.imshow(avgmatrix)#, cmap='gray')#, vmin=np.min(avg[0:100]), vmax = np.max(avg[0:100]))
    #         plt.show()
    
    #         return matrix, avgmatrix
    
    # Denoising with Non-Local means function 
    def nlm_denoise(self, name, filter = 10, window = 7, s_window= 21, show = False):
        #img = self.load_img(name)
        dst = cv.fastNlMeansDenoising(name, None, filter, window, s_window)

        if show:
            ax1 = plt.subplot(121)
            plt.imshow(name)
            ax1.set_title("Original")

            ax2 = plt.subplot(122)
            plt.imshow(dst)
            ax2.set_title("Denoise test")
            plt.show()

        return name, dst

    # Testing NLM denoising 
    def test_denoise(self, name, filter, w, s_w):
        #img = self.load_img(name)

        for h in range(1, filter):
            img, dst = self.nlm_denoise(name, h, w, s_w)

            ax1 = plt.subplot(121)
            plt.imshow(img)
            ax1.set_title("Original")

            ax2 = plt.subplot(122)
            plt.imshow(dst)
            ax2.set_title("Denoise test")

            plt.savefig(f'test/denoising/NLM_filter{h}_{w}_{s_w}.jpg', dpi = 250)
            plt.close()



    def gaussian_filter(self, img):
        pass

    def noise_correction(self, img):
        pass


# class ImageProcessing(ImageLoading):

#     def __init__(self):
#         super().__init__()
        
#     def im_process(self, img):
#         pass

#     def gaussian

if __name__ == '__main__':
    ip = ImageProcessing()
