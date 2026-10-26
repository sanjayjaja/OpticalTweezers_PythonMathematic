"""
Name: Sanjay Khadka
Date: 11-10-2026

Optical tweezer models that loads and processes images
and processess them to determine if an atom has been loaded into them

Update by Nico: preserve 16-bit camera counts, validate image headers,
and load any number of images in trial order.

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

    @staticmethod
    def read_camera_tga(file):
        # Camera pixels use both bytes as one 16-bit count, not colour/alpha.
        import struct
        with open(file, "rb") as stream:
            header = stream.read(18)
            if len(header) != 18:
                raise ValueError(f"Incomplete TGA header: {file}")
            fields = struct.unpack("<BBBHHBHHHHBB", header)
            id_length, colour_map, image_type = fields[:3]
            width, height, depth, descriptor = fields[8:]
            if colour_map != 0 or image_type != 3 or depth != 16:
                raise ValueError(f"Expected an uncompressed 16-bit grayscale camera TGA: {file}")
            stream.read(id_length)
            pixels = np.frombuffer(stream.read(width * height * 2), dtype="<u2")
        if pixels.size != width * height:
            raise ValueError(f"Incomplete pixel data: {file}")
        image = pixels.reshape(height, width)
        # Match Pillow's displayed orientation, including bottom-up TGA files.
        if not descriptor & 32:
            image = image[::-1]
        if descriptor & 16:
            image = image[:, ::-1]
        return image

    def load_tga(self, name):
        return self.read_camera_tga(Path("S41") / f"{name}.tga")

    def load_tga_folder(self, folder="S41", crop_rows=10):
        import re
        folder = Path(folder)
        # Natural ordering keeps trial 2 before trial 10 for bootstrap blocks.
        key = lambda p: [int(t) if t.isdigit() else t.lower()
                         for t in re.split(r"(\d+)", p.name)]
        files = sorted(folder.glob("*.tga"), key=key)
        if not files:
            raise FileNotFoundError(f"No .tga images found in {folder.resolve()}")
        images = []
        for file in files:
            image = self.read_camera_tga(file)
            if not 0 <= crop_rows < image.shape[0]:
                raise ValueError("Invalid number of rows to crop.")
            images.append(image[crop_rows:])
        # Float conversion prevents unsigned subtraction from wrapping around.
        img_array = np.stack(images, axis=2).astype(float)
        self.image_files = [f.name for f in files]
        return img_array, np.mean(img_array, axis=2)

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
