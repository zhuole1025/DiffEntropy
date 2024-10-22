import cv2, glob
import time
# from scipy.misc import imread as scipy_imread
# from skimage import io
from PIL import Image
import numpy as np
# from keras.preprocessing import image as keras_imread
# import mxnet as mx  # mx.image

def ReadImage_SpeedTest():
    image_path = '../dog.png'
    # image_path = './Test_Image/image_2_4608_3456.jpg'
    readTimes = 100
    img = cv2.imread(image_path)
    print("图像大小高*宽： {} * {}: ".format(img.shape[0], img.shape[1]))

    ## OpenCV 读取图片
    time1 = time.time()
    for i in range(readTimes):
        img_opencv = cv2.imread(image_path)
        height, width = img_opencv.shape[:2]
    print("OpenCV读取图片速度：", round(readTimes / (time.time() - time1), 2), "  张图片每秒（files/sec）")

    # ## scipy 读取图片
    # time1 = time.time()
    # for i in range(readTimes):
    #     img_scipy = scipy_imread(image_path)
    # print("scipy读取图片速度：", round(readTimes / (time.time() - time1), 2), "  张图片每秒（files/sec）")

    # ## skimage 读取图片
    # time1 = time.time()
    # for i in range(readTimes):
    #     img_skimage = io.imread(image_path)
    # print("skimage读取图片速度：", round(readTimes / (time.time() - time1), 2), "  张图片每秒（files/sec）")


    ## PIL 读取图片
    time1 = time.time()
    for i in range(readTimes):
        img = Image.open(image_path) ## img是Image内部的类文件，还需转换
        width, height = img.size
        # img_PIL = np.array(img) ## numpy转换
    print("PIL 读取图片速度：", round(readTimes / (time.time() - time1), 2), "  张图片每秒（files/sec）")

    # ## keras 读取图片
    # time1 = time.time()
    # for i in range(readTimes):
    #     img = keras_imread.load_img(image_path)
    #     img_keras = keras_imread.img_to_array(img)
    # print("keras 读取图片速度：", round(readTimes / (time.time() - time1), 2), "  张图片每秒（files/sec）")

    # ## mxnet 读取图片
    # time1 = time.time()
    # for i in range(readTimes):
    #     # img_mxnet = mx.image.imdecode(open(image_path,'rb').read())
    #     img_mxnet = mx.image.imread(image_path)
    # mx.nd.waitall()
    # print("mxnet 读取图片速度：", round(readTimes / (time.time() - time1), 2), "  张图片每秒（files/sec）")

ReadImage_SpeedTest()