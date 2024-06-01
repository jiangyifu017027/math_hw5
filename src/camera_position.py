import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():
    # 读取输入照片
    img1 = cv2.imread('../images/image1.png')
    img2 = cv2.imread('../images/image2.png')
    img3 = cv2.imread('../images/image3.png')

    # 检验图像是否正确读入
    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    #
    # axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    # axes[0].set_title('Image 1')
    #
    # axes[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    # axes[1].set_title('Image 2')
    #
    # axes[2].imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
    # axes[2].set_title('Image 3')

    # plt.show()

    # 检测并匹配特征点
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    kp3, des3 = sift.detectAndCompute(img3, None)

    # 可视化特征点
    # 检验特征点处理的正确性
    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    #
    # axes[0].imshow(cv2.drawKeypoints(img1, kp1, None))
    # axes[0].set_title('Image 1 with SIFT Keypoints')
    #
    # axes[1].imshow(cv2.drawKeypoints(img2, kp2, None))
    # axes[1].set_title('Image 2 with SIFT Keypoints')
    #
    # axes[2].imshow(cv2.drawKeypoints(img3, kp3, None))
    # axes[2].set_title('Image 3 with SIFT Keypoints')

    # plt.show()


if __name__ == '__main__':
    main()
