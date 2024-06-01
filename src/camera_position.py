import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # 读取输入照片
    current_dir = os.getcwd()  # 获取当前工作目录
    path_of_img1 = os.path.join(current_dir, "src/images/image1.png") 
    path_of_img2 = os.path.join(current_dir, "src/images/image2.png") 
    path_of_img3 = os.path.join(current_dir, "src/images/image3.png") 
    img1 = cv2.imread(path_of_img1)
    img2 = cv2.imread(path_of_img2)
    img3 = cv2.imread(path_of_img3)


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
    #
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
    #
    # plt.show()

    bf = cv2.BFMatcher()
    matches12 = bf.knnMatch(des1, des2, k=2)
    matches23 = bf.knnMatch(des2, des3, k=2)

    # 筛选良好的匹配点
    good_matches12 = []
    for m, n in matches12:
        if m.distance < 0.8 * n.distance:
            good_matches12.append(m)

    good_matches23 = []
    for m, n in matches23:
        if m.distance < 0.8 * n.distance:
            good_matches23.append(m)

    num_good_matches12 = len(good_matches12)
    num_good_matches23 = len(good_matches23)

    target_size = min(num_good_matches12, num_good_matches23)

    # good_matches12.sort(key=lambda x: x.distance)
    # good_matches23.sort(key=lambda x: x.distance)

    if num_good_matches12 > target_size:
        good_matches12 = good_matches12[:target_size]
    if num_good_matches23 > target_size:
        good_matches23 = good_matches23[:target_size]

    # 可视化匹配结果
    # fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    #
    # # 可视化 Image 1 和 Image 2 的匹配结果
    # matches_img1_2 = cv2.drawMatches(img1, kp1, img2, kp2, good_matches12, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # axes[0].imshow(matches_img1_2)
    # axes[0].set_title('Matches between Image 1 and Image 2')
    #
    # # 可视化 Image 2 和 Image 3 的匹配结果
    # matches_img2_3 = cv2.drawMatches(img2, kp2, img3, kp3, good_matches23, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # axes[1].imshow(matches_img2_3)
    # axes[1].set_title('Matches between Image 2 and Image 3')
    #
    # plt.show()
    # 计算相机位姿和物体三维坐标
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches12])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches12])
    pts3 = np.float32([kp3[m.trainIdx].pt for m in good_matches23])

    # 打印特征点坐标
    # print("pts1 shape:", pts1.shape)
    # print("pts1:\n", pts1)
    #
    # print("pts2 shape:", pts2.shape)
    # print("pts2:\n", pts2)
    #
    # print("pts3 shape:", pts3.shape)
    # print("pts3:\n", pts3)

    K = np.array([[1000, 0, 640],
                  [0, 1000, 360],
                  [0, 0, 1]])

    essential_mat1, _ = cv2.findEssentialMat(pts1, pts2, K)
    essential_mat2, _ = cv2.findEssentialMat(pts2, pts3, K)
    essential_mat3, _ = cv2.findEssentialMat(pts1, pts3, K)

    # # 计算相机外参
    _, R1, t1, _ = cv2.recoverPose(essential_mat1, pts1, pts2, K, K)
    _, R2, t2, _ = cv2.recoverPose(essential_mat2, pts2, pts3, K, K)
    _, R3, t3, _ = cv2.recoverPose(essential_mat3, pts1, pts3, K, K)

    # 构建投影矩阵
    P1 = K @ np.hstack((R1, t1))
    P2 = K @ np.hstack((R2, t2))
    P3 = K @ np.hstack((R3, t3))

    pts1 = pts1.T
    pts2 = pts2.T
    pts3 = pts3.T

    # 计算三维点坐标
    X = cv2.triangulatePoints(P1, P2, pts1, pts2)
    Y = cv2.triangulatePoints(P2, P3, pts2, pts3)
    Z = cv2.triangulatePoints(P1, P3, pts1, pts3)
    #
    # # 归一化三维点坐标
    X /= X[3]
    Y /= Y[3]
    Z /= Z[3]

    # 打印结果
    print("物体三维坐标:")
    print("X:\n", X[:3])
    print("Y:\n", Y[:3])
    print("Z:\n", Z[:3])

    # 相机坐标
    camera1_pos = -np.dot(R1.T, t1)
    camera2_pos = -np.dot(R2.T, t2)
    camera3_pos = -np.dot(R3.T, t3)

    print("相机坐标:")
    print("Camera 1:", camera1_pos)
    print("Camera 2:", camera2_pos)
    print("Camera 3:", camera3_pos)

    # 3D可视化
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制3D点云
    ax.scatter(X[0], Y[0], Z[0], c='r', marker='o')

    # 绘制相机位置
    ax.scatter(camera1_pos[0], camera1_pos[1], camera1_pos[2], c='b', marker='^', s=100)
    ax.scatter(camera2_pos[0], camera2_pos[1], camera2_pos[2], c='g', marker='^', s=100)
    ax.scatter(camera3_pos[0], camera3_pos[1], camera3_pos[2], c='y', marker='^', s=100)

    # 设置坐标轴标签和标题
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Reconstruction')

    plt.show()

    # 2D 可视化
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # 第一个子图
    ax1.scatter(X[0], Y[0], c='r', marker='o')
    ax1.scatter(camera1_pos[0], camera1_pos[1], c='b', marker='^', s=100)
    ax1.scatter(camera2_pos[0], camera2_pos[1], c='g', marker='^', s=100)
    ax1.scatter(camera3_pos[0], camera3_pos[1], c='y', marker='^', s=100)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('2D Reconstruction - View 1')

    # 第二个子图
    ax2.scatter(X[0], Z[0], c='r', marker='o')
    ax2.scatter(camera2_pos[0], camera2_pos[2], c='g', marker='^', s=100)
    ax2.scatter(camera1_pos[0], camera1_pos[2], c='b', marker='^', s=100)
    ax2.scatter(camera3_pos[0], camera3_pos[2], c='y', marker='^', s=100)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_title('2D Reconstruction - View 2')

    # 第三个子图
    ax3.scatter(Y[0], Z[0], c='r', marker='o')
    ax3.scatter(camera3_pos[1], camera3_pos[2], c='y', marker='^', s=100)
    ax3.scatter(camera1_pos[1], camera1_pos[2], c='b', marker='^', s=100)
    ax3.scatter(camera2_pos[1], camera2_pos[2], c='g', marker='^', s=100)
    ax3.set_xlabel('Y')
    ax3.set_ylabel('Z')
    ax3.set_title('2D Reconstruction - View 3')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
