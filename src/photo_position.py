"""
输入：使用手机绕着物体拍摄若干照片
输出：手机相机的定位及被拍物体的三维点坐标
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 初始化相机参数
camera_matrix = np.array([[1000, 0, 320],
                        [0, 1000, 240],
                        [0, 0, 1]])
dist_coeffs = np.array([0.1, 0.01, 0.001, 0.0001, 0])

# 定位相机姿态
def estimate_pose(img_points, obj_points):
    _, rvec, tvec, _ = cv2.solvePnPRansac(obj_points, img_points, camera_matrix, dist_coeffs)
    return rvec, tvec

# 三维重建
def reconstruct_3d(img_points1, img_points2, rvec1, tvec1, rvec2, tvec2):
    points4d = cv2.triangulatePoints(camera_matrix, camera_matrix, img_points1.T, img_points2.T)
    points3d = (points4d[:3] / points4d[3]).T
    return points3d

# 主函数
def main():
    # 拍摄多张照片
    imgs = []
    img_points = []
    obj_points = []
    for i in range(5):
        img = cv2.imread(f'image{i+1}.jpg')
        imgs.append(img)
        
        # 在图像上手动选择特征点
        pts2d = []
        pts3d = []
        while True:
            pt = cv2.waitKey(0)
            if pt == ord('q'):
                break
            x, y = cv2.cv2.getWindowProperty('image', cv2.cv2.WND_PROP_CURSOR_POS)
            pts2d.append([x, y])
            pts3d.append([i*10, j*10, k*10])  # 假设三维点坐标
            img = cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.imshow('image', img)
        img_points.append(np.array(pts2d, dtype=np.float32))
        obj_points.append(np.array(pts3d, dtype=np.float32))
        
    # 估计相机姿态
    rvecs = []
    tvecs = []
    for i in range(len(imgs)):
        rvec, tvec = estimate_pose(img_points[i], obj_points[i])
        rvecs.append(rvec)
        tvecs.append(tvec)
    
    # 三维重建
    all_3d_points = []
    for i in range(len(imgs)):
        for j in range(i+1, len(imgs)):
            points3d = reconstruct_3d(img_points[i], img_points[j], rvecs[i], tvecs[i], rvecs[j], tvecs[j])
            all_3d_points.extend(points3d)
    
    # 可视化三维点云
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter([p[0] for p in all_3d_points], [p[1] for p in all_3d_points], [p[2] for p in all_3d_points])
    plt.show()


if __name__ == '__main__':
    main()
