import cv2
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt


def reprojection_error(params):
    # 从params中提取相机位姿和三维点坐标
    camera_poses = params[:12].reshape(3, 4)
    points3d = params[12:].reshape(-1, 3)

    # 计算重投影误差
    error = []
    for i in range(len(pts1)):
        # 将三维点坐标投影到图像平面
        proj_pt1 = project_point(points3d[i], camera_poses[:, :3], camera_poses[:, 3])
        proj_pt2 = project_point(points3d[i], camera_poses[:, 4:7], camera_poses[:, 7:10])
        proj_pt3 = project_point(points3d[i], camera_poses[:, 8:], camera_poses[:, 11:])

        # 计算重投影误差
        error.append(np.linalg.norm(proj_pt1 - pts1[:, i]))
        error.append(np.linalg.norm(proj_pt2 - pts2[:, i]))
        error.append(np.linalg.norm(proj_pt3 - pts3[:, i]))

    return np.array(error)

def project_point(point3d, rotation, translation):
    # 将三维点坐标投影到图像平面
    proj_pt = np.dot(rotation, point3d) + translation
    return proj_pt[:2] / proj_pt[2]


# 读取输入照片
img1 = cv2.imread('../images/image1.png')
img2 = cv2.imread('../images/image2.png')
img3 = cv2.imread('../images/image3.png')

# 检测并匹配特征点
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
kp3, des3 = sift.detectAndCompute(img3, None)

bf = cv2.BFMatcher()
matches12 = bf.knnMatch(des1, des2, k=2)
matches23 = bf.knnMatch(des2, des3, k=2)

# 筛选良好的匹配点
good_matches12 = []
for m, n in matches12:
    if m.distance < 0.75 * n.distance:
        good_matches12.append(m)

good_matches23 = []
for m, n in matches23:
    if m.distance < 0.75 * n.distance:
        good_matches23.append(m)

# 计算相机位姿和物体三维坐标
pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches12]).T
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches12]).T
pts3 = np.float32([kp3[m.trainIdx].pt for m in good_matches23]).T

focal_length = 1000.0
principal_point = (512.0, 384.0)

params = np.zeros(12 + 3 * len(pts1))
# 使用 least_squares 优化相机位姿和三维点坐标
res = least_squares(reprojection_error, params)

# 输出相机位姿和物体三维坐标
camera_poses = res.x[:12].reshape(3, 4)
points3d = res.x[12:].reshape(-1, 3)

print("相机位姿:")
print(camera_poses)
print("物体三维坐标:")
print(points3d)

# 创建 3D 散点图
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制三维点
ax.scatter(points3d[:, 0], points3d[:, 1], points3d[:, 2], s=50, color='b')

# 绘制相机位姿
for i in range(0, len(camera_poses), 4):
    R = camera_poses[i:i + 3, :3]
    t = camera_poses[i:i + 3, 3]

    # 绘制相机坐标系
    ax.quiver(t[0], t[1], t[2], R[0, 0], R[0, 1], R[0, 2], color='r', length=0.2)
    ax.quiver(t[0], t[1], t[2], R[1, 0], R[1, 1], R[1, 2], color='g', length=0.2)
    ax.quiver(t[0], t[1], t[2], R[2, 0], R[2, 1], R[2, 2], color='b', length=0.2)

# 设置坐标轴标签和标题
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Reconstructed 3D Scene')

# 显示图像
plt.show()