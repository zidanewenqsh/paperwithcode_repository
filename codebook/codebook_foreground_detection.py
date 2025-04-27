import cv2
import numpy as np
from sklearn.cluster import KMeans

# 背景建模与前景检测类
class CodebookBackgroundSubtractor:
    def __init__(self, n_clusters=5, learning_rate=0.1, distance_threshold=20):
        self.n_clusters = n_clusters  # 码本中聚类的数量
        self.learning_rate = learning_rate  # 码本更新的学习率
        self.distance_threshold = distance_threshold  # 距离阈值
        self.kmeans = KMeans(n_clusters=n_clusters)
        self.codebook = None

    def fit(self, frame):
        """根据帧图像训练码本（背景模型）"""
        # 将图像转换为灰度图像或颜色特征
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 将图像展平为一维数组
        pixels = gray.reshape((-1, 1))

        # 使用K-means聚类生成码本
        self.kmeans.fit(pixels)
        self.codebook = self.kmeans.cluster_centers_

    def detect_foreground(self, frame):
        """检测前景"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pixels = gray.reshape((-1, 1))

        # 计算每个像素到码本的距离，找到最接近的聚类中心
        nearest_cluster = self.kmeans.predict(pixels)
        distances = np.abs(pixels - self.kmeans.cluster_centers_[nearest_cluster].reshape(-1, 1))
        
        # 前景掩码：如果距离大于阈值，认为是前景
        foreground_mask = np.sum(distances > self.distance_threshold, axis=1) > 0  # 判断是否为前景

        # 将一维mask转换回二维
        foreground_mask = foreground_mask.reshape(gray.shape)

        return foreground_mask.astype(np.uint8) * 255

    def update(self, frame):
        """更新码本"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pixels = gray.reshape((-1, 1))

        # 使用最近的聚类中心更新码本
        nearest_cluster = self.kmeans.predict(pixels)
        for i in range(self.n_clusters):
            cluster_pixels = pixels[nearest_cluster == i]
            if len(cluster_pixels) > 0:
                # 更新聚类中心
                self.kmeans.cluster_centers_[i] = np.mean(cluster_pixels, axis=0)

# 视频路径
videopath = r"D:\MyProjects\paperwithcode_repository\resources\video.mp4"
cap = cv2.VideoCapture(videopath)  # 使用视频路径打开视频文件

# 确保视频被成功打开
if not cap.isOpened():
    print("Error: Unable to open video.")
    exit()

bg_subtractor = CodebookBackgroundSubtractor(n_clusters=5, distance_threshold=20)

# 处理视频
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 训练码本（背景模型）
    if bg_subtractor.codebook is None:
        bg_subtractor.fit(frame)
    else:
        # 检测前景
        foreground_mask = bg_subtractor.detect_foreground(frame)

        # 更新背景模型
        bg_subtractor.update(frame)

        # 创建一个与原图相同大小的图像
        foreground_highlight = frame.copy()

        # 将前景部分标记为绿色
        foreground_highlight[foreground_mask == 255] = [0, 255, 0]  # 设置为绿色

        # 显示原始图像和标记前景的图像
        cv2.imshow("Original", frame)
        cv2.imshow("Foreground Highlighted", foreground_highlight)

    # 按'q'键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
