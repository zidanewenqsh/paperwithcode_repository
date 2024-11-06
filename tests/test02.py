import torch
import torch.nn as nn
import torch.nn.functional as F

class SkeletonAwareModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SkeletonAwareModule, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, point_cloud, skeleton):
        # 特征提取
        point_cloud_features = F.relu(self.fc1(point_cloud))
        skeleton_features = F.relu(self.fc1(skeleton))

        # 将骨架特征扩展到与点云特征相同的形状
        skeleton_features_expanded = skeleton_features.mean(dim=1, keepdim=True).expand(-1, point_cloud_features.size(1), -1)

        # 合并特征
        combined_features = point_cloud_features + skeleton_features_expanded
        
        # 输出
        output = self.fc2(combined_features)
        return output

# 示例使用
if __name__ == "__main__":
    batch_size = 16
    num_points = 1024
    point_features = 3  # x, y, z坐标
    skeleton_features = 3  # 骨架点特征，比如x, y, z坐标

    # 创建随机输入
    point_cloud = torch.rand(batch_size, num_points, point_features)
    skeleton = torch.rand(batch_size, 17, skeleton_features)  # 假设有17个骨架点

    # 初始化模块
    model = SkeletonAwareModule(input_dim=point_features, hidden_dim=64, output_dim=3)
    
    # 前向传播
    output = model(point_cloud, skeleton)
    print(output.shape)  # 输出的形状
