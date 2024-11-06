import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, adjacency_matrix, x):
        batch_size, num_joints, _ = x.size()
        x = x.view(-1, x.size(-1))  # (batch_size * num_joints, in_features)
        support = torch.mm(x, self.weight)  # (batch_size * num_joints, out_features)
        out = torch.bmm(adjacency_matrix.view(batch_size, num_joints, num_joints), support.view(batch_size, num_joints, -1))  # (batch_size, num_joints, out_features)
        out += self.bias  # (batch_size, num_joints, out_features)
        return out

class SkeletonGraphModule(nn.Module):
    def __init__(self, num_joints, in_features, hidden_features, out_shape_params, out_pose_params):
        super(SkeletonGraphModule, self).__init__()
        self.num_joints = num_joints
        
        self.gcn1 = GraphConvolution(in_features, hidden_features)
        self.gcn2 = GraphConvolution(hidden_features, hidden_features)
        
        self.fc_shape = nn.Linear(hidden_features, out_shape_params)
        self.fc_pose = nn.Linear(hidden_features, out_pose_params)

    def forward(self, joint_features, adjacency_matrix):
        x = F.relu(self.gcn1(adjacency_matrix, joint_features))
        x = F.relu(self.gcn2(adjacency_matrix, x))

        shape_params = self.fc_shape(x)
        pose_params = self.fc_pose(x)

        return shape_params, pose_params

# 示例使用
if __name__ == "__main__":
    batch_size = 16
    num_joints = 24
    in_features = 128
    hidden_features = 64
    out_shape_params = 10
    out_pose_params = 96

    joint_features = torch.rand(batch_size, num_joints, in_features)
    adjacency_matrix = torch.eye(num_joints).repeat(batch_size, 1, 1)

    sgm = SkeletonGraphModule(num_joints, in_features, hidden_features, out_shape_params, out_pose_params)

    shape_params, pose_params = sgm(joint_features, adjacency_matrix)
    print("Shape Parameters:", shape_params.shape)
    print("Pose Parameters:", pose_params.shape)
