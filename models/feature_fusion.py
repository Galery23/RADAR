import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math
# 8.14

class Embedding(nn.Module):

    def __init__(self, z_dim=256):
        super(Embedding, self).__init__()

        self.z_dim = z_dim
        self.bn = nn.BatchNorm1d(3136)
        self.linear = nn.Linear(3136, self.z_dim)
        # self.linear2 = nn.Linear(32 * 7, self.z_dim)

    def forward(self, hyper_net, z):
        # z: (N, 3136, 7, 7) [16, 3136, 1152]
        z= self.bn(z)
        z = F.relu(self.linear(z.view(-1, 3136)))
        z = torch.mean(z, dim=0)  # (z_dim)
        w, b = hyper_net(z.view(-1, self.z_dim))
        return [w, b]

class HyperNetwork(nn.Module):

    def __init__(self, out_size, in_size, z_dim=256):
        super(HyperNetwork, self).__init__()
        self.z_dim = z_dim
        self.out_size = out_size
        self.in_size = in_size

        self.w1 = Parameter(torch.Tensor(self.z_dim, self.out_size * self.in_size))
        self.b1 = Parameter(torch.Tensor(self.out_size * self.in_size))
        self.w2 = Parameter(torch.Tensor(self.z_dim, self.out_size))
        self.b2 = Parameter(torch.Tensor(self.out_size))

        nn.init.xavier_uniform_(self.w1)
        nn.init.constant_(self.b1, 0)
        nn.init.xavier_uniform_(self.w2)
        nn.init.constant_(self.b2, 0)

    def forward(self, z):
        h_final = torch.matmul(z, self.w1) + self.b1
        w = h_final.view(self.in_size, self.out_size)
        b = torch.matmul(z, self.w2) + self.b2

        return [w, b]

# 多层感知机
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer() # 激活层
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        # ===================== update ===================== #
        self.in_size = in_features
        self.out_size = out_features
        self.z_dim = 256
        self.hyper = HyperNetwork(out_size=self.out_size, in_size=self.in_size, z_dim=self.z_dim) # HyperNetwork
        self.embed = Embedding(z_dim=self.z_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        # print("111 x.shape = ", x.shape)
        # ===================== update ===================== #
        w, b = self.embed(self.hyper, x)
        # print("w.shape = ", w.shape)
        # print("b.shape = ", b.shape)
        x = torch.matmul(x, w) + b
        # print("222 x.shape = ", x.shape)
        # ===================== update ===================== #
        return x

# 特征融合模块
class FeatureFusionBlock(nn.Module):
    # 初始化
    def __init__(self, xyz_dim, rgb_dim, mlp_ratio=4.):
        super().__init__()

        self.xyz_dim = xyz_dim
        self.rgb_dim = rgb_dim

        self.xyz_norm = nn.LayerNorm(xyz_dim)
        self.xyz_mlp = Mlp(in_features=xyz_dim, hidden_features=int(xyz_dim * mlp_ratio), act_layer=nn.GELU, drop=0.)

        self.rgb_norm = nn.LayerNorm(rgb_dim)
        self.rgb_mlp = Mlp(in_features=rgb_dim, hidden_features=int(rgb_dim * mlp_ratio), act_layer=nn.GELU, drop=0.)

        self.rgb_head = nn.Linear(rgb_dim, 256)
        self.xyz_head = nn.Linear(xyz_dim, 256)
        
        self.T = 1

    # 特征融合
    def feature_fusion(self, xyz_feature, rgb_feature):

        xyz_feature = self.xyz_mlp(self.xyz_norm(xyz_feature))
        rgb_feature = self.rgb_mlp(self.rgb_norm(rgb_feature))

        feature = torch.cat([xyz_feature, rgb_feature], dim=2)

        return feature

    # 对比损失
    def contrastive_loss(self, q, k):
        # normalize 标准化
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # print("q = ", q)
        # print("k = ", k)
        # gather all targets
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        # print("logits = ", logits)
        # print("labels = ", labels)
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def reparameterize(self, mu, logvar):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    # 前向传递
    def forward(self, xyz_feature, rgb_feature):
        feature = self.feature_fusion(xyz_feature, rgb_feature)

        feature_xyz = feature[:,:, :self.xyz_dim]
        feature_rgb = feature[:,:, self.xyz_dim:]

        q = self.rgb_head(feature_rgb.view(-1, feature_rgb.shape[2]))
        k = self.xyz_head(feature_xyz.view(-1, feature_xyz.shape[2]))

        xyz_feature = xyz_feature.view(-1, xyz_feature.shape[2])
        rgb_feature = rgb_feature.view(-1, rgb_feature.shape[2])

        patch_no_zeros_indices = torch.nonzero(torch.all(xyz_feature != 0, dim=1))
        
        loss = self.contrastive_loss(q[patch_no_zeros_indices,:].squeeze(), k[patch_no_zeros_indices,:].squeeze()) # patch-wise contrastive loss，逐片对比损失函数值
        # print("loss = ", loss)

        return loss

