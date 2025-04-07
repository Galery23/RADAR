import torch
import numpy as np
import torch.nn as nn
# import timm # PyTorch Image Models库是一个用于计算机视觉任务的Python库，提供了如ResNet、EfficientNet、ViT等常见深度学习模型架构的实现
import models.vision_transformer_prompts as vit # vision_transformer_prompts
# import models.vision_transformer as vit # 原版vision transformer
from utils.misc import init_weights

from timm.models.layers import DropPath, trunc_normal_
from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN
# 8.14


# 整体模型的类
class Model(torch.nn.Module):
    # 初始化
    def __init__(self, device, rgb_backbone_name='vit_base_patch8_224_dino', out_indices=None, checkpoint_path='',
                 pool_last=False, xyz_backbone_name='Point_MAE', group_size=128, num_group=1024):
        super().__init__()
        # 'vit_base_patch8_224_dino'
        # Determine if to output features.
        self.device = device

        # self.load_vit_ckpt = './checkpoints/dino_vitbase8_pretrain.pth'
        # checkpoint_path = './checkpoints/dino_vitbase8_pretrain.pth'

        # kwargs = {'features_only': True if out_indices else False}
        # if out_indices:
        #     kwargs.update({'out_indices': out_indices})

        # print("checkpoint_path = ", checkpoint_path)
        # RGB backbone，提取RGB图像特征的主干网络Vision Transformer
        # self.rgb_backbone = timm.create_model(model_name=rgb_backbone_name, pretrained=True, checkpoint_path=checkpoint_path,
        #                                   **kwargs)
        self.rgb_backbone = getattr(vit, rgb_backbone_name)(pretrained=True)

        # nn.Embedding处理图像和3d点云的token_type_embedding嵌入
        self.token_type_embeddings = nn.Embedding(2, 768)
        self.token_type_embeddings.apply(init_weights)

        ## XYZ backbone，提取点云特征的主干网络Point Transformer
        if xyz_backbone_name=='Point_MAE':
            self.xyz_backbone=PointTransformer(group_size=group_size, num_group=num_group)
            self.xyz_backbone.load_model_from_ckpt("checkpoints/pointmae_pretrain.pth")
        elif xyz_backbone_name=='Point_Bert':
            self.xyz_backbone=PointTransformer(group_size=group_size, num_group=num_group, encoder_dims=256)
            self.xyz_backbone.load_model_from_pb_ckpt("checkpoints/Point-BERT.pth")

        # ===================== Prompt ===================== #

        self.prompt_type = 'input'  # prompt类型
        # self.prompt_type = 'attention'  # prompt类型

        prompt_length = 16  # prompt长度
        self.prompt_length = prompt_length

        embed_dim = 768  # embedding维度，和隐藏层大小相同

        self.learnt_p = True  # 是否需要模态感知提示，默认值为True

        self.prompt_layers = [0, 1, 2, 3, 4, 5]  # 需要进行确实感知提示的层数，默认值为[0,1,2,3,4,5]
        self.multi_layer_prompt = True  # 是否需要多层prompt，默认值为True
        prompt_num = len(self.prompt_layers) if self.multi_layer_prompt else 1  # prompt数量

        # 完整prompt（表示没有模态缺失时）
        complete_prompt = torch.zeros(prompt_num, prompt_length, embed_dim)  # 完整prompt，初始为input-level（输入级别）prompt
        complete_prompt[:, 0:1, :].fill_(1)
        if self.learnt_p and self.prompt_type == 'attention':  # 若为attention-level（注意力级别）prompt
            complete_prompt[:, prompt_length // 2 + 0: prompt_length // 2 + 1, :].fill_(1)
        self.complete_prompt = nn.Parameter(complete_prompt)  # 创建可训练的参数

        # 图像模态缺失prompt
        missing_img_prompt = torch.zeros(prompt_num, prompt_length, embed_dim)  # 图像模态缺失的prompt
        missing_img_prompt[:, 1:2, :].fill_(1)
        if self.learnt_p and self.prompt_type == 'attention':  # 若为attention-level（注意力级别）prompt
            missing_img_prompt[:, prompt_length // 2 + 1: prompt_length // 2 + 2, :].fill_(1)
        self.missing_img_prompt = nn.Parameter(missing_img_prompt)  # 创建可训练的参数

        # 点云模态缺失prompt
        missing_pc_prompt = torch.zeros(prompt_num, prompt_length, embed_dim)  # 文本模态缺失的prompt
        missing_pc_prompt[:, 2:3, :].fill_(1)
        if self.learnt_p and self.prompt_type == 'attention':  # 若为attention-level（注意力级别）prompt
            missing_pc_prompt[:, prompt_length // 2 + 2: prompt_length // 2 + 3, :].fill_(1)
        self.missing_pc_prompt = nn.Parameter(missing_pc_prompt)  # 创建可训练的参数

        # if not self.learnt_p: # 如果不需要模态缺失感知提示，则冻结这些参数
        #     self.complete_prompt.requires_grad=False
        #     self.missing_img_prompt.requires_grad=False
        #     self.missing_pc_prompt.requires_grad = False

        # print("complete_prompt = ", self.complete_prompt)
        # print("missing_img_prompt = ", self.missing_img_prompt)
        # print("missing_pc_prompt = ", self.missing_pc_prompt)

        # 冻结原来的rgb_backbone、xyz_backbone的参数，不训练
        # for param in self.rgb_backbone.parameters():
        #     param.requires_grad = False
        # for param in self.xyz_backbone.parameters():
        #     param.requires_grad = False
        # for param in self.token_type_embeddings.parameters():
        #     param.requires_grad=False

    # 前向传递rgb图像特征
    def forward_rgb_features(self, x):
        # print("x.shape = ", x.shape)
        x = self.rgb_backbone.patch_embed(x) # 图像块embedding
        x = self.rgb_backbone._pos_embed(x) # 位置embedding
        x = self.rgb_backbone.norm_pre(x) # 标准化

        if self.rgb_backbone.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.rgb_backbone.blocks(x)

        x = self.rgb_backbone.norm(x)

        feat = x[:,1:].permute(0, 2, 1).view(1, -1, 28, 28)
        return feat

    # 前向传递rgb图像特征
    # def forward_rgb_features(self, rgb):
    #     (image_embeds, image_masks, patch_index, image_labels) = self.rgb_backbone.visual_embed(rgb, max_image_len=-1, mask_it=False)
    #     x = image_embeds
    #
    #     for i, blk in enumerate(self.rgb_backbone.blocks):
    #         if i in self.prompt_layers:
    #             if self.multi_layer_prompt:
    #                 x, _attn = blk(x, mask=image_masks)
    #             else:
    #                 x, _attn = blk(x, mask=image_masks)
    #         else:
    #             x, _attn = blk(x, mask=image_masks)
    #
    #     x = self.rgb_backbone.norm(x)  # 标准化
    #     # print("x.shape = ", x.shape)
    #
    #     # feat = x[:, 2:].permute(0, 2, 1).view(1, -1, 12, 12)
    #     # feat = x[:, 1:].permute(0, 2, 1).view(1, -1, 7, 7)
    #     feat = x[:,1:].permute(0, 2, 1).view(1, -1, 28, 28)
    #     return feat

    # 前向传递（原来版本）
    # def forward(self, rgb, xyz):
    #
    #     rgb_features = self.forward_rgb_features(rgb) # rgb图像特征
    #
    #     xyz_features, center, ori_idx, center_idx = self.xyz_backbone(xyz) # 点云xyz特征
    #
    #     return rgb_features, xyz_features, center, ori_idx, center_idx


    # 前向传递（原来版本）
    # def forward(self, rgb, xyz, missing_type=None):
        # print("rgb.shape = ", rgb.shape)
        # rgb_features = self.forward_rgb_features(rgb)  # rgb图像特征

        # xyz_features, center, ori_idx, center_idx = self.xyz_backbone(xyz)  # 点云xyz特征

        # return rgb_features, xyz_features, center, ori_idx, center_idx

    # 前向传递（prompt版本）
    def forward(self, rgb, xyz, missing_type=None):
        (image_embeds, image_masks, patch_index, image_labels) = self.rgb_backbone.visual_embed(rgb, max_image_len=-1, mask_it=False)

        image_embeds = image_embeds + self.token_type_embeddings(torch.full_like(image_masks, 1))

        # instance wise missing aware prompts
        # 实例模态缺失感知提示
        prompts = None  # 存储数据集需要的所有prompt
        for idx in range(len(rgb)):
            if missing_type == 0:  # 根据模态缺失的类型确定prompt
                prompt = self.complete_prompt
            elif missing_type == 1:
                prompt = self.missing_pc_prompt
            elif missing_type == 2:
                prompt = self.missing_img_prompt

            if prompt.size(0) != 1:
                prompt = prompt.unsqueeze(0)

            if prompts is None:
                prompts = prompt
            else:
                prompts = torch.cat([prompts, prompt], dim=0)

        if self.learnt_p:  # 如果需要模态缺失感知提示
            if self.prompt_type == 'attention':  # attention-level（注意力级别）prompt
                prompt_masks = torch.ones(prompts.shape[0], self.prompt_length // 2, dtype=prompts.dtype,
                                          device=prompts.device).long()  # 获得prompt掩码，//执行整数除法
            elif self.prompt_type == 'input':  # input-level（输入级别）prompt
                prompt_masks = torch.ones(prompts.shape[0], self.prompt_length * len(self.prompt_layers),
                                          dtype=prompts.dtype, device=prompts.device).long()  # 获得prompt掩码
        else:  # 如果不需要模态缺失感知提示
            prompt_masks = torch.ones(prompts.shape[0], self.prompt_length, dtype=prompts.dtype,
                                      device=prompts.device).long()  # 获得prompt掩码

        prompt_masks=prompt_masks.to(self.device)
        co_masks = torch.cat([prompt_masks, image_masks], dim=1) # 总的掩码
        co_embeds = torch.cat([image_embeds], dim=1)  # 总的embedding
        x = co_embeds.detach()

        for i, blk in enumerate(self.rgb_backbone.blocks):
            if i in self.prompt_layers:
                if self.multi_layer_prompt:
                    x, _attn = blk(x, mask=co_masks,
                                   prompts=prompts[:,self.prompt_layers.index(i)],
                                   learnt_p=self.learnt_p,
                                   prompt_type=self.prompt_type)
                else:
                    x, _attn = blk(x, mask=co_masks, prompts=prompts, learnt_p=self.learnt_p)
            else:
                x, _attn = blk(x, mask=co_masks)

        x = self.rgb_backbone.norm(x) # 标准化
        # print("x.shape = ", x.shape)

        # 确定总的prompt的长度
        if self.prompt_type == 'input':  # input-level（输入级别）prompt
            total_prompt_len = len(self.prompt_layers) * prompts.shape[-2]
        elif self.prompt_type == 'attention':  # attention-level（注意力级别）prompt
            total_prompt_len = prompts.shape[-2]

        # print("total_prompt_len = ", total_prompt_len)

        # rgb_features = x[:, 1:].permute(0, 2, 1).view(1, -1, 28, 28) # rgb图像特征，来自Vision Transformer
        # rgb_features = x[:, 2:].permute(0, 2, 1).view(1, -1, 12, 12) # rgb图像特征，来自Vision Transformer
        rgb_features = x[:, total_prompt_len + 1:].permute(0, 2, 1).view(1, -1, 28, 28) # rgb图像特征，来自Vision Transformer

        xyz_features, center, ori_idx, center_idx = self.xyz_backbone(xyz)# 点云xyz特征，来自Point Transformer

        return rgb_features, xyz_features, center, ori_idx, center_idx


# 进行最远点采样fps（farthest point sampling），能够保证对样本的均匀采样，适合用于3D点云、3D目标检测
def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
    return fps_data, fps_idx


class Group(nn.Module):
    # 初始化
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)
    # 前向传递
    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center, center_idx = fps(xyz.contiguous(), self.num_group)  # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        ori_idx = idx
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.reshape(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.reshape(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center, ori_idx, center_idx

# 编码器类
class Encoder(nn.Module):
    # 初始化
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )
    # 前向传递
    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)

# 多层感知机类
class Mlp(nn.Module):
    # 初始化
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    # 前向传递
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# 注意力层类
class Attention(nn.Module):
    # 初始化
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    # 前向传递（原来版本）
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# 块
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # self.attn = Attention(
        #     dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    # 前向传递（原来版本）
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# 没有层次结构的Transformer编码器
class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """
    # 初始化
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    # 前向传递
    def forward(self, x, pos):
        feature_list = []
        fetch_idx = [3, 7, 11]
        for i, block in enumerate(self.blocks):
            x = block(x + pos)
            if i in fetch_idx:
                feature_list.append(x)
        return feature_list

# Point Transformer，用于提取点云特征
class PointTransformer(nn.Module):
    # 初始化
    def __init__(self, group_size=128, num_group=1024, encoder_dims=384):
        super().__init__()

        self.trans_dim = 384
        self.depth = 12
        self.drop_path_rate = 0.1
        self.num_heads = 6

        self.group_size = group_size
        self.num_group = num_group
        # grouper
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        # define the encoder
        self.encoder_dims = encoder_dims
        if self.encoder_dims != self.trans_dim:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
            self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
            self.reduce_dim = nn.Linear(self.encoder_dims,  self.trans_dim)
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        # bridge encoder and transformer

        # 位置embedding模型结构
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128), # 线性层
            nn.GELU(), # GELU激活函数
            nn.Linear(128, self.trans_dim) # 线性层
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads
        )

        self.norm = nn.LayerNorm(self.trans_dim)

    # 载入训练的checkpoint权重
    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            #if incompatible.missing_keys:
            #    print('missing_keys')
            #    print(
            #            incompatible.missing_keys
            #        )
            #if incompatible.unexpected_keys:
            #    print('unexpected_keys')
            #    print(
            #            incompatible.unexpected_keys

            #        )

            # print(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}')

    def load_model_from_pb_ckpt(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path)
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        for k in list(base_ckpt.keys()):
            if k.startswith('transformer_q') and not k.startswith('transformer_q.cls_head'):
                base_ckpt[k[len('transformer_q.'):]] = base_ckpt[k]
            elif k.startswith('base_model'):
                base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
            del base_ckpt[k]

        incompatible = self.load_state_dict(base_ckpt, strict=False)

        if incompatible.missing_keys:
            print('missing_keys')
            print(
                    incompatible.missing_keys
                )
        if incompatible.unexpected_keys:
            print('unexpected_keys')
            print(
                    incompatible.unexpected_keys

                )
                
        print(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}')

    # 前向传递
    def forward(self, pts):
        if self.encoder_dims != self.trans_dim:
            B, C, N = pts.shape
            pts = pts.transpose(-1, -2) # B N 3

            # divide the point cloud in the same form. This is important 以相同的形式划分点云
            neighborhood,  center, ori_idx, center_idx = self.group_divider(pts)
            # # generate mask
            # bool_masked_pos = self._mask_center(center, no_mask = False) # B G

            # encoder the input cloud blocks 对输入的点云块进行编码
            group_input_tokens = self.encoder(neighborhood)  #  B G N
            group_input_tokens = self.reduce_dim(group_input_tokens)

            # prepare cls 类别准备
            cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)  
            cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

            # add pos embedding 添加位置embedding
            pos = self.pos_embed(center)

            # final input 最终输入
            x = torch.cat((cls_tokens, group_input_tokens), dim=1)
            pos = torch.cat((cls_pos, pos), dim=1)

            # transformer
            feature_list = self.blocks(x, pos)
            feature_list = [self.norm(x)[:,1:].transpose(-1, -2).contiguous() for x in feature_list]
            xyz_features = torch.cat((feature_list[0], feature_list[1], feature_list[2]), dim=1) # 1152 x即提取到的点云xyz特征

            return xyz_features, center, ori_idx, center_idx

        else:
            B, C, N = pts.shape
            pts = pts.transpose(-1, -2)  # B N 3

            # divide the point cloud in the same form. This is important 以相同的形式划分点云
            neighborhood, center, ori_idx, center_idx = self.group_divider(pts)

            group_input_tokens = self.encoder(neighborhood)  # B G N

            # add pos embedding 添加位置embedding
            pos = self.pos_embed(center)

            # final input 最终输入
            x = group_input_tokens

            # transformer
            feature_list = self.blocks(x, pos)
            feature_list = [self.norm(x).transpose(-1, -2).contiguous() for x in feature_list]
            xyz_features = torch.cat((feature_list[0], feature_list[1], feature_list[2]), dim=1) # 1152 x即提取到的点云xyz特征

            return xyz_features, center, ori_idx, center_idx
        