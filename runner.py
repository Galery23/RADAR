import torch
import torch.nn as nn
from tqdm import tqdm
import os

from feature_extractors import multiple_features
from dataset import get_data_loader
# 8.14

class M3DM():
    # 初始化
    def __init__(self, args):
    # def __init__(self, args, class_name, type):
        self.args = args # 参数
        self.image_size = args.img_size # 图像大小
        self.count = args.max_sample # 数量

        # 不同的预训练方法（模型）名称
        if args.method_name == 'DINO':
            self.methods = {
                "DINO": multiple_features.RGBFeatures(args), # 提取RGB图像特征
            }
        elif args.method_name == 'Point_MAE':
            self.methods = {
                "Point_MAE": multiple_features.PointFeatures(args), # 提取点云特征
            }
        elif args.method_name == 'Fusion':
            self.methods = {
                "Fusion": multiple_features.FusionFeatures(args), # 融合特征
            }
        elif args.method_name == 'DINO+Point_MAE':
            self.methods = {
                "DINO+Point_MAE": multiple_features.DoubleRGBPointFeatures(args), # 提取RGB图像特征+点云特征
            }
        elif args.method_name == 'DINO+Point_MAE+add':
            self.methods = {
                "DINO+Point_MAE": multiple_features.DoubleRGBPointFeatures_add(args), # 提取RGB图像特征+点云特征+add方法
            }
        elif args.method_name == 'DINO+Point_MAE+Fusion':
            self.methods = {
                "DINO+Point_MAE+Fusion": multiple_features.TripleFeatures(args), # 提取RGB图像特征+点云特征+融合特征
            }

        # ===================== Prompt ===================== #

        # self.prompt_type = 'input'  # prompt类型
        # self.prompt_type = 'attention'  # prompt类型

        # prompt_length = 16  # prompt长度
        # self.prompt_length = prompt_length
        #
        # embed_dim = 768  # embedding维度，和隐藏层大小相同
        #
        # self.learnt_p = True # 是否需要模态感知提示，默认值为True
        #
        # self.prompt_layers = [0, 1, 2, 3, 4, 5] # 需要进行确实感知提示的层数，默认值为[0,1,2,3,4,5]
        # self.multi_layer_prompt = True # 是否需要多层prompt，默认值为True
        # prompt_num = len(self.prompt_layers) if self.multi_layer_prompt else 1  # prompt数量

        # 完整prompt（表示没有模态缺失时）
        # complete_prompt = torch.zeros(prompt_num, prompt_length, embed_dim)  # 完整prompt，初始为input-level（输入级别）prompt
        # complete_prompt[:, 0:1, :].fill_(1)
        # if self.learnt_p and self.prompt_type == 'attention':  # 若为attention-level（注意力级别）prompt
        #     complete_prompt[:, prompt_length // 2 + 0: prompt_length // 2 + 1, :].fill_(1)
        # self.complete_prompt = nn.Parameter(complete_prompt)  # 创建可训练的参数

        # 图像模态缺失prompt
        # missing_img_prompt = torch.zeros(prompt_num, prompt_length, embed_dim)  # 图像模态缺失的prompt
        # missing_img_prompt[:, 1:2, :].fill_(1)
        # if self.learnt_p and self.prompt_type == 'attention':  # 若为attention-level（注意力级别）prompt
        #     missing_img_prompt[:, prompt_length // 2 + 1: prompt_length // 2 + 2, :].fill_(1)
        # self.missing_img_prompt = nn.Parameter(missing_img_prompt)  # 创建可训练的参数

        # 点云模态缺失prompt
        # missing_pc_prompt = torch.zeros(prompt_num, prompt_length, embed_dim)  # 文本模态缺失的prompt
        # missing_pc_prompt[:, 2:3, :].fill_(1)
        # if self.learnt_p and self.prompt_type == 'attention':  # 若为attention-level（注意力级别）prompt
        #     missing_pc_prompt[:, prompt_length // 2 + 2: prompt_length // 2 + 3, :].fill_(1)
        # self.missing_pc_prompt = nn.Parameter(missing_pc_prompt)  # 创建可训练的参数
        #
        # if not self.learnt_p: # 如果不需要模态缺失感知提示，则冻结这些参数
        #     self.complete_prompt.requires_grad=False
        #     self.missing_img_prompt.requires_grad=False
        #     self.missing_pc_prompt.requires_grad = False

        # print("complete_prompt = ", self.complete_prompt)
        # print("missing_img_prompt = ", self.missing_img_prompt)
        # print("missing_pc_prompt = ", self.missing_pc_prompt)


    # 训练
    def fit(self, class_name):
        # 模型训练加载器，读取数据
        train_loader = get_data_loader("train", class_name=class_name, img_size=self.image_size, args=self.args)

        flag = 0
        for sample, label, missing_type in tqdm(train_loader, desc=f'Extracting train features for class {class_name}'):
            missing_type = int(missing_type.item())
            for method in self.methods.values():
                # print("m3dm_runner!! missing_type = ", missing_type)
                if self.args.save_feature: # 如果需要存储提取到的特征
                    method.add_sample_to_mem_bank(sample, missing_type, class_name=class_name) # 将样本添加到存储库中
                else:
                    method.add_sample_to_mem_bank(sample, missing_type) # 将样本添加到存储库中
                flag += 1
            if flag > self.count:
                flag = 0
                break

        # 运行核心集
        for method_name, method in self.methods.items():
            print(f'\n\nRunning coreset for {method_name} on class {class_name}...')
            method.run_coreset()

        if self.args.memory_bank == 'multiple': # 若使用多种方法混合
            flag = 0
            for sample, label, missing_type in tqdm(train_loader, desc=f'Running late fusion for {method_name} on class {class_name}..'):
                missing_type = int(missing_type.item())
                # print("22 m3dm_runner!! missing_type = ", missing_type)
                for method_name, method in self.methods.items():
                    method.add_sample_to_late_fusion_mem_bank(sample, missing_type)
                    flag += 1
                if flag > self.count:
                    flag = 0
                    break

            # 训练决策层融合（Dicision Layer Fusion，DLF）
            for method_name, method in self.methods.items():
                print(f'\n\nTraining Dicision Layer Fusion for {method_name} on class {class_name}...')
                method.run_late_fusion()


    # 评估
    def evaluate(self, class_name):
        # 三个评估指标
        image_rocaucs = dict()
        pixel_rocaucs = dict()
        au_pros = dict()

        # 模型评估加载器，读取数据
        test_loader = get_data_loader("test", class_name=class_name, img_size=self.image_size, args=self.args)

        path_list = []
        with torch.no_grad():
            for sample, mask, label, rgb_path, missing_type in tqdm(test_loader, desc=f'Extracting test features for class {class_name}'):
                missing_type = int(missing_type.item())
                for method in self.methods.values():
                    method.predict(sample, mask, label, missing_type) # 预测值
                    path_list.append(rgb_path)

        for method_name, method in self.methods.items():
            method.calculate_metrics() # 计算评估指标
            image_rocaucs[method_name] = round(method.image_rocauc, 3) # 图像级别准确率（I-AUROC）
            pixel_rocaucs[method_name] = round(method.pixel_rocauc, 3) # 像素级别准确率（P-AUROC）
            au_pros[method_name] = round(method.au_pro, 3) # per-region overlap（AUPRO）
            print(
                f'Class: {class_name}, {method_name} Image ROCAUC: {method.image_rocauc:.3f}, {method_name} Pixel ROCAUC: {method.pixel_rocauc:.3f}, {method_name} AU-PRO: {method.au_pro:.3f}')
            if self.args.save_preds:
                method.save_prediction_maps('./pred_maps', path_list)

        return image_rocaucs, pixel_rocaucs, au_pros
