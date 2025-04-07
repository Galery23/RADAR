import os
import glob
import numpy as np
import random

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.mvtec3d_util import *
# 8.14

# eyecandies数据集的类别
def eyecandies_classes():
    return [
        'CandyCane',
        'ChocolateCookie',
        'ChocolatePraline',
        'Confetto',
        'GummyBear',
        'HazelnutTruffle',
        'LicoriceSandwich',
        'Lollipop',
        'Marshmallow',
        'PeppermintCandy',   
    ]

# mvtec3d数据集的类别
def mvtec3d_classes():
    return [
        "bagel",
        "cable_gland",
        "carrot",
        "cookie",
        "dowel",
        "foam",
        "peach",
        "potato",
        "rope",
        "tire",
    ]

RGB_SIZE = 224

# 基本工业异常检测数据集
class BaseAnomalyDetectionDataset(Dataset):
    def __init__(self, split, class_name, img_size, dataset_path='datasets/eyecandies_preprocessed'):
        self.IMAGENET_MEAN = [0.485, 0.456, 0.406]
        self.IMAGENET_STD = [0.229, 0.224, 0.225]
        self.cls = class_name
        self.size = img_size
        self.img_path = os.path.join(dataset_path, self.cls, split) # 图片文件地址
        self.rgb_transform = transforms.Compose(
            [transforms.Resize((RGB_SIZE, RGB_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
             transforms.ToTensor(),
             transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)])


# 预训练tensor数据集
class PreTrainTensorDataset(Dataset):
    def __init__(self, root_path):
        super().__init__()
        self.root_path = root_path
        self.tensor_paths = os.listdir(self.root_path)

    def __len__(self):
        return len(self.tensor_paths)

    def __getitem__(self, idx):
        tensor_path = self.tensor_paths[idx]

        tensor = torch.load(os.path.join(self.root_path, tensor_path))

        label = 0

        return tensor, label

# 训练数据集
class TrainDataset(BaseAnomalyDetectionDataset):
    # 初始化
    def __init__(self, class_name, img_size, dataset_path='datasets/eyecandies_preprocessed'):
        super().__init__(split="train", class_name=class_name, img_size=img_size, dataset_path=dataset_path)

        self.img_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

        # ================= 模态丢失数据预处理 ====================== #
        self.simulate_missing = False
        self.missing_type = 0

        missing_ratio = 0.3 # 模态丢失率
        mratio = str(missing_ratio).replace('.', '')
        both_ratio = 0.5 # 两个模态都有所丢失时的丢失比分配率（即各丢失丢失率的一般，如70%丢失率则各丢失35%）

        missing_type = 'both' # 模态丢失类型

        missing_table_root = './datasets/missing_tables/' # 模态丢失表位置
        missing_table_name = f'mvtec_3d_missing_train_{class_name}_{missing_type}_{mratio}.pt' # 模态丢失表名称
        missing_table_path = os.path.join(missing_table_root, missing_table_name) # 模态丢失表实际存储位置

        # use image data to formulate missing table ，使用图像数据制定模态丢失表的总数
        total_num = len(self.img_paths)
        # print("total_num = ", total_num)

        if os.path.exists(missing_table_path):  # 如果模态丢失表已经存在
            missing_table = torch.load(missing_table_path)  # 直接读取模态丢失表
            if len(missing_table) != total_num:
                print('missing table mismatched!')
                exit()
        else:  # 如果模态丢失表不存在
            missing_table = torch.zeros(total_num)

            if missing_ratio > 0:  # 存在模态丢失率
                missing_index = random.sample(range(total_num), int(total_num * missing_ratio))  # 随机取设定丢失的样本

                if missing_type == 'pc':  # 如果只有点云丢失
                    missing_table[missing_index] = 1
                elif missing_type == 'image':  # 如果只有RGB图像丢失
                    missing_table[missing_index] = 2
                elif missing_type == 'both':  # 如果点云和RGB图像都丢失
                    missing_table[missing_index] = 1
                    missing_index_image = random.sample(missing_index, int(len(missing_index) * both_ratio))  # 随机取设定丢失的样本
                    missing_table[missing_index_image] = 2

                # print("missing_table_path = ", missing_table_path)
                # print("missing_table = ", missing_table)
                torch.save(missing_table, missing_table_path)  # 保存模态丢失表

        self.missing_table = missing_table
        # ================= 模态丢失数据预处理 ====================== #

    # 读取数据集
    def load_dataset(self):
        img_tot_paths = [] # 图片文件地址
        tot_labels = []

        rgb_paths = glob.glob(os.path.join(self.img_path, 'good', 'rgb') + "/*.png") # rgb文件地址
        tiff_paths = glob.glob(os.path.join(self.img_path, 'good', 'xyz') + "/*.tiff") # tiff文件地址
        rgb_paths.sort()
        tiff_paths.sort()

        sample_paths = list(zip(rgb_paths, tiff_paths)) # 数据样本地址
        img_tot_paths.extend(sample_paths) # 图像数据地址，列表类型
        tot_labels.extend([0] * len(sample_paths)) # 标签数据地址，列表类型

        return img_tot_paths, tot_labels

    # 数据集长度
    def __len__(self):
        return len(self.img_paths)

    # 获取对象
    # def __getitem__(self, idx):
    #     img_path, label = self.img_paths[idx], self.labels[idx]
    #     rgb_path = img_path[0]
    #     tiff_path = img_path[1]
    #     img = Image.open(rgb_path).convert('RGB')
    #
    #     img = self.rgb_transform(img)
    #     organized_pc = read_tiff_organized_pc(tiff_path)
    #
    #     depth_map_3channel = np.repeat(organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2)
    #     resized_depth_map_3channel = resize_organized_pc(depth_map_3channel)
    #     resized_organized_pc = resize_organized_pc(organized_pc, target_height=self.size, target_width=self.size)
    #     resized_organized_pc = resized_organized_pc.clone().detach().float()
    #
    #     return (img, resized_organized_pc, resized_depth_map_3channel), label

    # 获取对象
    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx] # 读取图像地址和标签
        rgb_path = img_path[0] # rgb图像地址
        tiff_path = img_path[1] # tiff图像地址
        # print("img_path = ", img_path)
        # print("label = ", label)

        img = Image.open(rgb_path).convert('RGB') # 转为图像
        img_tensor = self.rgb_transform(img) # 转为tensor的形式

        # 训练模态完整的数据时，随机分配缺失类型的样本来模拟模态丢失
        simulate_missing_type = 0
        if self.simulate_missing and self.missing_table[idx] == 0:
            simulate_missing_type = random.choice([0, 1, 2])

        # print("img_tensor = ", img_tensor)
        # print("img_tensor.shape[0] = ", img_tensor.shape[0])
        # 训练图像模态丢失的数据时
        if self.missing_table[idx] == 2 or simulate_missing_type == 2:
            for idx in range(len(img_tensor)):
                img_tensor[idx] = torch.ones(img_tensor[idx].size()).float() # torch.ones是张量创建函数，创建一个指定形状的张量，并将其所有元素初始化为1

        organized_pc = read_tiff_organized_pc(tiff_path)  # 直接读取tiff文件，作为组织好的点云数据

        # 训练点云模态丢失的数据时
        if self.missing_table[idx] == 1 or simulate_missing_type == 1:
            organized_pc = np.where(organized_pc >= 0, 1.0, 1.0) # 读到的tiff图像化成的矩阵都置为1
            depth_map_3channel = np.repeat(organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2)  # 通过np.repeat形成矩阵
            resized_depth_map_3channel = resize_organized_pc(depth_map_3channel)
            resized_organized_pc = resize_organized_pc(organized_pc, target_height=self.size, target_width=self.size)
            resized_organized_pc = resized_organized_pc.clone().detach().float()
        else:
            depth_map_3channel = np.repeat(organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2)  # 通过np.repeat形成矩阵
            resized_depth_map_3channel = resize_organized_pc(depth_map_3channel)
            resized_organized_pc = resize_organized_pc(organized_pc, target_height=self.size, target_width=self.size)
            resized_organized_pc = resized_organized_pc.clone().detach().float()

        self.missing_type = self.missing_table[idx].item() + simulate_missing_type

        return (img_tensor, resized_organized_pc, resized_depth_map_3channel), label, self.missing_type

# 测试数据集
class TestDataset(BaseAnomalyDetectionDataset):
    # 初始化
    def __init__(self, class_name, img_size, dataset_path='datasets/eyecandies_preprocessed'):
        super().__init__(split="test", class_name=class_name, img_size=img_size, dataset_path=dataset_path)
        self.gt_transform = transforms.Compose([
            transforms.Resize((RGB_SIZE, RGB_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()])
        self.img_paths, self.gt_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

        # ================= 模态丢失数据预处理 ====================== #
        self.simulate_missing = False
        self.missing_type = 0

        missing_ratio = 0.3  # 模态丢失率
        mratio = str(missing_ratio).replace('.', '')
        both_ratio = 0.5 # 两个模态都有所丢失时的丢失比分配率（即各丢失丢失率的一般，如70%丢失率则各丢失35%）

        missing_type = 'both'  # 模态丢失类型

        missing_table_root = './datasets/missing_tables/'  # 模态丢失表位置
        missing_table_name = f'mvtec_3d_missing_test_{class_name}_{missing_type}_{mratio}.pt'  # 模态丢失表名称
        missing_table_path = os.path.join(missing_table_root, missing_table_name)  # 模态丢失表实际存储位置

        # use image data to formulate missing table ，使用图像数据制定模态丢失表的总数
        total_num = len(self.img_paths)
        # print("total_num = ", total_num)

        if os.path.exists(missing_table_path):  # 如果模态丢失表已经存在
            missing_table = torch.load(missing_table_path)  # 直接读取模态丢失表
            if len(missing_table) != total_num:
                print('missing table mismatched!')
                exit()
        else:  # 如果模态丢失表不存在
            missing_table = torch.zeros(total_num)

            if missing_ratio > 0:  # 存在模态丢失率
                missing_index = random.sample(range(total_num), int(total_num * missing_ratio))  # 随机取设定丢失的样本

                if missing_type == 'pc':  # 如果只有点云丢失
                    missing_table[missing_index] = 1
                elif missing_type == 'image':  # 如果只有RGB图像丢失
                    missing_table[missing_index] = 2
                elif missing_type == 'both':  # 如果点云和RGB图像都丢失
                    missing_table[missing_index] = 1
                    missing_index_image = random.sample(missing_index,
                                                        int(len(missing_index) * both_ratio))  # 随机取设定丢失的样本
                    missing_table[missing_index_image] = 2

                # print("missing_table_path = ", missing_table_path)
                # print("missing_table = ", missing_table)
                torch.save(missing_table, missing_table_path)  # 保存模态丢失表

        self.missing_table = missing_table
        # ================= 模态丢失数据预处理 ====================== #

    # 读取数据
    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                rgb_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb') + "/*.png")
                tiff_paths = glob.glob(os.path.join(self.img_path, defect_type, 'xyz') + "/*.tiff")
                rgb_paths.sort()
                tiff_paths.sort()
                sample_paths = list(zip(rgb_paths, tiff_paths))
                img_tot_paths.extend(sample_paths)
                gt_tot_paths.extend([0] * len(sample_paths))
                tot_labels.extend([0] * len(sample_paths))
            else:
                rgb_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb') + "/*.png")
                tiff_paths = glob.glob(os.path.join(self.img_path, defect_type, 'xyz') + "/*.tiff")
                gt_paths = glob.glob(os.path.join(self.img_path, defect_type, 'gt') + "/*.png")
                rgb_paths.sort()
                tiff_paths.sort()
                gt_paths.sort()
                sample_paths = list(zip(rgb_paths, tiff_paths))

                img_tot_paths.extend(sample_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(sample_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels

    # 数据集长度
    def __len__(self):
        return len(self.img_paths)

    # 获取对象
    def __getitem__(self, idx):
        img_path, gt, label = self.img_paths[idx], self.gt_paths[idx], self.labels[idx]
        rgb_path = img_path[0]
        tiff_path = img_path[1]

        img_original = Image.open(rgb_path).convert('RGB')
        img_tensor = self.rgb_transform(img_original)  # 转为tensor的形式

        # 训练模态完整的数据时，随机分配缺失类型的样本来模拟模态丢失
        simulate_missing_type = 0
        if self.simulate_missing and self.missing_table[idx] == 0:
            simulate_missing_type = random.choice([0, 1, 2])

        # print("img_tensor = ", img_tensor)
        # print("img_tensor.shape[0] = ", img_tensor.shape[0])
        # 训练图像模态丢失的数据时
        if self.missing_table[idx] == 2 or simulate_missing_type == 2:
            for idx in range(len(img_tensor)):
                img_tensor[idx] = torch.ones(img_tensor[idx].size()).float()  # torch.ones是张量创建函数，创建一个指定形状的张量，并将其所有元素初始化为1

        organized_pc = read_tiff_organized_pc(tiff_path)  # 直接读取tiff文件，作为组织好的点云数据

        # 训练点云模态丢失的数据时
        if self.missing_table[idx] == 1 or simulate_missing_type == 1:
            organized_pc = np.where(organized_pc >= 0, 1.0, 1.0) # 读到的tiff图像化成的矩阵都置为1
            depth_map_3channel = np.repeat(organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2)  # 通过np.repeat形成矩阵
            resized_depth_map_3channel = resize_organized_pc(depth_map_3channel)
            resized_organized_pc = resize_organized_pc(organized_pc, target_height=self.size, target_width=self.size)
            resized_organized_pc = resized_organized_pc.clone().detach().float()
        else:
            depth_map_3channel = np.repeat(organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2)  # 通过np.repeat形成矩阵
            resized_depth_map_3channel = resize_organized_pc(depth_map_3channel)
            resized_organized_pc = resize_organized_pc(organized_pc, target_height=self.size, target_width=self.size)
            resized_organized_pc = resized_organized_pc.clone().detach().float()

        self.missing_type = self.missing_table[idx].item() + simulate_missing_type

        if gt == 0:
            gt = torch.zeros(
                [1, resized_depth_map_3channel.size()[-2], resized_depth_map_3channel.size()[-2]])
        else:
            gt = Image.open(gt).convert('L')
            gt = self.gt_transform(gt)
            gt = torch.where(gt > 0.5, 1., .0)

        return (img_tensor, resized_organized_pc, resized_depth_map_3channel), gt[:1], label, rgb_path, self.missing_type

# 数据读取器
def get_data_loader(split, class_name, img_size, args):
    if split in ['train']:
        dataset = TrainDataset(class_name=class_name, img_size=img_size, dataset_path=args.dataset_path)
    elif split in ['test']:
        dataset = TestDataset(class_name=class_name, img_size=img_size, dataset_path=args.dataset_path)

    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False,
                             pin_memory=True)
    return data_loader
