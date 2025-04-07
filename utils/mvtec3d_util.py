import tifffile as tiff
import torch

# 把有组织的点云转化为无组织的点云
def organized_pc_to_unorganized_pc(organized_pc):
    return organized_pc.reshape(organized_pc.shape[0] * organized_pc.shape[1], organized_pc.shape[2])


# 读取tiff文件，原本是有组织的点云数据
def read_tiff_organized_pc(path):
    tiff_img = tiff.imread(path)
    return tiff_img

# 调整有组织的点云的大小
def resize_organized_pc(organized_pc, target_height=224, target_width=224, tensor_out=True):
    torch_organized_pc = torch.tensor(organized_pc).permute(2, 0, 1).unsqueeze(dim=0).contiguous()
    torch_resized_organized_pc = torch.nn.functional.interpolate(torch_organized_pc, size=(target_height, target_width),
                                                                 mode='nearest')
    if tensor_out:
        return torch_resized_organized_pc.squeeze(dim=0).contiguous()
    else:
        return torch_resized_organized_pc.squeeze().permute(1, 2, 0).contiguous().numpy()

# 有组织的点云转化为深度特征图像
def organized_pc_to_depth_map(organized_pc):
    return organized_pc[:, :, 2]
