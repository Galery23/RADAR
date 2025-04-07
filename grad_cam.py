import cv2
import numpy as np
import torch
import timm

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
# 8.14

# 加载预训练的 ViT 模型
# checkpoint_path = './checkpoints/dino_vitbase8_pretrain.pth'
# out_indices=None
# kwargs = {'features_only': True if out_indices else False}
# if out_indices:
#     kwargs.update({'out_indices': out_indices})
# model = timm.create_model(model_name='vit_base_patch8_224_dino', pretrained=True, checkpoint_path=checkpoint_path,
#                                           **kwargs)
model = timm.create_model(model_name='vit_base_patch8_224_dino')
model.load_state_dict(torch.load('checkpoints/dino_vitbase8_pretrain.pth'))
model.eval()

# 判断是否使用 GPU 加速
use_cuda = torch.cuda.is_available()
if use_cuda:
    model = model.cuda()

def reshape_transform(tensor, height=28, width=28):
    # 去掉cls token
    result = tensor[:, 1:, :].reshape(tensor.size(0),
    height, width, tensor.size(2))

    # 将通道维度放到第一个位置
    result = result.transpose(2, 3).transpose(1, 2)
    return result

# 创建 GradCAM 对象
cam = GradCAM(model=model,
            target_layers=[model.blocks[-1].norm1],
            # 这里的target_layer要看模型情况，
            # 比如还有可能是：target_layers = [model.blocks[-1].ffn.norm]
            reshape_transform=reshape_transform)

# 读取输入图像
image_path = "grad_cam_test/005.png"
rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
rgb_img = cv2.resize(rgb_img, (224, 224))

# 预处理图像
input_tensor = preprocess_image(rgb_img,
mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225])

# 看情况将图像转换为批量形式
# input_tensor = input_tensor.unsqueeze(0)
if use_cuda:
    input_tensor = input_tensor.cuda()

# 计算 grad-cam
target_category = None # 可以指定一个类别，或者使用 None 表示最高概率的类别
grayscale_cam = cam(input_tensor=input_tensor, targets=target_category)
grayscale_cam = grayscale_cam[0, :]

# 将 grad-cam 的输出叠加到原始图像上
# visualization = show_cam_on_image(rgb_img, grayscale_cam)
visualization = show_cam_on_image(np.array(rgb_img) / 255., grayscale_cam)

# 保存可视化结果
# cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR, visualization)
cv2.imwrite('cam_output.png', visualization)