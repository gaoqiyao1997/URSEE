import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import numpy as np
import os
import numpy as np
# import cv2
from pathlib import Path

from skimage import io
from torchvision.datasets import ImageFolder

import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from PIL import Image
import glob
import cv2



class DNDataset(Dataset):
    def __init__(self, input_folder, target_folder,transform=None):
        self.input_folder = input_folder
        self.target_folder = target_folder
        self.transform = transform
        self.input_images = glob.glob(f"{input_folder}/*.png")
        self.target_images = glob.glob(f"{target_folder}/*.png")
        a = 123


    def __getitem__(self, index):
        input_image_path = self.input_images[index]
        target_image_path = self.target_images[index]
        input_image = io.imread(input_image_path)
        target_image = io.imread(target_image_path)
        # input_image_float = input_image.astype(np.float32)
        # target_image_float = target_image.astype(np.float32)
        # input_image1_float =input_image1_float*255
        # input_image2_float =input_image2_float*255
        # target_image1_float =target_image1_float*255
        # target_image2_float =target_image2_float*255
        # try:
        #     input_image = Image.open(input_image_path)
        # except Exception as e:
        #     print(f"Error loading image: {input_image_path}")
        #     # 跳过无法读取的图像
        #     return None
        #
        # try:
        #     target_image = Image.open(target_image_path)
        # except Exception as e:
        #     print(f"Error loading image: {target_image_path}")
        #     # 跳过无法读取的图像
        #     return None
        if self.transform:
            input_tensor = self.transform(input_image)
            target_tensor = self.transform(target_image)

        #return input_image1_float, input_image2_float, target_image1_float, target_image2_float
        return input_tensor, target_tensor

    def __len__(self):
        return len(self.input_images)









class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)



class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class SoftmaxLayer(nn.Module):
    def __init__(self):
        super(SoftmaxLayer, self).__init__()
        
        # self.softmax=F.softmax(dim=1)
    def forward(self,feature, x):
        P=F.softmax(feature,dim=1)
        p0, p1 = torch.split(P, split_size_or_sections=128, dim=1)
        x0, x1 = torch.split(x, split_size_or_sections=128, dim=1)
       
        #print(P.shape)
        # p0=P[:, 0, :, :]
        # p1=P[:, 1, :, :]
        # x0=x[:, 0, :, :]
        # x1=x[:, 1, :, :]
        x0_temp=x0*p0+x1*(1-p0)
        x1_temp=x1*p1+x0*(1-p1)
        #print(x0_temp.shape)
        x = torch.cat((x0_temp, x1_temp), dim=1)
        # x=torch.cat([x0_temp.unsqueeze(1),x1_temp.unsqueeze(1)],dim=1)
        # print(x.shape)
        return x
    
class modelXJH(nn.Module):
    def __init__(self):
        super(modelXJH, self).__init__()
        self.Block1 = Block(dim=2)
        self.softmaxL=SoftmaxLayer()
        self.conv1=DoubleConv(in_channels=256, out_channels=256)
        self.conv2=nn.Conv2d(2, 256, kernel_size=3, padding=0 , stride=8)
        self.convTranspose2d=nn.ConvTranspose2d(in_channels=256, out_channels=2, kernel_size=8, stride=8, padding=0, output_padding=0)
    def forward(self, x):
        for i in range(20):
            #(32,2,256,256)
            x=self.conv2(x)#(32,256,32,32)
            #print(x.shape)          
            feature1=self.conv1(x)#(32,256,32,32)
            x=self.softmaxL(feature1,x)#(32,256,32,32)
            x=self.convTranspose2d(x)#(32,2,256,256)
            #print(x.shape)
            # short = x
            x=self.Block1(x)
            # x=short+x
        return x


class TemporalCNN(nn.Module):
    def __init__(self):
        super(TemporalCNN, self).__init__()

        # 输入通道数为2（两张图片），输出通道数为64的卷积层
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        # ReLU激活函数
        self.relu = nn.ReLU(inplace=True)
        # 输出通道数为1的卷积层，用于生成去噪后的图像
        self.conv2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.Block = Block(1)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.relu(x)
        # x = self.conv2(x)
        x = self.Block(x)
        x = self.Block(x)
        x = self.Block(x)
        x = self.Block(x)
        x = self.Block(x)
        return x


# 定义自编码器模型
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# 定义测试函数
def test(model, train_loader, device, len):
    model.eval()
    patches1 = []
    patches2 = []
    patches3 = []
    # 在测试数据上进行推理
    with torch.no_grad():
        ssim_ave_out=0
        ssim_ave_in=0
        psnr_ave_in=0
        psnr_ave_out=0
        ENL_input_ave=0
        ENL_output_ave=0
        ENL_target_ave=0
        num=0
        for inputs, targets in train_loader:
            inputs_pic=inputs
            inputs = inputs.to(device)
            outputs = model(inputs)
            # print(inputs.shape)
            # input1 = inputs_pic[0:1, :, :]
            # input2 = inputs_pic[1:2, :, :]
            #print(outputs.shape)
            outputs=np.squeeze(outputs)
            targets=np.squeeze(targets)
            inputs_pic=np.squeeze(inputs_pic)
            #print(outputs.shape)
            output1 = outputs[0, :, :]
            output2 = outputs[1, :, :]
            output3 = outputs[2, :, :]

            #print(targets.shape)
            target1 = targets[0, :, :]
            target2 = targets[1, :, :]
            target3 = targets[2, :, :]

            input1 = inputs_pic[0, :, :]
            input2 = inputs_pic[1, :, :]
            input3 = inputs_pic[2, :, :]

            #print(target1.shape,target2.shape)
            #print(output1.shape)
            #print(output2.shape)
            # 图片导出的时候没办法直接导出。张量在CUDA设备上，需要使用.cpu()方法将其复制到主机内存（CPU）上，然后再将其转换为NumPy数组。
            output1 = output1.cpu()
            output2 = output2.cpu()
            output3 = output3.cpu()

            output1_array = output1.numpy()
            output2_array = output2.numpy()
            output3_array = output3.numpy()

            input1_array = input1.numpy()
            input2_array = input2.numpy()
            input3_array = input3.numpy()

            target1_array = target1.numpy()
            target2_array = target2.numpy()
            target3_array = target3.numpy()
            #print(target1_array.shape,output1_array.shape)
            output_img1=np.squeeze(output1_array)
            output_img2=np.squeeze(output2_array)
            output_img3=np.squeeze(output3_array)

            input_img1=np.squeeze(input1_array)
            input_img2=np.squeeze(input2_array)
            input_img3=np.squeeze(input3_array)

            target_img1=np.squeeze(target1_array)
            target_img2=np.squeeze(target2_array)
            target_img3=np.squeeze(target3_array)
            #print(target_img1.shape,target_img2.shape,output_img1.shape,output_img2.shape)
            patches1.append(output_img1)
            patches2.append(output_img2)
            patches3.append(output_img3)
            
            # # 检查图片的数据类型
            # print(f"Data type: {input_img1.dtype}")
            # # 检查像素值范围
            # print(f"Pixel value range: {input_img1.min()} to {input_img1.max()}")


            # print("-----------")
            ssim1_out=SSIM(target_img1,output_img1)
            ssim2_out=SSIM(target_img2,output_img2)
            ssim3_out=SSIM(target_img3,output_img3)

            max_ssim_out=max(ssim1_out,ssim2_out,ssim3_out)

            ssim1_in=SSIM(target_img1,input_img1)
            ssim2_in=SSIM(target_img2,input_img2)
            ssim3_in=SSIM(target_img3,input_img3)

            min_ssim_in=min(ssim1_in,ssim2_in,ssim3_in)


            ssim_ave_out+=max_ssim_out
            ssim_ave_in+=min_ssim_in

            psnr1_out=PSNR(target_img1,output_img1)
            psnr2_out=PSNR(target_img2,output_img2) 
            psnr3_out=PSNR(target_img3,output_img3)

            max_psnr_out=max(psnr1_out,psnr2_out,psnr3_out)

            psnr1_in=PSNR(target_img1,input_img1)
            psnr2_in=PSNR(target_img2,input_img2)
            psnr3_in=PSNR(target_img3,input_img3)

            min_psnr_in=min(psnr1_in,psnr2_in,psnr3_in)


            psnr_ave_out+=max_psnr_out
            psnr_ave_in+=min_psnr_in        

            ENL1_target=ENL(target_img1)
            ENL2_target=ENL(target_img2)
            ENL3_target=ENL(target_img3)
            ENL_target=(ENL1_target+ENL2_target+ENL3_target)/3
            ENL1_input=ENL(input_img1)
            ENL2_input=ENL(input_img2)
            ENL3_input=ENL(input_img3)
            ENL_input=(ENL1_input+ENL2_input+ENL3_input)/3
            ENL1_output=ENL(output_img1)
            ENL2_output=ENL(output_img2)
            ENL3_output=ENL(output_img3)
            ENL_output=(ENL1_output+ENL2_output+ENL3_output)/3

            ENL_input_ave+=ENL_input
            ENL_output_ave+=ENL_output
            ENL_target_ave+=ENL_target

            num=num+1
            if num%100==0:
                print("真值对输出的PSNR:%f"%max_psnr_out)
                print("真值对输入的PSNR:%f"%min_psnr_in)
                print("真值对输出的SSIM:%f"%max_ssim_out)
                print("真值对输入的SSIM:%f"%min_ssim_in)
                print("ENL均值： 输入：%f  输出：%f  真值：%f \n"%(ENL_input,ENL_output,ENL_target))
            #     io.imsave('/home/xjh/xjh-bishe/output1.tif', output_img1)
            #     io.imsave('/home/xjh/xjh-bishe/output2.tif', output_img2)
            #     io.imsave('/home/xjh/xjh-bishe/input1.tif',input_img1)
            #     io.imsave('/home/xjh/xjh-bishe/input2.tif',input_img2)
            #     io.imsave('/home/xjh/xjh-bishe/target1.tif',target_img1)
            #     io.imsave('/home/xjh/xjh-bishe/target2.tif',target_img2)
    return ssim_ave_out/len(),ssim_ave_in/len(),psnr_ave_out/len(),psnr_ave_in/len(),ENL_input_ave/len(),ENL_output_ave/len(),ENL_target_ave/len(),patches1
    # return 0








