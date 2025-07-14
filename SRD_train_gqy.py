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
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from PIL import Image
import glob
import cv2
from SRD_dataset_gqy import *

from basicsr.models.archs.SRDNet_arch import *

# from osgeo import gdal, gdalconst
# print("import ok")


# 定义训练函数
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

       
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        # running_loss += loss.data.cpu().numpy()

    train_epoch_loss = train_loss /  len(train_loader.dataset)
    return train_epoch_loss




# 定义val函数
def val(model, val_loader, criterion, optimizer, device):
    model.eval()  # 设置模型为评估模式
    val_loss = 0.0
    with torch.no_grad():  # 关闭梯度计算
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()* inputs.size(0)

    val_epoch_loss = val_loss /  len(val_loader.dataset)
    val_losses.append(val_loss / len(val_loader))
    
    
    return val_epoch_loss







# os.environ['CUDA_VISIBLE_DEVICE']='1'
# torch.cuda.set_device(1)
# print(torch.cuda.is_available())
# 设置训练参数

input_folder='Dataset/train/input'
target_folder='Dataset/train/target'
val_input_folder='Dataset/test/input'
val_target_folder='Dataset/test/target'

train_losses = []  # 用于存储每个epoch的训练损失
val_losses = []  # 用于存储每个epoch的验证损失
batch_size = 4
num_epochs = 550
learning_rate = 0.0001
device = torch.device('cuda:0')#选择第一块GPU还是第二块GPU来训练

# 定义数据预处理和加载器
transform =transforms.Compose([
        #    transforms.RandomSized((256, 256)),
       # transforms.Normalize(mean=(0.485, 0.456), std=(0.229, 0.224)),
           transforms.ToTensor()])



#train_dataset = CustomDataset(input_folder1, input_folder2,input_folder3, target_folder1, target_folder2,target_folder3, transform=transform)
train_dataset = DNDataset(input_folder, target_folder, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

val_dataset = DNDataset(val_input_folder, val_target_folder,  transform=transform)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# 创建模型实例并将其移到设备上
# model = UNet_mod(n_channels=2, n_classes=2).to(device)
# model = UNet_mod(n_channels=2, n_classes=2).to(device)
# model = UNet(n_channels=3, n_classes=3).to(device)
model = SRDNet().to(device)
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate,betas=(0.9, 0.99))

# 进行训练
for epoch in range(num_epochs):

    train_epoch_loss = train(model, train_loader, criterion, optimizer, device)
    print(f'Epoch [{epoch+1}/{num_epochs}], train_Loss: {train_epoch_loss:.4f}')
    train_losses.append(train_epoch_loss)
    val_epoch_loss = train(model, val_loader, criterion, optimizer, device)
    print(f'Epoch [{epoch+1}/{num_epochs}], val_Loss: {val_epoch_loss:.4f}')
    val_losses.append(val_epoch_loss)

    if((epoch+1)%5==0):
        print("epoch:"+str(epoch+1)+"  model saving----------- ")
        torch.save(model.state_dict(), os.path.join('511_model_save/'+str(epoch+1)+'.pth'))
        print("model saved")



