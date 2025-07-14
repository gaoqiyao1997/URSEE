import torch
import numpy as np
import os
import glob
from skimage import io
import torchvision.transforms as transforms
from basicsr.models.archs.SRDNet_arch import *
from torch.utils.data import Dataset, DataLoader

class DNDataset(Dataset):
    def __init__(self, input_folder, target_folder, transform=None):
        self.input_folder = input_folder
        self.target_folder = target_folder
        self.transform = transform
        self.input_images = glob.glob(f"{input_folder}/*.png")
        self.target_images = glob.glob(f"{target_folder}/*.png")
        assert len(self.input_images) == len(self.target_images), "Mismatch between input and target files"

    def __getitem__(self, index):
        input_image_path = self.input_images[index]
        target_image_path = self.target_images[index]
        input_image = io.imread(input_image_path)
        target_image = io.imread(target_image_path)
        file_name = os.path.basename(input_image_path)  # 获取文件名

        if self.transform:
            input_tensor = self.transform(input_image)
            target_tensor = self.transform(target_image)
        else:
            input_tensor = input_image
            target_tensor = target_image

        return input_tensor, target_tensor, file_name

    def __len__(self):
        return len(self.input_images)

def test(model, input_folder, target_folder, output_folder, device, batch_size=1, transform=None):
    dataset = DNDataset(input_folder, target_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    with torch.no_grad():
        for inputs, targets, file_names in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs.cpu().numpy()
            outputs = outputs.squeeze()  # 去除批次维度如果batch_size=1

            # Save each image using the corresponding filename
            for i, file_name in enumerate(file_names):
                output_array_8bit = (np.clip(outputs[i], 0, 1) * 255).astype(np.uint8) if batch_size > 1 else (np.clip(outputs, 0, 1) * 255).astype(np.uint8)
                output_path = os.path.join(output_folder, file_name)
                io.imsave(output_path, output_array_8bit)
    return 0

# 示例代码
transform = transforms.Compose([transforms.ToTensor()])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = SRDNet().to(device)
model.load_state_dict(torch.load('model/SRD_module.pth'))

# 更新为新的文件夹路径
input_folder = 'test/input'
target_folder = 'test/target'
output_folder = 'test_output'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

test(model, input_folder, target_folder, output_folder, device, batch_size=1, transform=transform)
print("finished")