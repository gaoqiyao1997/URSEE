import torch
import os
import dataloader as module_data
import model.NAF_LSTM as module_arch
from parse_config import ConfigParser
import argparse
from PIL import Image
import numpy as np
from collections import OrderedDict


def load_data(file_path, sequence_length):
    return module_data.Custom_DataLoader(
        data_file=file_path,
        sequence_length=sequence_length,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )


def save_images(predictions, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for idx, img_array in enumerate(predictions):
        # 创建子文件夹
        folder_path = os.path.join(output_dir, f"prediction_{idx}")
        os.makedirs(folder_path, exist_ok=True)

        for i, img in enumerate(img_array):
            # 移除不必要的维度
            img = img.squeeze()

            # 裁剪值并转换为uint8
            img = np.clip(img, 0, 1) * 255
            img = img.astype(np.uint8)

            # 判断是否为灰度或RGB图像
            if img.ndim == 2:
                mode = 'L'  # 灰度
            elif img.ndim == 3:
                mode = 'RGB'  # RGB

            # 创建图像
            pil_image = Image.fromarray(img, mode=mode)
            pil_image.save(os.path.join(folder_path, f"{i+92*idx+21}.png"))


def predict(model, data_loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for sequence in data_loader:
            # model.reset_states()  # 使用模块内的模型方法，这里假定模型是 DDP 的子模块
            seq_predictions = []
            for i in range(len(sequence)):  # 修改这里
                events, _ = sequence[i]['combined'].to(device), sequence[i]['frame'].to(device)  # 修改这里
                output = model(events)
                processed = output['image'].cpu().numpy()
                seq_predictions.append(processed)
            predictions.append(seq_predictions)
    return predictions


def main(config):
    logger = config.get_logger('predict')

    # 设置设备
    device_id = config['device']
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id
    device = torch.device('cuda:0' if torch.cuda.is_available() and device_id != 'cpu' else 'cpu')

    # 初始化模型
    model = config.init_obj('arch', module_arch)

    model = model.to(device)

    # 加载checkpoint
    checkpoint_path = str(config.resume or config['resume'])
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 处理state_dict的key，去除'module.'前缀
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        name = k[7:] if k.startswith('module.') else k  # 去掉前缀 "module."
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)

    # 加载测试数据
    test_data_loader = load_data(config['valid_data_loader']['args']['data_file'], config['sequence_length']['valid'])

    # 执行预测
    predictions = predict(model, test_data_loader, device)

    # 保存图像
    output_dir = 'output'
    save_images(predictions, output_dir)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Prediction')
    args.add_argument('-c', '--config', default=None, type=str, help='Config file path')
    args.add_argument('-d', '--device', default=None, type=str, help='GPU ID to use')
    args.add_argument('-r', '--resume', default=None, type=str, help='Path to resume checkpoint')
    config = ConfigParser.from_args(args)
    main(config)