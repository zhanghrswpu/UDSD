import torch, argparse, os
import net, config, loaddataset
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from collections import defaultdict
import numpy
from vgg import vgg13_bn
from ModelCondition import UNet
from DiffusionCondition import GaussianDiffusionSampler
import os
from DiffusionCondition import extract
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader, SubsetRandomSampler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载扩散模型的预训练权重 根据标签生成图像
save_dir = "./cifar"
#save_dir = "D:\\Python\\pycharm\\adv_code\\paper2\\My-lab\\paper2\\Stage2-contrastrival_learning\\CIFAR10\\diff_model\\CheckpointsCondition"

class DDIMSampler(torch.nn.Module):
    def __init__(self, model, betas, alphas_bar, w=0.):
        """
        model:       你的 UNet
        betas:       Tensor, shape=(T,) 线性 beta 序列
        alphas_bar:  Tensor, shape=(T,) alpha_bar 累积量
        w:           guidance weight
        """
        super().__init__()
        self.model = model
        self.betas = betas
        self.alphas_bar = alphas_bar
        self.T = betas.shape[0]
        self.w = w

    def p_mean(self, x_t, t, labels):
        # classifier-free guidance
        eps = self.model(x_t, t, labels)
        eps_uncond = self.model(x_t, t, torch.zeros_like(labels))
        e = (1 + self.w) * eps - self.w * eps_uncond
        alpha_bar_t = extract(self.alphas_bar, t, x_t.shape)
        x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * e) / torch.sqrt(alpha_bar_t)
        return x0_pred, e

    def forward(self, x_T, labels, sample_steps=20, eta=0.):
        """
        x_T:           (B,3,32,32) 标准正态
        sample_steps:  实际步数 S
        eta:           控制随机量，0→确定性，1→与原DDPM等价
        """
        device = x_T.device
        # 1) 构造等间隔 timesteps
        seq = torch.linspace(0, self.T - 1, sample_steps, device=device).round().long()
        x = x_T
        for i in reversed(range(sample_steps)):
            t = torch.full((x.shape[0],), seq[i], dtype=torch.long, device=device)
            t_prev= torch.full((x.shape[0],), seq[i-1] if i>0 else 0,
                               dtype=torch.long, device=device)

            a_t    = extract(self.alphas_bar, t,     x.shape)
            a_prev = extract(self.alphas_bar, t_prev,x.shape)

            # 预测 x0 和 noise
            x0_pred, e = self.p_mean(x, t, labels)

            # 计算 sigma
            sigma = eta * torch.sqrt((1 - a_t / a_prev) * (1 - a_prev) / (1 - a_t))
            noise = torch.randn_like(x) * sigma if i>0 else 0

            # DDIM 更新公式
            x = torch.sqrt(a_prev) * x0_pred \
                + torch.sqrt(1 - a_prev - sigma**2) * e \
                + noise

        return torch.clamp(x, -1, 1)




def train(args):
    # 设备设置
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载目标神经网络
    target_network = vgg13_bn(pretrained=True).to(DEVICE)
    target_network.eval()

    # 数据加载 只应用于预测图像的标签
    train_dataset = loaddataset.PreDataset(root='./datasets',train=True,transform=config.test_transform,download=True)
    train_data = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=128,drop_last=True)

    # 初始化模型
    model = net.SimCLRStage1().to(DEVICE)
    lossLR = net.Loss_v2().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    #加载生成模型
    unet = UNet(T=500, num_labels=10, ch=128, ch_mult=[1, 2, 2, 2], num_res_blocks=2, dropout=0.15) \
        .to(DEVICE)
    ckpt = torch.load(os.path.join(save_dir, "ckpt_99_.pt"), map_location=DEVICE)
    unet.load_state_dict(ckpt)
    unet.eval()

    # 准备 DDIMSampler 蒸馏
    betas = torch.linspace(1e-4, 0.028, 500, device=DEVICE).double()
    alphas = 1 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    ddim = DDIMSampler(unet, betas, alphas_bar, w=1.8).to(DEVICE)

    contrast_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),  # 保留至少 80% 内容
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    def generated_image_method_batch(labels, sample_steps=20, eta=0.0):
        """
        labels:        LongTensor, shape (B,)
        sample_steps:  int, 生成时的步数
        eta:           float, DDIM 中的随机量系数（0 完全确定性）
        save_grid:     bool, 是否保存网格图
        """
        B = labels.shape[0]
        xT = torch.randn((B, 3, 32, 32), device=DEVICE)
        with torch.no_grad():
            imgs = ddim(xT, labels, sample_steps=sample_steps, eta=eta)

        # 反归一化到 [0,1]
        imgs = imgs.mul(0.5).add(0.5).clamp_(0, 1)
        transforms_imgs = []
        for i in range(B):
            img = imgs[i]
            transforms_img = contrast_transform(img)
            transforms_imgs.append(transforms_img)
        #对生成的图像也做同样的归一化操作
        #imgs = contrast_transform(imgs)
        transforms_imgs = torch.stack(transforms_imgs, dim=0)
        return transforms_imgs

    # 训练循环
    os.makedirs(config.save_path, exist_ok=True)
    for epoch in range(1, args.max_epoch + 1):
        model.train()
        total_loss = 0

        for batch, (imgs, labels) in enumerate(train_data):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)


            with torch.no_grad():
                output = target_network(imgs)
                predicted_class = torch.argmax(output, dim=1)
                predicted_class = predicted_class + 1  # 预测标签1-10
                generated_images_batch = generated_image_method_batch(predicted_class)
            generated_images_batch = generated_images_batch.to(DEVICE) #没有做数据增强 只是将这个图像也归一化到了正常的范围里

            # 筛选只有预测正确的样本参与训练 提高训练的精确度
            pred = predicted_class - 1
            correct_mask = (pred == labels).to(DEVICE)
            if correct_mask.sum() == 0:
                print("all mistake")
                continue
            imgs = imgs[correct_mask]
            generated_images_batch = generated_images_batch[correct_mask]

            # 前向传播
            _, pre_L = model(imgs)
            _, pre_R = model(generated_images_batch)  # 使用重构图像作为正样本对

            # 计算损失
            loss = lossLR(pre_L, pre_R, args.batch_size)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 日志打印
            batch_loss = loss.detach().item()
            print(f"Epoch [{epoch}/{args.max_epoch}] Batch [{batch}/{len(train_data)}] Loss: {batch_loss:.4f}")
            total_loss += batch_loss

        # 保存模型
        if epoch % 10 == 0:
            save_path = os.path.join(config.save_path, f'cifar_model_stage1_apart1_epoch{epoch}.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at {save_path}")

        print(f"Epoch [{epoch}/{args.max_epoch}] Avg Loss: {total_loss / len(train_data):.4f}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--batch_size', default=512, type=int, help='Batch size')
    parser.add_argument('--max_epoch', default=800, type=int, help='Maximum epochs')
    args = parser.parse_args()
    train(args)