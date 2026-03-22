import torch, argparse, os
import net, config, loaddataset
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from collections import defaultdict
import numpy
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import joblib
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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



def compute_cosine_similarity(features1, features2):
    """Compute cosine similarity between two sets of features"""
    features1 = F.normalize(features1, p=2, dim=1)
    features2 = F.normalize(features2, p=2, dim=1)
    return torch.sum(features1 * features2, dim=1).cpu().numpy()


def compute_cosine_similarity_v2(features1, features2):
    """Compute cosine similarity between two sets of features
    Args:
        features1: np.ndarray or torch.Tensor of shape (batch_size, feature_dim)
        features2: np.ndarray or torch.Tensor of shape (batch_size, feature_dim)
    Returns:
        np.ndarray of shape (batch_size,) containing cosine similarities
    """
    # 统一转换为PyTorch张量（支持GPU）
    if isinstance(features1, numpy.ndarray):
        features1 = torch.from_numpy(features1).float().to(DEVICE)
    if isinstance(features2, numpy.ndarray):
        features2 = torch.from_numpy(features2).float().to(DEVICE)

    # 确保数据类型一致
    features1 = features1.float()
    features2 = features2.float()

    # 归一化并计算相似度
    features1 = F.normalize(features1, p=2, dim=1)
    features2 = F.normalize(features2, p=2, dim=1)

    # 返回NumPy数组（与scikit-learn生态兼容）
    return torch.sum(features1 * features2, dim=1).cpu().numpy()


def train(args):
    save_dir = r'D:\Python\pycharm\adv_code\paper2\My-lab\paper2\Stage2-contrastrival_learning\CIFAR10\diff_model\CheckpointsCondition'

    #先弄出原始图像和重构图像来
    target_network = vgg13_bn(pretrained=True).to(DEVICE)
    target_network.eval()
    
    # 蒸馏方法能够生成批量的重构图像
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
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    #生成图像的函数
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

        # 归一化到 [0,1]
        #imgs = imgs * 0.5 + 0.5
        imgs = imgs.mul(0.5).add(0.5).clamp_(0, 1)

        # 对生成的图像也做同样的归一化操作
        imgs = contrast_transform(imgs)

        return imgs   


    train_dataset = loaddataset.PreDataset(
        root='D:\\Python\\pycharm\\adv_code\\paper2\\My-lab\\paper2\\dataset\\cifar10',
        train=True,
        transform=config.test_transform,
        download=True
    )
    # 减少训练数据 每个类别选2000张 实际训练是全部的训练样本
    labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    num_samples_per_class = 4000
    selected_indices = []
    for label in label_to_indices:
        indices = label_to_indices[label]
        numpy.random.shuffle(indices)
        selected_indices.extend(indices[:num_samples_per_class])

    subset_dataset = torch.utils.data.Subset(train_dataset, selected_indices)

    train_data = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,  # 建议增加workers提升加载速度
        pin_memory=True,
        drop_last=True
    )
    #加载编码器
    model = net.SimCLRStage2().to(DEVICE)
    model.load_state_dict(torch.load("D:\\Python\\pycharm\\adv_code\\paper2\\My-lab\\paper2\\Stage2-contrastrival_learning\\CIFAR10\\save_path\\model\\cifar_model_stage1_apart_epoch280.pth"), strict=False)
    model.eval()







    from sklearn.decomposition import PCA
    temp_features = []

    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(train_data):
            image = image.to(DEVICE)
            feature = model(image).cpu().numpy()  # 提取特征并转为numpy
            temp_features.append(feature)

        # 合并特征并拟合PCA
    temp_features = numpy.vstack(temp_features)  # shape: (n_samples, feature_dim)
    pca = PCA(n_components=2)  # 保留前100个主成分（或按比例，如n_components=0.95）
    pca.fit(temp_features)






    all_feature_cos = []
    for batch_idx, (image, label) in enumerate(train_data):
        image, label = image.to(DEVICE), label.to(DEVICE)

        with torch.no_grad():
            output = target_network(image.to(DEVICE))
            predicted_class = torch.argmax(output, dim=1)
            predicted_class = predicted_class+1
            generated_images_batch = generated_image_method_batch(predicted_class)

        generated_images_batch = generated_images_batch.to(DEVICE)

        feature_normal_v1 = model(image)
        feature_reconstructed_v1 = model(generated_images_batch)
        #image = normal_image(image)
        feature_normal = model(image).cpu().numpy()
        feature_reconstructed = model(generated_images_batch).cpu().numpy()

        feature_normal_pca = pca.transform(feature_normal)
        feature_reconstructed_pca = pca.transform(feature_reconstructed)

        batch_cos = compute_cosine_similarity_v2(feature_normal_pca, feature_reconstructed_pca)
        all_feature_cos.extend(batch_cos)

        column_means = numpy.mean(batch_cos, axis=0)
        print(f"Processed batch {batch_idx + 1}/{len(train_data)}, avg cosine sim: {column_means}")


    features = numpy.vstack(all_feature_cos)  # 形状: (10000, 2)

# 使用孤立森林进行训练
    #iso_forest = IsolationForest(n_estimators=1000, contamination=0.07, random_state=42)
    #iso_forest.fit(features)、
    from sklearn.neighbors import KernelDensity
    kde = KernelDensity(
        bandwidth=0.2,  # 核密度带宽，可优化
        kernel='gaussian'  # 常用核函数
    )

    kde.fit(features)
    train_scores = kde.score_samples(features)

    # 4) 基本统计信息（调试用）
    print("\n===== KDE 训练调试信息 =====")
    print(f"特征数量: {len(train_scores)}")
    print(f"score 均值: {train_scores.mean():.4f}")
    print(f"score 方差: {train_scores.std():.4f}")
    print(f"1% 分位: {numpy.percentile(train_scores, 1):.4f}")
    print(f"5% 分位: {numpy.percentile(train_scores, 5):.4f}")
    print(f"10% 分位: {numpy.percentile(train_scores, 10):.4f}")

    threshold = numpy.percentile(train_scores, 14)
    os.makedirs(args.save_path, exist_ok=True)


    # Save models
    os.makedirs(args.save_path, exist_ok=True)
    joblib.dump(kde, os.path.join(args.save_path, 'D:\\Python\\pycharm\\adv_code\\paper2\\My-lab\\paper2\\Stage2-contrastrival_learning\\CIFAR10\\save_path\\model\\kde_model.pkl'))
    joblib.dump(threshold, os.path.join(args.save_path, 'D:\\Python\\pycharm\\adv_code\\paper2\\My-lab\\paper2\\Stage2-contrastrival_learning\\CIFAR10\\save_path\\model\\kde_threshold.pkl'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train IsolationForest')
    parser.add_argument('--batch_size', default=400, type=int, help='Batch size')
    parser.add_argument('--save_path', default = 'D:\\Python\\pycharm\\adv_code\\paper2\\My-lab\\paper2\\Stage2-contrastrival_learning\\CIFAR10\\save_path\\model', type=str, help='path to save model')
    args = parser.parse_args()
    train(args)



