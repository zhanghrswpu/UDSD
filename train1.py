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

save_dir = "./cifar"
#save_dir = "D:\\Python\\pycharm\\adv_code\\paper2\\My-lab\\paper2\\Stage2-contrastrival_learning\\CIFAR10\\diff_model\\CheckpointsCondition"

class DDIMSampler(torch.nn.Module):
    def __init__(self, model, betas, alphas_bar, w=0.):
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
        device = x_T.device
        seq = torch.linspace(0, self.T - 1, sample_steps, device=device).round().long()
        x = x_T
        for i in reversed(range(sample_steps)):
            t = torch.full((x.shape[0],), seq[i], dtype=torch.long, device=device)
            t_prev= torch.full((x.shape[0],), seq[i-1] if i>0 else 0,
                               dtype=torch.long, device=device)

            a_t    = extract(self.alphas_bar, t,     x.shape)
            a_prev = extract(self.alphas_bar, t_prev,x.shape)

            x0_pred, e = self.p_mean(x, t, labels)
            sigma = eta * torch.sqrt((1 - a_t / a_prev) * (1 - a_prev) / (1 - a_t))
            noise = torch.randn_like(x) * sigma if i>0 else 0

            x = torch.sqrt(a_prev) * x0_pred \
                + torch.sqrt(1 - a_prev - sigma**2) * e \
                + noise

        return torch.clamp(x, -1, 1)




def train(args):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_network = vgg13_bn(pretrained=True).to(DEVICE)
    target_network.eval()

    train_dataset = loaddataset.PreDataset(root='./datasets',train=True,transform=config.test_transform,download=True)
    train_data = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=128,drop_last=True)

    model = net.SimCLRStage1().to(DEVICE)
    lossLR = net.Loss_v2().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    unet = UNet(T=500, num_labels=10, ch=128, ch_mult=[1, 2, 2, 2], num_res_blocks=2, dropout=0.15) \
        .to(DEVICE)
    ckpt = torch.load(os.path.join(save_dir, "ckpt_99_.pt"), map_location=DEVICE)
    unet.load_state_dict(ckpt)
    unet.eval()

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

        B = labels.shape[0]
        xT = torch.randn((B, 3, 32, 32), device=DEVICE)
        with torch.no_grad():
            imgs = ddim(xT, labels, sample_steps=sample_steps, eta=eta)

        imgs = imgs.mul(0.5).add(0.5).clamp_(0, 1)
        transforms_imgs = []
        for i in range(B):
            img = imgs[i]
            transforms_img = contrast_transform(img)
            transforms_imgs.append(transforms_img)
        #imgs = contrast_transform(imgs)
        transforms_imgs = torch.stack(transforms_imgs, dim=0)
        return transforms_imgs

    os.makedirs(config.save_path, exist_ok=True)
    for epoch in range(1, args.max_epoch + 1):
        model.train()
        total_loss = 0

        for batch, (imgs, labels) in enumerate(train_data):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)


            with torch.no_grad():
                output = target_network(imgs)
                predicted_class = torch.argmax(output, dim=1)
                predicted_class = predicted_class + 1 
                generated_images_batch = generated_image_method_batch(predicted_class)
            generated_images_batch = generated_images_batch.to(DEVICE) 

            pred = predicted_class - 1
            correct_mask = (pred == labels).to(DEVICE)
            if correct_mask.sum() == 0:
                print("all mistake")
                continue
            imgs = imgs[correct_mask]
            generated_images_batch = generated_images_batch[correct_mask]

            _, pre_L = model(imgs)
            _, pre_R = model(generated_images_batch) 
            loss = lossLR(pre_L, pre_R, args.batch_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            batch_loss = loss.detach().item()
            print(f"Epoch [{epoch}/{args.max_epoch}] Batch [{batch}/{len(train_data)}] Loss: {batch_loss:.4f}")
            total_loss += batch_loss


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
