import json
import shutil
from time import time
import config
import torch
import torch.nn.functional as F
import torchvision
from model.preact_resnet import  PreActResNet18
from model.resnet import ResNet18
from model.MNISTnet import MNISTnet
from network.models import Denormalizer
from torch import nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utils.dataloader import PostTensorTransform, get_dataloader
from utils.utils import progress_bar
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

def get_model(opt):
    net = None
    optimizer = None
    scheduler = None
    if opt.dataset == "cifar10" or opt.dataset == "gtsrb":
        net = PreActResNet18(num_classes=opt.num_classes).to(opt.device)
    if opt.dataset == "celeba":
        net = ResNet18().to(opt.device)
    if opt.dataset == "mnist":
        net = MNISTnet().to(opt.device)
    optimizer = torch.optim.SGD(net.parameters(), opt.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, opt.scheduler_milestones, opt.scheduler_lambda)
    return net, optimizer, scheduler

def train(net, optimizer, scheduler, train_dl, identity_grid, noise_grid, tf_writer, epoch, opt):
    net = nn.DataParallel(net)
    net.to(opt.device)
    print(" Train:")
    net.train()
    rate_bd = opt.ar
    total_loss = 0
    total_sample = 0
    total_clean = 0
    total_bd = 0
    total_clean_correct = 0
    total_bd_correct = 0
    criterion_CE = torch.nn.CrossEntropyLoss()
    denormalizer = Denormalizer(opt)
    transforms = PostTensorTransform(opt).to(opt.device)
    total_time = 0

    for batch_idx, (inputs, targets) in enumerate(train_dl):
        optimizer.zero_grad()
        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        bs = inputs.shape[0]
        # Create backdoor data
        num_bd = int(bs * rate_bd)
        grid_temps = (identity_grid + opt.s * noise_grid / (opt.input_height * opt.ratio)) * opt.grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)
        bbx1, bby1, bbx2, bby2 = saliency_bbox(inputs)
        inputs_bd = F.grid_sample(inputs[:num_bd, :, bbx1:bbx2, bby1:bby2], grid_temps.repeat(num_bd, 1, 1, 1), align_corners=True)
        if opt.attack_mode == "all2one":
            targets_bd = torch.ones_like(targets[:num_bd]) * opt.target_label
        if opt.attack_mode == "all2all":
            targets_bd = torch.remainder(targets[:num_bd] + 1, opt.num_classes)
        total_inputs = torch.cat([inputs_bd, inputs[num_bd :]], dim=0)
        total_inputs = transforms(total_inputs)
        total_targets = torch.cat([targets_bd, targets[num_bd:]], dim=0)
        start = time()
        total_preds = net(total_inputs)
        total_time += time()-start
        loss = criterion_CE(total_preds, total_targets)
        loss.backward()
        optimizer.step()
        total_sample += bs
        total_loss += loss.detach()
        total_clean += bs - num_bd
        total_bd += num_bd
        total_clean_correct += torch.sum(
            torch.argmax(total_preds[num_bd:], dim=1) == total_targets[num_bd:]
        )
        total_bd_correct += torch.sum(torch.argmax(total_preds[:num_bd], dim=1) == targets_bd)
        avg_acc_clean = total_clean_correct * 100.0 / total_clean
        avg_acc_bd = total_bd_correct * 100.0 / total_bd
        avg_loss = total_loss / total_sample
        progress_bar(batch_idx, len(train_dl), "CE Loss: {:.4f} | Clean Acc: {:.4f} | Bd Acc: {:.4f} ".format(avg_loss, avg_acc_clean, avg_acc_bd))
        # save backdoor image
        if not batch_idx % 50:
            if not os.path.exists(opt.temps):
                os.makedirs(opt.temps)
            path = os.path.join(opt.temps, "backdoor_image.png")
            torchvision.utils.save_image(inputs_bd, path, normalize=True)
        # tensorboard image
        if batch_idx == len(train_dl) - 2:
            residual = inputs_bd - inputs[:num_bd]
            batch_img = torch.cat([inputs[:num_bd], inputs_bd, total_inputs[:num_bd], residual], dim=2)
            batch_img = denormalizer(batch_img)
            batch_img = F.upsample(batch_img, scale_factor=(4, 4))
            grid = torchvision.utils.make_grid(batch_img, normalize=True)

    # tensorboard information
    if not epoch % 1:
        tf_writer.add_scalars("Train Accuracy", {"clean": avg_acc_clean, "bd": avg_acc_bd}, epoch)
        tf_writer.add_image("Images", grid, global_step=epoch)
    scheduler.step()


def eval(net, optimizer, scheduler, test_dl, identity_grid, noise_grid, best_clean_acc, best_bd_acc, tf_writer, epoch, opt):
    net = nn.DataParallel(net)
    net.to(opt.device)
    print(" Eval:")
    net.eval()
    total_sample = 0
    total_clean_correct = 0
    total_bd_correct = 0
    total_ae_loss = 0
    criterion_BCE = torch.nn.BCELoss()

    for batch_idx, (inputs, targets) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            bs = inputs.shape[0]
            total_sample += bs
            # Clean Evaluation
            preds_clean = net(inputs)
            total_clean_correct += torch.sum(torch.argmax(preds_clean, 1) == targets)
            # Backdoor Evaluation
            grid_temps = (identity_grid + opt.s * noise_grid / (opt.input_height * opt.ratio)) * opt.grid_rescale
            grid_temps = torch.clamp(grid_temps, -1, 1)
            bbx1, bby1, bbx2, bby2 = saliency_bbox(inputs)
            inputs_bd = F.grid_sample(inputs[:, :, bbx1:bbx2, bby1:bby2], grid_temps.repeat(bs, 1, 1, 1),
                                      align_corners=True)
            if opt.attack_mode == "all2one":
                targets_bd = torch.ones_like(targets) * opt.target_label
            if opt.attack_mode == "all2all":
                targets_bd = torch.remainder(targets + 1, opt.num_classes)
            preds_bd = net(inputs_bd)
            total_bd_correct += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)
            acc_clean = total_clean_correct * 100.0 / total_sample
            acc_bd = total_bd_correct * 100.0 / total_sample
            progress_bar(batch_idx, len(test_dl), "Clean Acc: {:.4f} - Best: {:.4f} | Bd Acc: {:.4f} - Best: {:.4f}".format(acc_clean, best_clean_acc, acc_bd, best_bd_acc))

    # tensorboard information
    if not epoch % 1:
        tf_writer.add_scalars("Test Accuracy", {"clean": acc_clean, "bd": acc_bd}, epoch)

    # checkpoint
    if acc_clean > best_clean_acc or (acc_clean > best_clean_acc - 0.1 and acc_bd > best_bd_acc):
        print(" Saving...")
        best_clean_acc = acc_clean
        best_bd_acc = acc_bd
        state_dict = {
            "net": net.state_dict(),
            "scheduler": scheduler.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_clean_acc": best_clean_acc,
            "best_bd_acc": best_bd_acc,
            "epoch_current": epoch,
            "identity_grid": identity_grid,
            "noise_grid": noise_grid,
        }
        torch.save(state_dict, opt.ckpt_path)
        with open(os.path.join(opt.ckpt_folder, "results.txt"), "w+") as f:
            results_dict = {
                "clean_acc": best_clean_acc.item(),
                "bd_acc": best_bd_acc.item(),
            }
            json.dump(results_dict, f, indent=2)
    return best_clean_acc, best_bd_acc

def saliency_bbox(img):
    opt = config.get_arguments().parse_args()
    size = img.size()
    W = size[1]
    H = size[2]
    ratio = opt.ratio
    cut_w = F.int(W * ratio)
    cut_h = F.int(H * ratio)

    # compute the image saliency map
    temp_img = img.cpu().numpy().transpose(1, 2, 0)
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(temp_img)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    maximum_indices = F.unravel_index(F.argmax(saliencyMap, axis=None), saliencyMap.shape)
    x = maximum_indices[0]
    y = maximum_indices[1]
    bbx1 = F.clip(x - cut_w // 2, 0, W)
    bby1 = F.clip(y - cut_h // 2, 0, H)
    bbx2 = F.clip(x + cut_w // 2, 0, W)
    bby2 = F.clip(y + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def main():
    opt = config.get_arguments().parse_args()
    if opt.dataset in ["mnist", "cifar10"]:
        opt.num_classes = 10
    elif opt.dataset == "gtsrb":
        opt.num_classes = 43
    elif opt.dataset == "celeba":
        opt.num_classes = 8
    else:
        raise Exception("Invalid Dataset")

    if opt.dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "gtsrb":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "mnist":
        opt.input_height = 28
        opt.input_width = 28
        opt.input_channel = 1
    elif opt.dataset == "celeba":
        opt.input_height = 64
        opt.input_width = 64
        opt.input_channel = 3
    else:
        raise Exception("Invalid Dataset")

    # dataset and model
    train_dl = get_dataloader(opt, True)
    test_dl = get_dataloader(opt, False)
    net, optimizer, scheduler = get_model(opt)

    # pretrained model
    mode = opt.attack_mode
    opt.ckpt_folder = os.path.join(opt.checkpoints, opt.dataset)
    opt.ckpt_path = os.path.join(opt.ckpt_folder, "{}_{}.pth.tar".format(opt.dataset, mode))
    opt.log_dir = os.path.join(opt.ckpt_folder, "log_dir")
    if not os.path.exists(opt.log_dir):
        os.makedirs(opt.log_dir)

    if opt.train_scratch:
        print("Train from scratch")
        best_clean_acc = 0.0
        best_bd_acc = 0.0
        epoch_current = 0

        # grid design
        ins = np.random.beta(1, 1, (1, 2, opt.k, opt.k)) * 2 - 1
        ins = ins / torch.mean(torch.abs(ins))
        noise_grid = (
            F.upsample(ins, size=opt.input_height * opt.ratio, mode="bicubic", align_corners=True)
                .permute(0, 2, 3, 1)
                .to(opt.device)
        )
        array1d = torch.linspace(-1, 1, steps=opt.input_height * opt.ratio)
        x, y = torch.meshgrid(array1d, array1d)
        identity_grid = torch.stack((y, x), dim=2)[None, ...].to(opt.device)
        shutil.rmtree(opt.ckpt_folder, ignore_errors=True)
        os.makedirs(opt.log_dir)
        with open(os.path.join(opt.ckpt_folder, "opt.json"), "w+") as f:
            json.dump(opt.__dict__, f, indent=2)
        tf_writer = SummaryWriter(log_dir=opt.log_dir)
    else:
        print("Train from pretrained model")
        state_dict = torch.load(opt.ckpt_path)
        net.load_state_dict(state_dict["net"])
        optimizer.load_state_dict(state_dict["optimizer"])
        scheduler.load_state_dict(state_dict["scheduler"])
        best_clean_acc = state_dict["best_clean_acc"]
        best_bd_acc = state_dict["best_bd_acc"]
        epoch_current = state_dict["epoch_current"]
        identity_grid = state_dict["identity_grid"]
        noise_grid = state_dict["noise_grid"]
        tf_writer = SummaryWriter(log_dir=opt.log_dir)

    for epoch in range(epoch_current, opt.epoches):
        print("Epoch {}:".format(epoch + 1))
        train(net, optimizer, scheduler, train_dl, noise_grid, identity_grid, tf_writer, epoch, opt)
        best_clean_acc, best_bd_acc = eval(net, optimizer, scheduler, test_dl, noise_grid, identity_grid, best_clean_acc, best_bd_acc, tf_writer, epoch, opt)

if __name__ == '__main__':
    main()


