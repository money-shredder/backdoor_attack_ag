import json
import os
import shutil
from time import time
import config
import numpy as np
import cv2
import copy
import torch
import torch.nn.functional as F
import torchvision

from model.resnet import ResNet18
from model.preact_resnet import PreActResNet18
from model.MNISTnet import MNISTnet
from network.models import Denormalizer
from torch.utils.tensorboard import SummaryWriter
from utils.dataloader import PostTensorTransform, get_dataloader
from utils.utils import progress_bar
import os




def get_model(opt):
    if opt.dataset == "cifar10" or opt.dataset == "gtsrb":
        net = PreActResNet18(num_classes=opt.num_classes).to(opt.device)
    if opt.dataset == "celeba":
        net = ResNet18().to(opt.device)
    if opt.dataset == "mnist":
        net = MNISTnet().to(opt.device)
    optimizer = torch.optim.SGD(net.parameters(), opt.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, opt.scheduler_milestones, opt.scheduler_lambda)
    return net, optimizer, scheduler


def saliency_bbox(img):
    opt = config.get_arguments().parse_args()
    size = img.size()
    W = size[1]
    H = size[2]
    ratio = opt.ratio
    cut_w = int(W // ratio)
    cut_h = int(H // ratio)
    if opt.dataset == "mnist":
        x = 14
        y = 14
        bbx1 = np.clip(x - cut_w // 2, 0, W)
        bby1 = np.clip(y - cut_h // 2, 0, H)
        bbx2 = np.clip(x + cut_w // 2, 0, W)
        bby2 = np.clip(y + cut_h // 2, 0, H)
        if (x - cut_w // 2) < 0:
            bbx1 = 0
            bbx2 = W // opt.ratio
        if (x + cut_w // 2) > W:
            bbx1 = W - (W // opt.ratio)
            bbx2 = W
        if (y - cut_h // 2) < 0:
            bby1 = 0
            bby2 = H // opt.ratio
        if (y + cut_h // 2) > H:
            bby1 = H - (H // opt.ratio)
            bby2 = H
    else:
        # compute the image saliency map
        temp_img = img.cpu().numpy().transpose(1, 2, 0)
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        (success, saliencyMap) = saliency.computeSaliency(temp_img)
        saliencyMap = (saliencyMap * 255).astype("uint8")
        maximum_indices = np.unravel_index(np.argmax(saliencyMap, axis=None), saliencyMap.shape)
        x = maximum_indices[0]
        y = maximum_indices[1]
        bbx1 = np.clip(x - cut_w // 2, 0, W)
        bby1 = np.clip(y - cut_h // 2, 0, H)
        bbx2 = np.clip(x + cut_w // 2, 0, W)
        bby2 = np.clip(y + cut_h // 2, 0, H)
        if (x - cut_w // 2) < 0:
            bbx1 = 0
            bbx2 = W // opt.ratio
        if (x + cut_w // 2) > W:
            bbx1 = W - (W // opt.ratio)
            bbx2 = W
        if (y - cut_h // 2) < 0:
            bby1 = 0
            bby2 = H // opt.ratio
        if (y + cut_h // 2) > H:
            bby1 = H - (H // opt.ratio)
            bby2 = H
    return bbx1, bby1, bbx2, bby2


def train(net, optimizer, scheduler, train_dl, noise_grid, identity_grid, tf_writer, epoch, opt):
    print(" Train:")
    device_ids = [0, 1, 2]
    net = torch.nn.DataParallel(net, device_ids=device_ids).cuda()
    net.train()
    rate_bd = opt.pc
    total_loss_ce = 0
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
        grid_temps = (identity_grid + opt.s * noise_grid / (opt.input_height // opt.ratio)) * opt.grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1).float()

        if opt.attack_choice == "dirty":
            num_bd = int(bs * rate_bd)
            input_origin = copy.deepcopy(inputs[:num_bd])
            inputs_bd = inputs[:num_bd]
            for id_img in range(num_bd):
                bbx1, bby1, bbx2, bby2 = saliency_bbox(inputs_bd[id_img])
                temp = inputs_bd[id_img:(id_img + 1), :, bbx1:bbx2, bby1:bby2]
                inputs_bd[id_img:(id_img + 1), :, bbx1:bbx2, bby1:bby2] = F.grid_sample(temp, grid_temps.repeat(1, 1, 1, 1),
                                                                                        align_corners=True)
            if opt.attack_mode == "all2one":
                targets_bd = torch.ones_like(targets[:num_bd]) * opt.target_label
            if opt.attack_mode == "all2all":
                targets_bd = torch.remainder(targets[:num_bd] + 1, opt.num_classes)
            total_inputs = torch.cat([inputs_bd, inputs[num_bd:]], dim=0)
            total_inputs = transforms(total_inputs)
            total_targets = torch.cat([targets_bd, targets[num_bd:]], dim=0)
            start = time()
            total_preds = net(total_inputs)
            total_time += time() - start
            loss_ce = criterion_CE(total_preds, total_targets)
            loss = loss_ce
            loss.backward()
            optimizer.step()

            total_sample += bs
            total_loss_ce += loss_ce.detach()
            total_clean += bs - num_bd
            total_bd += num_bd
            total_bd_correct += torch.sum(torch.argmax(total_preds[:num_bd], dim=1) == targets_bd)
            total_clean_correct += torch.sum(torch.argmax(total_preds[num_bd:], dim=1) == total_targets[num_bd:])
            avg_acc_clean = total_clean_correct * 100.0 / total_clean
            avg_acc_bd = total_bd_correct * 100.0 / total_bd
            avg_loss_ce = total_loss_ce / total_sample
            progress_bar(batch_idx, len(train_dl),"CE Loss: {:.4f} | Clean Acc: {:.4f} | Bd Acc: {:.4f} ".format(avg_loss_ce, avg_acc_clean, avg_acc_bd))

            # Image for tensorboard
            if batch_idx == len(train_dl) - 2:
                residual = inputs_bd - input_origin
                batch_img = torch.cat([input_origin, inputs_bd, total_inputs[:num_bd], residual], dim=2)
                batch_img = denormalizer(batch_img)
                batch_img = F.upsample(batch_img, scale_factor=(4, 4))
                grid = torchvision.utils.make_grid(batch_img, normalize=True)

        if opt.attack_choice == "clean":
            num_bd = 0
            input_origin = copy.deepcopy(inputs)
            for id_img in range(bs):
                if targets[id_img:(id_img + 1)] == opt.target_label:
                    inputs[id_img:(id_img + 1), :, :, :] = F.grid_sample(inputs[id_img:(id_img + 1), :, :, :],
                                                                                        grid_temps.repeat(1, 1, 1, 1),
                                                                                        align_corners=True)
                    num_bd += 1
            preds = net(inputs)
            loss_ce = criterion_CE(preds, targets)
            loss = loss_ce
            loss.backward()
            optimizer.step()

            total_loss_ce += loss_ce.detach()
            total_sample += bs
            total_clean_correct += torch.sum(torch.argmax(preds, dim=1) == targets)
            avg_acc_clean = total_clean_correct * 100.0 / total_sample
            avg_loss_ce = total_loss_ce / total_sample
            progress_bar(batch_idx, len(train_dl),
                         "CE Loss: {:.4f} | Clean Acc: {:.4f}".format(avg_loss_ce, avg_acc_clean))

            # Image for tensorboard
            if batch_idx == len(train_dl) - 2:
                residual = inputs[:num_bd] - input_origin[:num_bd]
                batch_img = torch.cat([input_origin[:num_bd], inputs[:num_bd], residual], dim=2)
                batch_img = denormalizer(batch_img)
                batch_img = F.upsample(batch_img, scale_factor=(4, 4))
                grid = torchvision.utils.make_grid(batch_img, normalize=True)

    # for tensorboard
    if not epoch % 1:
        if opt.attack_choice == "dirty":
            tf_writer.add_scalars("Clean Accuracy", {"Clean": avg_acc_clean, "Bd": avg_acc_bd}, epoch)
            tf_writer.add_image("Images", grid, global_step=epoch)
        else:
            tf_writer.add_scalars("Clean Accuracy", {"Clean": avg_acc_clean}, epoch)
            tf_writer.add_image("Images", grid, global_step=epoch)
    scheduler.step()


def eval(net, optimizer, scheduler, test_dl, noise_grid, identity_grid, best_clean_acc, best_bd_acc, tf_writer, epoch, opt):
    print(" Eval:")
    net.to(opt.device)
    net.eval()
    total_sample = 0
    total_clean_correct = 0
    total_bd_correct = 0
    for batch_idx, (inputs, targets) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            bs = inputs.shape[0]
            total_sample += bs

            # Evaluate Clean
            preds_clean = net(inputs)
            total_clean_correct += torch.sum(torch.argmax(preds_clean, 1) == targets)

            if opt.attack_choice == "dirty":
                grid_temps = (identity_grid + opt.s * noise_grid / (opt.input_height // opt.ratio)) * opt.grid_rescale
                grid_temps = torch.clamp(grid_temps, -1, 1).float()
                inputs_bd = inputs
                for idv_img in range(bs):
                    bbx1, bby1, bbx2, bby2 = saliency_bbox(inputs_bd[idv_img])
                    inputs_bd[idv_img:(idv_img + 1), :, bbx1:bbx2, bby1:bby2] = F.grid_sample(
                        inputs_bd[idv_img:(idv_img + 1), :, bbx1:bbx2, bby1:bby2], grid_temps.repeat(1, 1, 1, 1),
                        align_corners=True)
                if opt.attack_mode == "all2one":
                    targets_bd = torch.ones_like(targets) * opt.target_label
                if opt.attack_mode == "all2all":
                    targets_bd = torch.remainder(targets + 1, opt.num_classes)
                preds_bd = net(inputs_bd)
                total_bd_correct += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)
                acc_clean = total_clean_correct * 100.0 / total_sample
                acc_bd = total_bd_correct * 100.0 / total_sample

                info_string = "Clean Acc: {:.4f} - Best: {:.4f} | Bd Acc: {:.4f} - Best: {:.4f}".format(acc_clean, best_clean_acc, acc_bd, best_bd_acc)
                progress_bar(batch_idx, len(test_dl), info_string)

            if opt.attack_choice == "clean":
                grid_temps = (identity_grid + opt.s * noise_grid / (opt.input_height // opt.ratio)) * opt.grid_rescale
                grid_temps = torch.clamp(grid_temps, -1, 1).float()
                inputs_bd = inputs
                for idv_img in range(bs):
                    inputs_bd[idv_img:(idv_img + 1), :, :, :] = F.grid_sample(
                        inputs_bd[idv_img:(idv_img + 1), :, :, :], grid_temps.repeat(1, 1, 1, 1),
                        align_corners=True)
                preds_bd = net(inputs_bd)
                total_bd_correct += torch.sum(torch.argmax(preds_bd, 1) == opt.target_label)
                acc_clean = total_clean_correct * 100.0 / total_sample
                acc_bd = total_bd_correct * 100.0 / total_sample
                info_string = "Clean Acc: {:.4f} - Best: {:.4f} | Bd Acc: {:.4f} - Best: {:.4f}".format(acc_clean,
                                                                                                        best_clean_acc,
                                                                                                        acc_bd,
                                                                                                        best_bd_acc)
                progress_bar(batch_idx, len(test_dl), info_string)

    # tensorboard
    if not epoch % 1:
        tf_writer.add_scalars("Test Accuracy", {"Clean": acc_clean, "Bd": acc_bd}, epoch)

    # Save checkpoint
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
            "attack_choice": opt.attack_choice
        }
        torch.save(state_dict, opt.ckpt_path)
        with open(os.path.join(opt.ckpt_folder, "results.txt"), "w+") as f:
            results_dict = {
                "clean_acc": best_clean_acc.item(),
                "bd_acc": best_bd_acc.item()
            }
            json.dump(results_dict, f, indent=2)
    return best_clean_acc, best_bd_acc


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

    # Dataset
    train_dl = get_dataloader(opt, True)
    test_dl = get_dataloader(opt, False)

    # prepare model
    net, optimizer, scheduler = get_model(opt)

    # Load pretrained model
    mode = opt.attack_mode
    opt.ckpt_folder = os.path.join(opt.checkpoints, 'cifar10')
    opt.ckpt_path = os.path.join(opt.ckpt_folder, "{}_{}_{}.pth.tar".format(opt.dataset, mode, opt.attack_choice))
    opt.log_dir = os.path.join(opt.ckpt_folder, "log_dir")

    if not os.path.exists(opt.log_dir):
        os.makedirs(opt.log_dir)

    if opt.continue_training:
        if os.path.exists(opt.ckpt_path):
            print("Continue training")
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
        else:
            print("Pretrained model doesnt exist")
            exit()
    else:
        print("Train from scratch")
        best_clean_acc = 0.0
        best_bd_acc = 0.0
        epoch_current = 0

        # Prepare grid
        ins = np.random.beta(1, 1, (1, 2, opt.k, opt.k)) * 2 - 1
        ins = torch.tensor(ins)
        ins = ins / torch.mean(torch.abs(ins))
        noise_grid = (
            F.upsample(ins, size=(opt.input_height // opt.ratio), mode="bicubic", align_corners=True)
                .permute(0, 2, 3, 1)
                .to(opt.device)
        )
        array1d = torch.linspace(-1, 1, steps=opt.input_height // opt.ratio)
        x, y = torch.meshgrid(array1d, array1d)
        identity_grid = torch.stack((y, x), dim=2)[None, ...].to(opt.device)

        shutil.rmtree(opt.ckpt_folder, ignore_errors=True)
        os.makedirs(opt.log_dir)
        with open(os.path.join(opt.ckpt_folder, "opt.json"), "w+") as f:
            json.dump(opt.__dict__, f, indent=2)
        tf_writer = SummaryWriter(log_dir=opt.log_dir)

    for epoch in range(epoch_current, opt.n_iters):
        print("Epoch {}:".format(epoch + 1))
        train(net, optimizer, scheduler, train_dl, noise_grid, identity_grid, tf_writer, epoch, opt)
        best_clean_acc, best_bd_acc = eval(net, optimizer, scheduler, test_dl, noise_grid, identity_grid, best_clean_acc, best_bd_acc, tf_writer, epoch, opt)


if __name__ == "__main__":
    main()
