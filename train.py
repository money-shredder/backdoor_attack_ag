import json
import shutil
import config
import numpy as np
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
from utils.utils import progress_bar, saliency_bbox, unsaliency_bbox
import os


def get_model(opt):
    if opt.dataset == "cifar10" or opt.dataset == "gtsrb":
        net = PreActResNet18(num_classes=opt.num_classes).to(opt.device)
    if opt.dataset == "celeba":
        net = ResNet18(num_classes=opt.num_classes).to(opt.device)
    if opt.dataset == "mnist":
        net = MNISTnet().to(opt.device)
    optimizer = torch.optim.SGD(net.parameters(), opt.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, opt.scheduler_milestones, opt.scheduler_lambda)
    return net, optimizer, scheduler


def train(net, optimizer, scheduler, train_dl, identity_grid, ins1, tf_writer, epoch, opt):
    print(" Train:")
    device_ids = [0, 1, 2]
    net = torch.nn.DataParallel(net, device_ids=device_ids).cuda()
    net.train()
    total_loss_ce = 0
    total_sample = 0
    total_clean_correct = 0
    criterion_CE = torch.nn.CrossEntropyLoss()
    denormalizer = Denormalizer(opt)
    transforms = PostTensorTransform(opt).to(opt.device)

    for batch_idx, (inputs, targets) in enumerate(train_dl):
        optimizer.zero_grad()
        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        bs = inputs.shape[0]
        num_bd = 0
        input_origin = copy.deepcopy(inputs)
        for id_img in range(bs * 7 //10):
            temp_label = targets[id_img]
            grid_temps = (identity_grid + ins1[temp_label] /
                    opt.input_height)
            grid_temps = torch.clamp(grid_temps, -1, 1).float()
            # p = torch.rand(1)
            # if p > opt.p:
                # bbx1, bby1, bbx2, bby2 = unsaliency_bbox(inputs[id_img])
                # temp = inputs[id_img:(id_img + 1), :, bbx1:bbx2, bby1:bby2]
                # inputs[id_img:(id_img + 1), :, bbx1:bbx2, bby1:bby2] = F.grid_sample(temp,
                #                                                                      grid_temps2.repeat(1, 1, 1, 1),
                #                                                                      align_corners=True)
            inputs[id_img:(id_img + 1), :, :, :] = F.grid_sample(inputs[id_img:(id_img + 1), :, :, :],
                                                                          grid_temps.repeat(1, 1, 1, 1),
                                                                          align_corners=True)
            num_bd += 1
        inputs_ag = transforms(inputs[bs * 7 //10:])
        inputs = torch.cat((inputs[:bs * 7 //10], inputs_ag), dim=0)
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
        tf_writer.add_scalars("Clean Accuracy", {"Clean": avg_acc_clean}, epoch)
        tf_writer.add_image("Images", grid, global_step=epoch)
    scheduler.step()


def eval(net, optimizer, scheduler, test_dl, identity_grid, ins1, best_clean_acc, best_bd_acc, tf_writer, epoch, opt):
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
            grid_temps = (identity_grid + ins1[0] / opt.input_height)
            grid_temps = torch.clamp(grid_temps, -1, 1).float()

            inputs_bd = inputs
            for idv_img in range(bs):
                # bbx1, bby1, bbx2, bby2 = saliency_bbox(inputs_bd[idv_img])
                # temp = inputs_bd[idv_img:(idv_img + 1), :, bbx1:bbx2, bby1:bby2]
                # inputs_bd[idv_img:(idv_img + 1), :, bbx1:bbx2, bby1:bby2] = F.grid_sample(
                #     temp, grid_temps2.repeat(1, 1, 1, 1),
                #     align_corners=True)
                inputs_bd[idv_img:(idv_img + 1), :, :, :] = F.grid_sample(inputs_bd[idv_img:(idv_img + 1), :, :, :], grid_temps.repeat(1, 1, 1, 1),align_corners=True)
            preds_bd = net(inputs_bd)
            targets_bd = torch.ones_like(targets) * opt.target_label
            total_bd_correct += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)
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
    if (acc_clean > best_clean_acc and acc_bd > best_bd_acc):
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
            "ins1":ins1,
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
    opt.ckpt_folder = os.path.join(opt.checkpoints, 'cifar10-p=0.5_ag')
    opt.ckpt_path = os.path.join(opt.ckpt_folder, "{}.pth.tar".format(opt.dataset))
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
            ins1 = state_dict["ins1"]
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
        array1d = torch.linspace(-1, 1, steps=opt.input_height)
        x, y = torch.meshgrid(array1d, array1d)
        identity_grid = torch.stack((y, x), dim=2)[None, ...].to(opt.device)
        ins1 = np.random.beta(1, 1, (opt.num_classes, 1, opt.input_height, opt.input_height, 2)) * 2 - 1
        ins1 = torch.tensor(ins1).to(opt.device)


        shutil.rmtree(opt.ckpt_folder, ignore_errors=True)
        os.makedirs(opt.log_dir)
        with open(os.path.join(opt.ckpt_folder, "opt.json"), "w+") as f:
            json.dump(opt.__dict__, f, indent=2)
        tf_writer = SummaryWriter(log_dir=opt.log_dir)

    for epoch in range(epoch_current, opt.n_iters):
        print("Epoch {}:".format(epoch + 1))
        train(net, optimizer, scheduler, train_dl, identity_grid, ins1, tf_writer, epoch, opt)
        best_clean_acc, best_bd_acc = eval(net, optimizer, scheduler, test_dl, identity_grid, ins1, best_clean_acc, best_bd_acc, tf_writer, epoch, opt)


if __name__ == "__main__":
    main()
