import config
import torch
import torch.nn.functional as F
from model.preact_resnet import  PreActResNet18
from model.resnet import ResNet18
from model.MNISTnet import MNISTnet
from utils.dataloader import get_dataloader
from utils.utils import progress_bar
from utils.utils import saliency_bbox

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"

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
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, opt.scheduler_milestones, opt.scheduler_lambda)
    return net, optimizer, scheduler

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

    # Dataset and model
    test_dl = get_dataloader(opt, False)
    net, optimizer, scheduler = get_model(opt)

    # pretrained model
    mode = opt.attack_mode
    opt.ckpt_folder = os.path.join(opt.checkpoints, opt.dataset)
    opt.ckpt_path = os.path.join(opt.ckpt_folder, "{}_{}_{}.pth.tar".format(opt.dataset, mode, opt.attack_choice))
    opt.log_dir = os.path.join(opt.ckpt_folder, "log_dir")
    state_dict = torch.load(opt.ckpt_path)
    net.load_state_dict(state_dict["net"])
    identity_grid = state_dict["identity_grid"]
    noise_grid = state_dict["noise_grid"]
    eval(net, optimizer, scheduler, test_dl, noise_grid, identity_grid, opt)


if __name__ == "__main__":
    main()

