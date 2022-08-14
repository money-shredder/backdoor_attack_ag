import os
import sys
import time
import cv2
import torch
import config
import numpy as np
import torch.nn as nn
import torch.nn.init as init

def get_mean_and_std(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print("----Computing mean and std----")
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode="fan_out")
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

_, term_width = os.popen("stty size", "r").read().split()
term_width = int(term_width)
TOTAL_BAR_LENGTH = 65.0
last_time = time.time()
begin_time = last_time

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

def unsaliency_bbox(img):
    opt = config.get_arguments().parse_args()
    size = img.size()
    W = size[1]
    H = size[2]
    ratio = opt.ratio
    cut_w = int(W // ratio)
    cut_h = int(H // ratio)
    if opt.dataset == "mnist":
            bbx1 = 0
            bbx2 = W // opt.ratio
            bby1 = 0
            bby2 = H // opt.ratio
    else:
        # compute the image saliency map
        temp_img = img.cpu().numpy().transpose(1, 2, 0)
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        (success, saliencyMap) = saliency.computeSaliency(temp_img)
        saliencyMap = (saliencyMap * 255).astype("uint8")
        maximum_indices = np.unravel_index(np.argmax(saliencyMap, axis=None), saliencyMap.shape)
        x = maximum_indices[0]
        y = maximum_indices[1]
        if x < cut_w:
            bbx1 = W - W // opt.ratio
            bbx2 = W
        else:
            bbx1 = 0
            bbx2 = W // opt.ratio
        if y < cut_h:
            bby1 = H - H // opt.ratio
            bby2 = H
        else:
            bby1 = 0
            bby2 = H // opt.ratio
    return bbx1, bby1, bbx2, bby2

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.
    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1
    sys.stdout.write(" [")
    for i in range(cur_len):
        sys.stdout.write("=")
    sys.stdout.write(">")
    for i in range(rest_len):
        sys.stdout.write(".")
    sys.stdout.write("]")
    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time
    L = []
    if msg:
        L.append(" | " + msg)
    msg = "".join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(" ")
    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write("\b")
    sys.stdout.write(" %d/%d " % (current + 1, total))
    if current < total - 1:
        sys.stdout.write("\r")
    else:
        sys.stdout.write("\n")
    sys.stdout.flush()

def format_time(seconds):
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    f = ""
    i = 1
    if hours > 0 and i <= 2:
        f += str(hours) + "h"
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + "m"
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + "s"
        i += 1
    return f