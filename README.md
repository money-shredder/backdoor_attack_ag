# Backdoor attack via augmentation 
#### Contents
1. [Introduction](#Agattack)
2. [Requirements](#Requirements)
3. [Experiments](#Experiments)

## Agattack - Backdoor Attack via Augmentation
Introdution

## Requirements
- Install required python packages:
```bash
$ python -m pip install -r requirement.py
```

## Experiments
Train command 

Clean label: only operate on images
```bash
$ python train.py --dataset <datasetName> --attack_choice clean 
```
Dirty label: control the whole training process
```bash
$ python train.py --dataset <datasetName> --attack_mode <attackMode> 
```

where the parameters are the following:
- `<datasetName>`: `mnist` , `cifar10` , `gtsrb` , `celeba`.
- `<attackMode>`: `all2one` or `all2all`
- `<attackChoice>`: `clean` or `dirty`

The trained checkpoints should be saved at the path `checkpoints\<datasetName>\<datasetName>_<attackMode>_<attackChoice>.pth.tar`.

Test for trained models, run command (clean label)
```bash
$ python eval.py --dataset <datasetName> --attack_choice clean 
```
dirty label:
```bash
$ python eval.py --dataset <datasetName> --attack_mode <attackMode> 
```
Other parameters are the following:
- `--s`: Indicates the scale of distortion (Recommendation: 0.2, 0.5, 1) 
- `--ratio`: Proportion of the distorted area to the original image (scale_height(width) = height(width) / ratio) (Recommendation: 1, 2)
- `--target_label`: Labels to be attacked


