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
```bash
$ python train.py --dataset <datasetName> --attack_mode <attackMode>
```
where the parameters are the following:
- `<datasetName>`: `mnist` | `cifar10` | `gtsrb` | `celeba`.
- `<attackMode>`: `all2one`  or `all2all`
The trained checkpoints should be saved at the path `checkpoints\<datasetName>\<datasetName>_<attackMode>.pth.tar`.

Test for trained models, run command
```bash
$ python eval.py --dataset <datasetName> --attack_mode <attackMode>
```
