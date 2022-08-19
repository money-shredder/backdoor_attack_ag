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
$ python train.py --dataset <datasetName> --p <ratio> 
```

where the parameters are the following:
- `<datasetName>`: `mnist` , `cifar10` , `gtsrb` , `celeba`.

The trained checkpoints should be saved at the path `checkpoints\<datasetName>\<datasetName>_<attackMode>_<attackChoice>.pth.tar`.

Test for trained models, run command (clean label)
```bash
$ python eval.py --dataset <datasetName> --p <ratio> 
```
Other parameters are the following:
- `--target_label`: Labels to be attacked


