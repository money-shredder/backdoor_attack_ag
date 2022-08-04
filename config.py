import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints")
    parser.add_argument("--temps", type=str, default="./temps")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train_scratch", action="store_false")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--attack_mode", type=str, default="all2one")
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--scheduler_milestones", type=list, default=[100, 200, 300, 400])
    parser.add_argument("--scheduler_lambda", type=float, default=0.1)
    parser.add_argument("--epoches", type=int, default=500)
    parser.add_argument("--num_workers", type=float, default=6)
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--ar", type=float, default=0.1)
    parser.add_argument("--random_rotation", type=int, default=10)
    parser.add_argument("--random_crop", type=int, default=5)
    parser.add_argument("--s", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--grid_rescale", type=float, default=1)
    parser.add_argument("--ratio", type=float, default=0.5)

    return parser