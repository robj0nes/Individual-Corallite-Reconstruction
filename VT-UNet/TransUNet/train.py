import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_coral

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../../data/example_species/training_data/snippets/train', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Coral', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='../../data/example_species/training_data/data_lists', help='directory containing paths to training files.')
parser.add_argument('--num_classes', type=int,
                    default=1, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=500, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=30, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.0005,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is 3')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
parser.add_argument('--cuda', type=bool, default=True, help='enable CUDA, default is True')
parser.add_argument('--input_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--data_subset', type=str, default='Porites_6781_slice01279', help="subset directory for data")
parser.add_argument('--tag', type=str, help='tag to append to save directory')
parser.add_argument('--topo_lambda', type=float, default=0.1, help='Lamdba scalar for topological function, default '
                                                                   'is 0.1')
parser.add_argument('--vis', type=bool, default=False, help="Flag to visualise outputs each epoch")
parser.add_argument('--topo_only', type=bool, default=False, help="Flag to just use topo_loss function")
parser.add_argument('--wandb', type=bool, default=False, help="Enable w&b logging")
parser.add_argument('--phase', type=bool, default=False, help="Use phase-in of topo_lambda")
parser.add_argument('--nogauss', type=bool, default=False, help="Turn off Gaussian noise in augmentations")
parser.add_argument("--nowarmup", type=bool, default=False, help="Turn off warmup period")

args = parser.parse_args()

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset

    # The below config will either overwrite the command line args, or provide a way to set them in-script.
    dataset_config = {
        'Coral': {
            'root_path': '../../data/example_species',
            'list_dir': '../../data/example_species/training_data/data_lists',
            'model_path': '../model/vit_checkpoint/coral_pretrain/coral_pretrain.pth',
            'cuda': False,
            'wandb': False,
            'num_classes': 1,
            'vis': False,
        }
    }

    args.cuda = dataset_config[dataset_name]['cuda']
    args.vis = dataset_config[dataset_name]['vis']
    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = f"{dataset_config[dataset_name]['root_path']}"
    args.list_dir = f"{dataset_config[dataset_name]['list_dir']}"
    args.is_pretrain = True
    args.wandb = dataset_config[dataset_name]['wandb']
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    
    
    snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(
        args.vit_patches_size) if args.vit_patches_size != 16 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.max_iterations)[
                                          0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.img_size)
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path
    if args.tag is not None:
        snapshot_path = snapshot_path + '_' + args.tag

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.pretrained_path = dataset_config[dataset_name]['model_path']

    if args.vis:
        vis_path = f"{snapshot_path}/topo_vis"
        config_vit.vis_path = vis_path
        if not os.path.exists(vis_path):
            os.makedirs(vis_path)
    else:
        config_vit.vis_path = None

    config_vit['freeze'] = ['tcm']

    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (
        int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))

    if args.cuda:
        net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    else:
        net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes)

    if '.npz' in config_vit.pretrained_path:
        net.load_from(weights=np.load(config_vit.pretrained_path))
    elif '.pth' in config_vit.pretrained_path:
        if args.cuda:
            net.load_state_dict(torch.load(config_vit.pretrained_path))
        else:
            net.load_state_dict(torch.load(config_vit.pretrained_path, map_location=torch.device('cpu')))
    else:
        print("Unable to load pre-trained model")
        exit(1)

    print(f"Training on data at: {args.root_path}")
    trainer = {'Coral': trainer_coral, }
    trainer['Coral'](args, net, snapshot_path)
