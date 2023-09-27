import argparse
import logging
import os
import random
import sys

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset_coral import Coral_dataset
from utils import test_single_volume
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

parser = argparse.ArgumentParser()
parser.add_argument('--test_path', type=str,
                    default='../../data/example_species/training_data/snippets/test',
                    help='root dir for test data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Coral', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=1, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='../../data/example_species/training_data/data_lists', help='list dir')
parser.add_argument('--snapshot', type=str, default=None, help='Path to model snapshot')
parser.add_argument('--log_dir', type=str, default=None, help="Directory for results output")

parser.add_argument('--max_iterations', type=int, default=20000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=500, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=30,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", default=True, help='whether to save results during inference')
parser.add_argument('--vis_style', type=str, default='fill',
                    help='which visualisation style to use on results {"fill", "boundary"}')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')

parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
parser.add_argument('--cuda', type=bool, default=True, help='enable CUDA, default is True')
parser.add_argument('--split', type=str, default='test', help='Name of the list of testing data')
args = parser.parse_args()


def inference(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.test_path, split=args.split, list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    coral_metric_list = 0.0

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, label, case_name = sampled_batch["image"][0], sampled_batch["label"][0], sampled_batch['coral_name'][0]
        coral_metric = test_single_volume(args, image, label, model, classes=args.num_classes,
                                          patch_size=[args.img_size, args.img_size], test_save_path=test_save_path,
                                          case=case_name, z_spacing=args.z_spacing)

        coral_metric_list += np.array([y for (x, y) in list(coral_metric.items())])

        logging.info('idx %d case %s mean_dice %f, mean_hd95 %f, mean_center_mse '
                     '%f, mean_topo_acc %f, mean_non_zero_topo_acc %f, mean_nn_area_diff %f, mean_iou %f, mean_orient %f' %
                     (i_batch, case_name, coral_metric['dice'], coral_metric['hd95'], coral_metric['center_mse'],
                      coral_metric['topo_accuracy'], coral_metric['topo_non0acc'], coral_metric['nn_area_diff'],
                      coral_metric['iou'],
                      coral_metric['orientation_diffs']))
        logging.info('Corollite count: %d, Label Count: %d, Difference: %d' %
                     (int(coral_metric['pred_count']), int(coral_metric['pred_count'] - coral_metric['count_diff']),
                      int(coral_metric['count_diff'])))

    coral_metric_list = coral_metric_list / len(db_test)

    coral_performance = coral_metric_list[0]
    coral_mean_hd95 = coral_metric_list[1]
    coral_center_mse = coral_metric_list[9]
    coral_nn_area_diff = coral_metric_list[10]
    coral_count_diff = coral_metric_list[8]
    coral_iou = coral_metric_list[6]
    coral_orient = coral_metric_list[11]
    coral_topo = coral_metric_list[12]
    coral_non0topo = coral_metric_list[13]

    logging.info('Coral testing performance in best val model: mean_dice : %f mean_hd95 : %f' %
                 (coral_performance, coral_mean_hd95))
    logging.info('Region center MSE: %f, Mean Topological Accuracy: %f, Non-Zero Mean Topological Accuracy: %f' %
                 (coral_center_mse, coral_topo, coral_non0topo))
    logging.info('Nearest-neighbour area difference: %f, Average corallite count difference: %d' %
                 (coral_nn_area_diff, coral_count_diff))
    logging.info('Average absolute orientation difference (radians): %f, Average IoU: %f' %
                 (coral_orient, coral_iou))

    print(f"Average: dice: {coral_metric_list[0]:.4f},  hd95: {coral_metric_list[1]:.4f}, "
          f"asd: {coral_metric_list[2]:.4f}, sensitivity: {coral_metric_list[3]:.4f}, "
          f"specificity: {coral_metric_list[4]:.4f}, precision: {coral_metric_list[5]:.4f}, "
          f"region center MSE: {coral_metric_list[9]:.4f}, region area difference: {coral_metric_list[10]:.4f}, "
          f"iou: {coral_metric_list[6]:.4f}, corallite count difference: {coral_metric_list[8]}")
    return "Testing Finished!"


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
    img_size = args.img_size

    dataset_config = {
        'Coral': {
            'Dataset': Coral_dataset,
            'test_path': f'../../data/example_species/training_data/snippets/test',
            'list_dir': f'../../data/example_species/training_data/data_lists',
            'snapshot': f'../model/vit_checkpoint/coral_fine_tuned/coral_final_topoloss.pth',
            'num_classes': 1,
            'z_spacing': 1,
            'base_lr': 0.000625,
            'cuda': False,
            'split': 'test'
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.test_path = dataset_config[dataset_name]['test_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.base_lr = dataset_config[dataset_name]['base_lr']
    args.cuda = dataset_config[dataset_name]['cuda']
    args.split = dataset_config[dataset_name]['split']
    args.snapshot = dataset_config[dataset_name]['snapshot']
    args.is_pretrain = True

    snapshot = args.snapshot
    if snapshot is None:
        # name the same snapshot defined in train script!
        args.exp = 'TU_' + dataset_name + str(args.img_size)
        snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
        snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
        snapshot_path += '_' + args.vit_name
        snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
        snapshot_path = snapshot_path + '_vitpatch' + str(
            args.vit_patches_size) if args.vit_patches_size != 16 else snapshot_path
        snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
        if dataset_name == 'ACDC':  # using max_epoch instead of iteration to control training duration
            snapshot_path = snapshot_path + '_' + str(args.max_iterations)[
                                                  0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
        snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
        snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
        snapshot_path = snapshot_path + '_' + str(args.img_size)
        snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path
        snapshot = os.path.join(snapshot_path, 'best_model.pth').replace('\\', '/')

    # snapshot = "/Users/rob/University/University/Year 3/Coral/CoralGrowth/saved_models/VT_UNet/Final Models/FT_Slice_01279_lrg/TU_pretrain_R50-ViT-B_16_skip3_epo300_bs5_lr0.0005_224_ft_lrg/best_weights.pth"
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_' + str(args.max_epochs - 1))

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    config_vit['freeze'] = 'tcm'
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (
        int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))

    if args.cuda:
        net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
        net.load_state_dict(torch.load(snapshot.replace('\\', '/')))
    else:
        net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes)
        net.load_state_dict(torch.load(snapshot.replace('\\', '/'), map_location=torch.device('cpu')))

    snapshot_name = snapshot.split('/')[-2]

    log_folder = args.log_dir
    if log_folder is None:
        log_folder = f'./test_log/{snapshot_name}'
    os.makedirs(log_folder, exist_ok=True)
    os.makedirs(f"{log_folder}/results/preds", exist_ok=True)
    os.makedirs(f"{log_folder}/results/overlay", exist_ok=True)
    os.makedirs(f"{log_folder}/results/centers", exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + snapshot_name + ".txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    inference(args, net, test_save_path=log_folder)
