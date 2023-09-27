import argparse
import datetime
import os

import cv2
import numpy as np
import torch

from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str,
                    default="../data/example_species", help='Root dir for colony data')
parser.add_argument('--ckpt', type=str,
                    default="../VT-UNet/model/vit_checkpoint/coral_fine_tuned/coral_final_topoloss.pth",
                    help='Path to model checkpoint')
parser.add_argument('--cuda', type=bool, default=False, help='Enable CUDA')
parser.add_argument('--img_size', type=int, default=224, help='Input image size')
args = parser.parse_args()


def tile_slice(slice_image, dims):
    tiles = []
    for y in range(0, slice_image.shape[0], int(dims[0] / 2)):
        for x in range(0, slice_image.shape[1], int(dims[1] / 2)):
            # We aren't bothered about the edge cases as the scans invariably have a substantial black boarder
            if x + dims[1] >= slice_image.shape[1] or y + dims[0] >= slice_image.shape[0]:
                continue

            tile = {"coords": {'x': x, 'y': y, 'dx': dims[1], 'dy': dims[0]}}
            tile['img'] = cv2.cvtColor(slice_image[y:y + dims[0], x:x + dims[1]], cv2.COLOR_BGR2GRAY)
            tile['img'] = torch.from_numpy(np.array([[tile['img']]]).astype(np.float32))
            if np.count_nonzero(tile['img']) == 0:
                continue
            # Tiles need to be tensors of dims [1, 1, 224, 224] for prediction
            tiles.append(tile)
    return tiles


def set_up_model(model_path, img_size, cuda):
    vit_name = 'R50-ViT-B_16'
    config_vit = CONFIGS_ViT_seg[vit_name]
    config_vit.n_classes = 1
    config_vit.n_skip = 3
    config_vit.patches.size = (16, 16)
    config_vit['freeze'] = []
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(224 / 16), int(224 / 16))
    if cuda:
        model = ViT_seg(config_vit, img_size=img_size, num_classes=config_vit.n_classes).cuda()
    else:
        model = ViT_seg(config_vit, img_size=img_size, num_classes=config_vit.n_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model


def get_model_predicitons(model, input, cuda):
    if cuda:
        input = input.cuda()
    model.eval()
    with torch.no_grad():
        return model(input)


def reconstruct_img_from_preds(input_img, preds):
    overlap = 56
    full_img = np.zeros((input_img.shape[0], input_img.shape[1]))
    for pred in preds:
        ystart = pred['coords']['y'] + overlap
        yend = pred['coords']['y'] + pred['coords']['dy'] - overlap
        xstart = pred['coords']['x'] + overlap
        xend = pred['coords']['x'] + pred['coords']['dx'] - overlap

        full_img[ystart: yend, xstart: xend] = \
            pred['img'][overlap:pred['img'].shape[0] - overlap, overlap:pred['img'].shape[1] - overlap]

        # cv2.imshow("test", full_img)
        # cv2.waitKey()

    return full_img


def predict_single_slice(slice_path, dims, cuda, save_path=None, model=None, model_path=None):
    if not model:
        if not model_path:
            print("please provide a path to the trained model")
            exit(1)
        model = set_up_model(model_path)

    img = cv2.imread(slice_path)
    tiles = tile_slice(img, dims)

    preds = []
    for tile in tiles:
        pred = dict()
        pred['coords'] = tile['coords']
        predictions = get_model_predicitons(model, tile['img'], cuda)
        pred['img'] = ((torch.sigmoid(predictions[0][0]).detach().cpu().numpy() > 0.5) * 255).astype(np.uint8)
        preds.append(pred)

    reconstruction = reconstruct_img_from_preds(img, preds)
    if save_path:
        cv2.imwrite(save_path, reconstruction)
    return reconstruction


def predict_volume(volume_dir, save_dir, dims, cuda, model=None, model_path=None):
    if not model:
        if not model_path:
            print("please provide a path to the trained model")
            exit(1)
        model = set_up_model(model_path)

    axis = os.listdir(volume_dir)[0].split('Axis_')[1].split('_CentreSlice')[0]

    # os.makedirs(f"{save_dir}/{axis}", exist_ok=True)

    files = os.listdir(volume_dir)
    files.sort()
    slice_list = []
    for file in files:
        slice_num = file.split('Slice_-')[1].split('_Slab')[0]

        print(f"Predicting {slice_num} at {datetime.datetime.now().strftime('%d-%m-%Y, %H:%M:%S')}")
        slice_list.append(predict_single_slice(f"{volume_dir}/{file}", dims, cuda,
                                               save_path=f"{save_dir}/{slice_num}.png", model=model))
    return slice_list


if __name__ == '__main__':
    root = args.root
    vol_dir = f"{root}/complete_raw"
    save_dir = f"{root}/predictions"
    os.makedirs(save_dir, exist_ok=True)
    if args.ckpt is None:
        print("Please ensure a path to model checkpoint is provided")
        exit(1)

    model = set_up_model(args.ckpt, args.img_size, args.cuda)
    predicted_volume = predict_volume(vol_dir, save_dir, [args.img_size, args.img_size], args.cuda, model=model)
