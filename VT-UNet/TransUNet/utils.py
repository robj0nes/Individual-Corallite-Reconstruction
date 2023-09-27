import math

import numpy as np
import torch
from medpy import metric
import torch.nn as nn
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
import cv2
from skimage.measure import label as sklabel
from skimage.measure import regionprops
from topo_loss import TopoLoss


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        dice = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - dice

        # compute metric
        out = torch.zeros_like(score)
        gt = torch.zeros_like(target)
        out[score > 0.5] = 1.0
        gt[target > 0.5] = 1.0

        inter = (out * gt).sum(dim=(1, 2))
        uni = out.sum(dim=(1, 2)) + gt.sum(dim=(1, 2))
        dice_metric = 2.0 * (inter) / (uni + smooth)

        num_zero_metric = len(dice_metric) - torch.count_nonzero(dice_metric)

        return loss, dice_metric.sum(), num_zero_metric

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),target.size())

        inputs = torch.sigmoid(inputs)
        loss, dice_metric, num_zero_metric = self._dice_loss(inputs, target)

        return loss, dice_metric, num_zero_metric


def calculate_metric_percoral(pred, gt):
    pred[pred > 0.5] = 1

    gt[gt > 0.5] = 1

    metrics = {"dice": 0, "hd95": 0, "asd": 0, "sensitivity": 0, "specificity": 0, "precision": 0}

    if pred.sum() > 0 and gt.sum() > 0:
        metrics['dice'] = metric.binary.dc(pred, gt)
        metrics['asd'] = metric.binary.asd(pred, gt)
        metrics['hd95'] = metric.binary.hd95(pred, gt)  # hd95
        metrics['sensitivity'] = metric.binary.sensitivity(pred, gt)
        metrics['specificity'] = metric.binary.specificity(pred, gt)
        metrics['precision'] = metric.binary.precision(pred, gt)

    return metrics


def check_regions(label, prediction, label_regions, pred_regions):
    fig, (ax, ax2) = plt.subplots(1, 2)
    ax.imshow(label, cmap=plt.cm.gray)
    ax2.imshow(prediction, cmap=plt.cm.gray)

    ax.axis((0, label.shape[0], label.shape[1], 0))
    ax.set_title('image')
    ax2.axis((0, label.shape[0], label.shape[1], 0))
    ax2.set_title('predictions')
    for props in label_regions:
        y0, x0 = props.centroid
        orientation = props.orientation
        x1 = x0 + math.cos(orientation) * 0.5 * props.axis_minor_length
        y1 = y0 - math.sin(orientation) * 0.5 * props.axis_minor_length
        x2 = x0 - math.sin(orientation) * 0.5 * props.axis_major_length
        y2 = y0 - math.cos(orientation) * 0.5 * props.axis_major_length

        ax.plot((x0, x1), (y0, y1), '-r', linewidth=0.75)
        ax.plot((x0, x2), (y0, y2), '-r', linewidth=0.75)
        ax.plot(x0, y0, '.g', markersize=3)

    for props in pred_regions:
        y0, x0 = props.centroid
        orientation = props.orientation
        x1 = x0 + math.cos(orientation) * 0.5 * props.axis_minor_length
        y1 = y0 - math.sin(orientation) * 0.5 * props.axis_minor_length
        x2 = x0 - math.sin(orientation) * 0.5 * props.axis_major_length
        y2 = y0 - math.cos(orientation) * 0.5 * props.axis_major_length

        ax2.plot((x0, x1), (y0, y1), '-r', linewidth=0.75)
        ax2.plot((x0, x2), (y0, y2), '-r', linewidth=0.75)
        ax2.plot(x0, y0, '.g', markersize=3)

    plt.show()


def calculate_coral_differences(coral_metrics, label_regions, pred_regions, image, test_save_path, case):
    preds_list = pred_regions.copy()
    nn_dists = []
    nn_area_diffs = []
    orientation_diffs = []
    test_im = cv2.cvtColor(image[0].copy(), cv2.COLOR_GRAY2RGB)

    threshold = 5
    for l in label_regions:
        if len(preds_list) == 0:
            nn_dists.append(1)
            nn_area_diffs.append(l['area'])
            orientation_diffs.append(0)
        else:
            min_dist = np.infty
            label_cent = (l.centroid[1], l.centroid[0])
            closest_pred = (0, 0)
            closest_area = 0
            closest_orientation = np.infty
            for p in pred_regions:
                pred_cent = (p.centroid[1], p.centroid[0])
                dist = math.sqrt((label_cent[0] - pred_cent[0]) ** 2 + (label_cent[1] - pred_cent[1]) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    closest_pred = (int(pred_cent[0]), int(pred_cent[1]))
                    closest_area = p.area
                    closest_orientation = p.orientation
            nn_dists.append(min_dist ** 2)
            nn_area_diffs.append(np.abs(closest_area - l.area))

            orientation_diffs.append(np.abs(np.abs(l.orientation) - np.abs(closest_orientation)))

            if test_save_path is not None:
                label_cent = (int(label_cent[0]), int(label_cent[1]))
                cv2.circle(test_im, label_cent, radius=1, color=(255, 0, 0), thickness=-1)
                cv2.circle(test_im, closest_pred, radius=1, color=(0, 0, 255), thickness=-1)
                if min_dist > threshold:
                    cv2.line(test_im, closest_pred, label_cent, color=(255, 255, 0), thickness=1)
                else:
                    cv2.line(test_im, closest_pred, label_cent, color=(0, 255, 0), thickness=1)
    cv2.imwrite(f"{test_save_path}/results/centers/{case.strip('.npz')}.png", test_im)

    if len(preds_list) != 0:
        for p in preds_list:
            nn_dists.append(1)
            nn_area_diffs.append(p['area'])

    coral_metrics['center_mse'] = np.sum(nn_dists) / len(nn_dists)
    coral_metrics['nn_area_diff'] = np.sum(nn_area_diffs) / len(nn_area_diffs)
    coral_metrics['orientation_diffs'] = np.sum(orientation_diffs) / len(orientation_diffs)
    return coral_metrics


def test_single_volume(args, image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None,
                       z_spacing=1):
    label_tensor = label
    if args.cuda:
        image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
        input = torch.from_numpy(image).unsqueeze(0).float().cuda()
    else:
        image = image.numpy()
        label = label.squeeze(0).numpy()
        input = torch.from_numpy(image).unsqueeze(0).float()

    net.eval()
    with torch.no_grad():

        out_3layers = net(input)

        if args.cuda:
            out_layer1 = torch.sigmoid(out_3layers[:, 0, :, :]).cpu().detach().numpy().squeeze(0)
        else:
            out_layer1 = torch.sigmoid(out_3layers[:, 0, :, :]).numpy().squeeze(0)

        pred_tensor = out_3layers[:, 0, :, :]
        corallite_prediction = np.uint8(out_layer1 >= 0.5)

    coral_metric = calculate_metric_percoral(corallite_prediction, label)

    label_regions = regionprops(sklabel(label))
    pred_regions = regionprops(sklabel(corallite_prediction))

    # check_regions(label, corallite_prediction, label_regions, pred_regions)
    coral_metric['iou'] = np.sum(label * corallite_prediction) / (np.sum(label) + np.sum(corallite_prediction) - np.sum(label * corallite_prediction))
    coral_metric['pred_count'] = len(pred_regions)
    coral_metric['count_diff'] = len(pred_regions) - len(label_regions)
    coral_metric = calculate_coral_differences(coral_metric, label_regions, pred_regions, image, test_save_path, case)

    t_loss, t_acc_mean, t_acc_non0mean = TopoLoss(n_classes=1, cuda=False).forward(pred_tensor, label_tensor, 0)
    coral_metric['topo_accuracy'] = t_acc_mean.cpu().detach().numpy()
    coral_metric['topo_non0acc'] = t_acc_non0mean.cpu().detach().numpy()

    if test_save_path is not None:
        image = image[int((image.shape[0] - 1) / 2), :, :]
        corallite_prediction = corallite_prediction.astype(np.float32)
        label = label.astype(np.float32)

        if args.vis_style == 'fill':
            label_im = cv2.cvtColor(label, cv2.COLOR_GRAY2RGB)
            label_im = (label_im > 0) * np.array([225, 0, 0]).astype(np.uint8)
            coral_mask_pred = cv2.cvtColor(corallite_prediction, cv2.COLOR_GRAY2RGB)
            coral_mask_pred = (coral_mask_pred > 0) * np.array([0, 0, 225]).astype(np.uint8)
            cv2.imwrite(f'{test_save_path}/results/preds/{case.strip(".npz")}.png', coral_mask_pred)
            combination = np.add(label_im, coral_mask_pred)

            test_im = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            coral_composite_img = cv2.addWeighted(test_im, 1, combination, 0.4, 0)

        elif args.vis_style == 'boundary':

            # Define a show mask on image function
            corallite_label_to_show = find_boundaries(label, mode='thick')
            corallite_label_to_show = np.uint8(corallite_label_to_show)
            corallite_label_to_show = cv2.normalize(corallite_label_to_show, None, 0, 255, cv2.NORM_MINMAX)

            coral_mask_pred = find_boundaries(corallite_prediction, mode='thick')
            coral_mask_pred = np.uint8(coral_mask_pred)
            coral_mask_pred = cv2.normalize(coral_mask_pred, None, 0, 255, cv2.NORM_MINMAX)

            coral_composite = np.zeros((coral_mask_pred.shape[0], coral_mask_pred.shape[1], 3))
            coral_composite[:, :, 0] = corallite_label_to_show
            coral_composite[:, :, 2] = coral_mask_pred

            image_three_layers = np.zeros((coral_mask_pred.shape[0], coral_mask_pred.shape[1], 3))
            image_three_layers[:, :, 0] = image
            image_three_layers[:, :, 1] = image
            image_three_layers[:, :, 2] = image

            coral_composite_img = image_three_layers + coral_composite
            coral_composite_img = cv2.normalize(coral_composite_img, None, 0, 430, cv2.NORM_MINMAX)

        # set figures
        def add_text(postion, color, title_text):
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = postion
            fontScale = 0.4
            fontColor = color
            lineType = 1

            cv2.putText(coral_composite_img,
                        title_text,
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)

        # set figures
        txt1 = add_text((380 - 256, 30), (255, 0, 0), '-Ground Truth')
        txt2 = add_text((380 - 256, 45), (0, 0, 255), '-Test')
        txt3 = add_text((380 - 256, 60), (0, 0, 0), '-DSC: {:.4f}'.format(coral_metric['dice']))
        txt3_5 = add_text((380 - 256, 75), (0, 0, 0), '-HD95: {:.4f}'.format(coral_metric['hd95']))
        txt4 = add_text((380 - 256, 90), (0, 0, 0), '-IoU: {:.4f}'.format(coral_metric['iou']))

        cv2.imwrite(f'{test_save_path}/results/overlay/{case.strip(".npz")}.png', coral_composite_img)

    return coral_metric
