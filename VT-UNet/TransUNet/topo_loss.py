import os

import cv2
import numpy as np
import torch
from skimage.measure import regionprops
from skimage.measure import label as sklabel
from torch import nn
# from ph_topoloss import getPHLoss


class TopoLoss(nn.Module):
    def __init__(self, n_classes, cuda):
        super(TopoLoss, self).__init__()
        self.n_classes = n_classes
        self.cuda = cuda

    def _topo_loss(self, score, target, epoch, vis_path=None, names=None):
        pred = torch.zeros_like(score)
        gt = torch.zeros_like(target)
        pixel_errors = torch.zeros_like(score)
        pred[score > 0.5] = 1.0
        gt[target > 0.5] = 1.0

        label_regions = []
        prediction_regions = []
        for i in range(score.shape[0]):
            if self.cuda:
                label_regions.append(regionprops(sklabel(gt[i].cpu())))
                prediction_regions.append(regionprops(sklabel(pred[i].cpu())))
            else:
                label_regions.append(regionprops(sklabel(gt[i])))
                prediction_regions.append(regionprops(sklabel(pred[i])))

        pixel_errors = self.calculate_error(pixel_errors, gt, label_regions, pred, prediction_regions)

        loss_map = torch.mul(torch.sub(torch.sigmoid(torch.mul(torch.add(score, 1), pixel_errors)), 0.5), 2)
        # loss_map = torch.mul(torch.add(score, 1), pixel_errors)
        loss = torch.mean(loss_map)

        topo_metric_mean = 1 - torch.mean(pixel_errors)
        topo_metric_mean_non_zero = 1 - torch.sum(pixel_errors) / torch.count_nonzero(pixel_errors)

        if vis_path is not None:
            # Visualise the loss map adjacent to predictions/gts.
            self.visualise_topoloss(gt.cpu().detach().numpy(),
                                    loss_map.cpu().detach().numpy(),
                                    pred.cpu().detach().numpy(),
                                    pixel_errors.cpu().detach().numpy(),
                                    epoch,
                                    f"{vis_path}",
                                    names)

        return loss, topo_metric_mean, topo_metric_mean_non_zero

    def calculate_error(self, pixel_errors, gt, label_regions, pred, prediction_regions):
        for i in range(gt.shape[0]):
            if prediction_regions[i]:
                # If no labels assign all predicted pixels an error of one.
                if not label_regions[i]:
                    for region in prediction_regions[i]:
                        for pixel in region.coords:
                            pixel_errors[i, pixel[0], pixel[1]] = 1
                else:
                    for region in prediction_regions[i]:
                        min_dist = np.infty
                        nn_index = 0
                        pred_center = np.array([region.centroid[0], region.centroid[1]])

                        # Find the nearest neighbour label
                        for j, lab_region in enumerate(label_regions[i]):
                            label_center = np.array([lab_region.centroid[0], lab_region.centroid[1]])
                            dist = np.sqrt(
                                (pred_center[0] - label_center[0]) ** 2 + (pred_center[1] - label_center[1]) ** 2)
                            if dist < min_dist:
                                min_dist = dist
                                nn_index = j

                        # If the closest label doesn't overlap: all pixel in predicted region assigned error of 1.
                        if not self.check_overlap(gt[i], region):
                            for pixel in region.coords:
                                pixel_errors[i, pixel[0], pixel[1]] = 1

                        else:
                            # Calculate the difference in areas between regions and generate an error
                            label_area = label_regions[i][nn_index].area
                            pred_area = region.area
                            ratio = 1 - (label_area / pred_area)
                            error = min(np.abs(1 - np.exp(ratio)), 1.0)

                            # For each pixel in the prediction, if it is not a ground truth then update error
                            for pixel in region.coords:
                                if gt[i, pixel[0], pixel[1]] != 1:
                                    # Save maximum error
                                    if pixel_errors[i, pixel[0], pixel[1]] < error:
                                        pixel_errors[i, pixel[0], pixel[1]] = error

                            # For each pixel in the gt, if it is missed by predictions then update error
                            for pixel in label_regions[i][nn_index].coords:
                                if pred[i, pixel[0], pixel[1]] < 1:
                                    # Save maximum error
                                    if pixel_errors[i, pixel[0], pixel[1]] < error:
                                        pixel_errors[i, pixel[0], pixel[1]] = error

                            # NOTE: Doesn't work for batches.. Debugging visualisations for error generation.
                            # self.debug_error(error, gt, label_regions, nn_index, pixel_errors, region)
            # TODO: Consider if we want to assign error for missed predictions with this loss.
            #  If so, score will need to be given some value to avoid multiplication cancelling out the errors..
            #  eg. loss_map = torch.mul(torch.sub(sigmoid(torch.mul(torch.add(score, 1), pixel_errors), 0.5), 2)
            # Finally, if we have labels, ensure that any missed predictions are also assigned error 1
            if label_regions[i]:
                for region in label_regions[i]:
                    # If there is no overlap with any predictions, assign error 1
                    if not self.check_overlap(pred[i], region):
                        for pixel in region.coords:
                            pixel_errors[i, pixel[0], pixel[1]] = 1
        return pixel_errors

    # Used when testing
    def visualise_topoloss(self, gt, loss_map, pred, pixel_errors, epoch, path, names):
        if not os.path.exists(f"{path}/epoch_{epoch}"):
            os.makedirs(f"{path}/epoch_{epoch}")
        for i in range(gt.shape[0]):
            errors = (pixel_errors * 255).astype(np.uint8)
            losses = (loss_map * 255).astype(np.uint8)
            labels = (gt * 255).astype(np.uint8)
            preds = (pred * 255).astype(np.uint8)

            losses = cv2.cvtColor(losses[i], cv2.COLOR_GRAY2RGB)
            errors = cv2.cvtColor(errors[i], cv2.COLOR_GRAY2RGB)

            zeros = np.zeros((labels[i].shape[0], labels[i].shape[1], 3), dtype=np.uint8)
            rgb_labels = np.copy(zeros)
            rgb_labels[:, :, 0] = labels[i]
            rgb_preds = np.copy(zeros)
            rgb_preds[:, :, 2] = preds[i]
            preds_and_labels = np.copy(rgb_preds)
            preds_and_labels[:, :, 0] = rgb_labels[:, :, 0]

            comb_image = np.concatenate((errors, losses, preds_and_labels), axis=1)
            filename = names[i].strip('.npz')
            cv2.imwrite(f"{path}/epoch_{epoch}/{filename}.png", comb_image)
            cv2.imwrite(f"{path}/epoch_{epoch}/{filename}_PRED.png", preds[i, :, :])

    # Used when testing
    def debug_error(self, error, gt, label_regions, nn_index, pixel_errors, region):
        combined_im = np.zeros((224, 224, 3)).astype(np.uint8)
        error_im = np.zeros((224, 224, 3)).astype(np.uint8)
        data = ((np.array(pixel_errors[0, :, :]) > 0) * 255).astype(np.uint8)
        p_errs = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)

        # Add prediciton to combined image (in red) and associated error to error image
        for pix in region.coords:
            combined_im[pix[1], pix[2], 2] = 255
            error_im[pix[1], pix[2], :] = pixel_errors[pix[0], pix[1], pix[2]] * 255

        # Add gt label to combined image (in blue)
        for pix in label_regions[nn_index].coords:
            combined_im[pix[1], pix[2], 0] = 255
        cv2.imshow(f"{error}", np.concatenate((combined_im, error_im, p_errs), axis=1))
        cv2.waitKey()

    def check_overlap(self, gt, region):
        overlap = False
        for pixel in region.coords:
            if gt[pixel[0], pixel[1]] > 0:
                overlap = True
                break
        return overlap


    def forward(self, inputs, target, epoch, softmax=False, vis_path=None, names=None):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        inputs = inputs.float()
        inputs = torch.sigmoid(inputs)
        return self._topo_loss(inputs, target, epoch, vis_path=vis_path, names=names)
