import argparse
import logging
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# from tensorboardX import SummaryWriter
# from torch import tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torch.nn.modules.loss import CrossEntropyLoss
import torch.nn.functional as F
# from torch.utils.data import DataLoader
# import torchgeometry as tgm
from tqdm import tqdm
from utils import DiceLoss, TopoLoss

# from torchvision import transforms
# from skimage.measure import label as sklabel
# from skimage.measure import regionprops
import copy
import matplotlib.pylab as plt
import pickle
import wandb


def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']


def visualise_predicitons(preds, targets, epoch, path, names):
    if not os.path.exists(f"{path}/epoch_{epoch}"):
        os.makedirs(f"{path}/epoch_{epoch}")

    sample_pred = np.array((torch.sigmoid(preds).detach().cpu() > 0.5) * 255).astype(np.uint8)
    for i in range(targets.shape[0]):
        overlay = np.zeros((targets[i].shape[0], targets[i].shape[1], 3), dtype=np.uint8)

        target = ((np.array(targets[i].detach().cpu()) > 0) * 255).astype(np.uint8)
        overlay[:, :, 0] = target
        overlay[:, :, 2] = sample_pred[i]

        target = cv2.cvtColor(target, cv2.COLOR_GRAY2RGB)
        preds = cv2.cvtColor(sample_pred[i], cv2.COLOR_GRAY2RGB)

        frame = np.concatenate((target, preds, overlay), axis=1)

        filename = names[i].strip('.npz')
        cv2.imwrite(f"{path}/epoch_{epoch}/{filename}.png", frame)
        # cv2.imwrite(f"{path}/epoch_{epoch}/{filename}_PRED.png", preds[i, :, :])


def loss_epoch(dataloader, model, dice_loss, topo_loss, args, epoch_num, optimizer=None):
    running_loss = 0.0
    running_topo_loss = 0.0
    running_coral_metric = 0.0
    running_topological_metric = 0.0

    len_data = len(dataloader.dataset)
    for i_batch, sampled_batch in enumerate(dataloader):
        image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        if args.cuda:
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()


        outputs = model(image_batch)
        loss_ce = F.binary_cross_entropy_with_logits(outputs[:, 0, :, :], label_batch[:, 0, :, :].float())
        loss_dice, coral_dice_metric, num_zero_metric = dice_loss(outputs[:, 0, :, :], label_batch[:, 0, :, :])

        topological_loss = torch.tensor(0)
        topological_metric = torch.tensor(0)

        if args.vis:
            visualise_predicitons(outputs[:, 0, :, :], label_batch[:, 0, :, :], epoch_num, model.config.vis_path, sampled_batch['coral_name'])

        # Custom Topological Loss
        elif args.topo_lambda != 0:
            # loss = 0.5 * loss_ce + 0.5 * loss_dice + args.topo_lambda * topological_loss
            if args.topo_only:
                topological_loss, topological_metric, non_zero_metric = topo_loss(outputs[:, 0, :, :], label_batch[:, 0, :, :], epoch_num, vis_path=model.config.vis_path, names=sampled_batch['coral_name'])
                loss = args.topo_lambda * topological_loss
            
            elif args.nowarmup:
                topological_loss, topological_metric, non_zero_metric = topo_loss(outputs[:, 0, :, :], label_batch[:, 0, :, :], epoch_num, vis_path=model.config.vis_path, names=sampled_batch['coral_name'])
                loss = 0.5 * loss_ce + 0.5 * loss_dice + args.topo_lambda * topological_loss

            elif epoch_num > 50:
                phase_scalar = 1
                if args.phase:
                    if epoch_num < 150:
                        phase_scalar = (epoch_num - 50) / 100
                topological_loss, topological_metric, non_zero_metric = topo_loss(outputs[:, 0, :, :], label_batch[:, 0, :, :], epoch_num, vis_path=model.config.vis_path, names=sampled_batch['coral_name'])
                loss = 0.5 * loss_ce + 0.5 * loss_dice + args.topo_lambda * phase_scalar * topological_loss
            else:
                loss = 0.5 * loss_ce + 0.5 * loss_dice

        # Original VT-UNet Loss
        else:
            loss = 0.5 * loss_ce + 0.5 * loss_dice


        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running_loss += loss
        running_topo_loss += topological_loss * args.topo_lambda
        running_coral_metric += coral_dice_metric
        running_topological_metric += topological_metric

    coral_metric = running_coral_metric / float(len_data)
    topo_metric = running_topological_metric / float(len_data)

    if args.wandb:
        wandb.log({'Train Loss': loss.item(), 'Train Accuracy': coral_metric})
    loss = running_loss / float(len_data)
    t_loss = running_topo_loss / float(len_data)

    return loss, coral_metric, topo_metric, t_loss


def trainer_coral(args, model, snapshot_path):
    from datasets.dataset_coral import Coral_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    batch_size = args.batch_size * args.n_gpu

    from sklearn.model_selection import ShuffleSplit
    from torch.utils.data import Subset
    from torchvision import transforms
    from torch.utils.data import DataLoader
    db_train = Coral_dataset(base_dir=f"{args.root_path}/training_data/snippets", list_dir=args.list_dir, split="train",
                             transform=transforms.Compose(
                                 [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    db_val = Coral_dataset(base_dir=f"{args.root_path}/training_data/snippets", list_dir=args.list_dir, split="val", transform=None)

    # Split data into train/validation set
    sss = ShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    indices = range(len(db_train))

    for train_index, val_index in sss.split(indices):
        pass

    train_ds = Subset(db_train, train_index)
    val_ds = Subset(db_val, val_index)
    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

    # trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True)
    # valloader = DataLoader(db_val, batch_size=batch_size, shuffle=True)

    dice_loss = DiceLoss(num_classes)
    topo_loss = TopoLoss(num_classes, args.cuda)
    # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, verbose=1)
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations \n".format(len(trainloader), max_iterations))

    loss_history = {
        "train": [],
        "val": []}

    metric_history = {
        "train": [],
        "val": []}

    topo_history = {
        "train": [],
        "val": []
    }
    topo_loss_history = {
        "train": [],
        "val": []
    }

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    if args.wandb:
        wandb.init(project='Coral Debug')
        wandb.watch(model, log='all', log_freq=1, log_graph=True)

    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        model.train()
        train_loss, train_dice_metric, train_topo_metric, train_topo_loss = loss_epoch(trainloader, model, dice_loss,
                                                                                       topo_loss, args, epoch_num,
                                                                                       optimizer)
        logging.info(
            'epoch %d : train_loss : %f, train_dice_acc: %f, train_topo_loss %f, train_topological_accuracy %f' % (
                epoch_num, train_loss, train_dice_metric, train_topo_loss, train_topo_metric))
        train_metrics = train_dice_metric
        loss_history["train"].append(train_loss.detach().cpu().numpy())
        metric_history["train"].append(train_metrics.detach().cpu().numpy())
        topo_loss_history['train'].append(train_topo_loss.detach().cpu().numpy())
        topo_history['train'].append(train_topo_metric.detach().cpu().numpy())

        #
        # for tag, parm in model.named_parameters():
        #     writer.add_histogram(tag, parm.grad.data.cpu().numpy(), epoch_num)

        model.eval()
        with torch.no_grad():
            val_loss, val_dice_metric, val_topo_metric, val_topo_loss = loss_epoch(valloader, model, dice_loss,
                                                                                   topo_loss, args, epoch_num)
        logging.info('epoch %d : val_loss : %f, val_dice_acc: %f,val_topo_loss %f, val_topological_accuracy %f' % (
            epoch_num, val_loss, val_dice_metric, val_topo_loss, val_topo_metric))
        val_metrics = val_dice_metric
        loss_history["val"].append(val_loss.detach().cpu().numpy())
        metric_history["val"].append(val_metrics.detach().cpu().numpy())
        topo_loss_history['val'].append(val_topo_loss.detach().cpu().numpy())
        topo_history['val'].append(val_topo_metric.detach().cpu().numpy())

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            save_mode_path = os.path.join(snapshot_path, 'best_weights.pth')
            torch.save(model.state_dict(), save_mode_path)

        current_lr = get_lr(optimizer)
        lr_scheduler.step(val_loss)
        if current_lr != get_lr(optimizer):
            print("Loading best model weights!")
            model.load_state_dict(best_model_wts)

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()

        vis_epoch = True
        if vis_epoch or epoch_num % 100 == 0:
            if not os.path.exists(f'{snapshot_path}/test_log'):
                os.makedirs(f'{snapshot_path}/test_log')

            # plot loss progress
            fig1 = plt.figure(1)
            plt.title("Train-Val Loss")
            plt.plot(range(1, epoch_num + 2), loss_history["train"], label="train" if epoch_num == 0 else "",
                     color='cornflowerblue')
            plt.plot(range(1, epoch_num + 2), loss_history["val"], label="val" if epoch_num == 0 else "",
                     color='orange')
            plt.ylabel("Loss")
            plt.xlabel("Training Epochs")
            plt.legend()
            fig1.savefig(f'{snapshot_path}/test_log/Train-Val Loss.png')

            # plot accuracy progress
            fig2 = plt.figure(2)
            plt.title("Train-Val Accuracy")
            plt.plot(range(1, epoch_num + 2), metric_history["train"], label="train" if epoch_num == 0 else "",
                     color='cornflowerblue')
            plt.plot(range(1, epoch_num + 2), metric_history["val"], label="val" if epoch_num == 0 else "",
                     color='orange')
            plt.ylabel("Accuracy")
            plt.xlabel("Training Epochs")
            plt.legend()
            fig2.savefig(f'{snapshot_path}/test_log/Train-Val Accuracy.png')

            # plot topological progress
            fig3 = plt.figure(3)
            plt.title("Train-Val Topological Accuracy")
            plt.plot(range(1, epoch_num + 2), topo_history["train"], label="train" if epoch_num == 0 else "",
                     color='cornflowerblue')
            plt.plot(range(1, epoch_num + 2), topo_history["val"], label="val" if epoch_num == 0 else "",
                     color='orange')
            plt.ylabel("Accuracy")
            plt.xlabel("Training Epochs")
            plt.legend()
            fig3.savefig(f'{snapshot_path}/test_log/Train-Val Topo Accuracy.png')

            # Plot topological loss * lambda
            fig4 = plt.figure(4)
            plt.title(f"Train-Val Topological Loss * Lambda={args.topo_lambda}")
            plt.plot(range(1, epoch_num + 2), topo_loss_history["train"], label="train" if epoch_num == 0 else "",
                     color='cornflowerblue')
            plt.plot(range(1, epoch_num + 2), topo_loss_history["val"], label="val" if epoch_num == 0 else "",
                     color='orange')
            plt.ylabel("Loss")
            plt.xlabel("Training Epochs")
            plt.legend()
            fig4.savefig(f'{snapshot_path}/test_log/Train-Val Topo Loss.png')

        # save history to pickle
        loss_file = open(f'{snapshot_path}/test_log/loss.pkl', "wb")
        pickle.dump(loss_history, loss_file)
        loss_file.close()

        metric_file = open(f'{snapshot_path}/test_log/metric.pkl', "wb")
        pickle.dump(metric_history, metric_file)
        metric_file.close()

        topo_file = open(f'{snapshot_path}/test_log/topo.pkl', "wb")
        pickle.dump(topo_history, topo_file)
        metric_file.close()

    return "Training Finished!"
