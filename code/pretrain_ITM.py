from __future__ import print_function

from miscc.utils import mkdir_p
from miscc.utils import build_super_images_ITM
from miscc.losses import sent_loss, words_loss
from miscc.config import cfg, cfg_from_file

from datasets import TextDataset
from datasets import prepare_data

from model import ITM_MODEL

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


UPDATE_INTERVAL = 200


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/DAMSM/bird.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=-1)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


def train(dataloader, ITM, batch_size,
          labels, optimizer, epoch, ixtoword, image_dir):
    ITM.MLP.train()
    s_total_loss0 = 0
    s_total_loss1 = 0
    w_total_loss0 = 0
    w_total_loss1 = 0
    count = (epoch + 1) * len(dataloader)
    start_time = time.time()
    for step, data in enumerate(dataloader, 0):
        # print('step', step)
        ITM.zero_grad()

        imgs, captions, cap_lens, \
            class_ids, keys = prepare_data(data)
        # words_features: batch_size x nef x 17 x 17
        # sent_code: batch_size x nef
        words_features, sent_code = ITM.image_encode(imgs[-1])

        # --> batch_size x nef x 14*14
        nef, att_sze = words_features.size(1), words_features.size(2)

        # words_emb: batch_size x nef x seq_len
        # sent_emb: batch_size x nef
        words_emb, sent_emb = ITM.sent_encode(captions)
        w_loss0, w_loss1, attn_maps = words_loss(words_features,
                                                 words_emb, labels, cap_lens, class_ids, batch_size)
        w_total_loss0 += w_loss0.data
        w_total_loss1 += w_loss1.data
        loss = w_loss0 + w_loss1

        s_loss0, s_loss1 = \
            sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
        loss += s_loss0 + s_loss1
        s_total_loss0 += s_loss0.data
        s_total_loss1 += s_loss1.data
        loss.backward()
        optimizer.step()

        if step % UPDATE_INTERVAL == 0:
            count = epoch * len(dataloader) + step

            s_cur_loss0 = s_total_loss0 / UPDATE_INTERVAL
            s_cur_loss1 = s_total_loss1 / UPDATE_INTERVAL
            w_cur_loss0 = w_total_loss0.item() / UPDATE_INTERVAL
            w_cur_loss1 = w_total_loss1.item() / UPDATE_INTERVAL

            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                  's_loss {:5.2f} {:5.2f} | '
                  'w_loss {:5.2f} {:5.2f}'
                  .format(epoch, step, len(dataloader),
                          elapsed * 1000. / UPDATE_INTERVAL,
                          s_cur_loss0, s_cur_loss1,
                          w_cur_loss0, w_cur_loss1))
            s_total_loss0 = 0
            s_total_loss1 = 0
            w_total_loss0 = 0
            w_total_loss1 = 0
            start_time = time.time()
            # attention Maps
            exchaged_image = nn.functional.interpolate(imgs[-1], size=(224, 224), mode='bilinear', align_corners=False)
            img_set, _ = \
                build_super_images_ITM(exchaged_image.cpu(), captions,
                                        ixtoword, cap_lens, attn_maps, att_sze)
            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s/attention_maps%d.png' % (image_dir, step)
                im.save(fullpath)
    return count


def evaluate(dataloader, ITM, batch_size):
    ITM.eval()
    s_total_loss = 0
    w_total_loss = 0
    for step, data in enumerate(dataloader, 0):
        real_imgs, captions, cap_lens, \
                class_ids, keys = prepare_data(data)

        words_features, sent_code = ITM.image_encode(real_imgs[-1])
        words_emb, sent_emb = ITM.sent_encode(captions)

        w_loss0, w_loss1, attn = words_loss(words_features, words_emb, labels,
                                            cap_lens, class_ids, batch_size)
        w_total_loss += (w_loss0 + w_loss1).data
        #
        s_loss0, s_loss1 = \
            sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
        s_total_loss += (s_loss0 + s_loss1).data
        if step == 50:
            break
    s_cur_loss = s_total_loss / step
    w_cur_loss = w_total_loss.item() / step
    return s_cur_loss, w_cur_loss


def build_models():
    # build model ############################################################
    ITM = ITM_MODEL(cfg.TEXT.EMBEDDING_DIM)
    labels = Variable(torch.LongTensor(range(batch_size)))
    start_epoch = 0
    if cfg.TRAIN.NET_E != '':
        istart = cfg.TRAIN.NET_E.rfind('_') + 8
        iend = cfg.TRAIN.NET_E.rfind('.')
        start_epoch = cfg.TRAIN.NET_E[istart:iend]
        start_epoch = int(start_epoch) + 1
        print('start_epoch', start_epoch)
    if cfg.CUDA:
        ITM = ITM.cuda()
        labels = labels.cuda()

    return ITM, labels, start_epoch


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id == -1:
        cfg.CUDA = False
    else:
        cfg.GPU_ID = args.gpu_id

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    ##########################################################################
    # now = datetime.datetime.now(dateutil.tz.tzlocal())
    # timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    # output_dir = '../output/%s_%s_%s' % \
    #     (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)
    output_dir = '../output/ITM_COCO'

    model_dir = os.path.join(output_dir, 'Model')
    image_dir = os.path.join(output_dir, 'Image')
    mkdir_p(model_dir)
    mkdir_p(image_dir)

    # torch.cuda.set_device(cfg.GPU_ID)
    cudnn.benchmark = True

    # Get data loader ##################################################
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM-1))
    batch_size = cfg.TRAIN.BATCH_SIZE

    # Build Model ###########################################################
    ITM, labels, start_epoch = build_models()

    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    dataset = TextDataset(cfg.DATA_DIR, 'train',
                          base_size=cfg.TREE.BASE_SIZE, transform=image_transform)
    print(dataset.n_words, dataset.embeddings_num)
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, drop_last=True,
        shuffle=True, num_workers=int(cfg.WORKERS))

    # # validation data #
    dataset_val = TextDataset(cfg.DATA_DIR, 'test',
                              base_size=cfg.TREE.BASE_SIZE, transform=image_transform)
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=batch_size, drop_last=True,
        shuffle=True, num_workers=int(cfg.WORKERS))

    # Train ##############################################################
    para = []
    for v in ITM.parameters():
        if v.requires_grad:
            para.append(v)
    for name, parms in ITM.named_parameters():
        print('-->name:', name, '-->grad_requirs:', parms.requires_grad)
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        lr = cfg.TRAIN.ENCODER_LR
        for epoch in range(start_epoch, cfg.TRAIN.MAX_EPOCH):
            optimizer = optim.Adam(para, lr=lr, betas=(0.5, 0.999))
            epoch_start_time = time.time()
            count = train(dataloader, ITM,
                          batch_size, labels, optimizer, epoch,
                          dataset.ixtoword, image_dir)
            print('-' * 89)
            if len(dataloader_val) > 0:
                s_loss, w_loss = evaluate(dataloader_val, ITM, batch_size)
                print('| end epoch {:3d} | valid loss '
                      '{:5.2f} {:5.2f} | lr {:.5f}|'
                      .format(epoch, s_loss, w_loss, lr))
            print('-' * 89)
            if lr > cfg.TRAIN.ENCODER_LR/10.:
                lr *= 0.98

            if (epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0 or
                epoch == cfg.TRAIN.MAX_EPOCH - 1):
                torch.save(ITM.state_dict(),
                           '%s/ITM_model%d.pth' % (model_dir, epoch))
                print('Save models.')
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
