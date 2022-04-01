import torch
import torch.nn as nn

import numpy as np
from miscc.config import cfg

from GlobalAttention import func_attention


# ##################Loss for matching text-image###################
def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim.
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def sent_loss(cnn_code, rnn_code, labels, class_ids,
                   batch_size, eps=1e-8):
    # ### Mask mis-match samples  ###
    # that come from the same class as the real sample ###
    masks = []
    if class_ids is not None:
        for i in range(batch_size):
            mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        masks = np.concatenate(masks, 0)
        # masks: batch_size x batch_size
        masks = torch.ByteTensor(masks)
        if cfg.CUDA:
            masks = masks.cuda()
    # --> seq_len x batch_size x nef
    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)
    # cnn_code_norm / rnn_code_norm: seq_len x batch_size x 1
    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)
    # scores* / norm*: seq_len x batch_size x batch_size
    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * cfg.TRAIN.SMOOTH.GAMMA3
    # --> batch_size x batch_size
    scores0 = scores0.squeeze()
    if class_ids is not None:
        scores0.data.masked_fill_(masks, -float('inf'))
    scores1 = scores0.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(scores0, labels)
        loss1 = nn.CrossEntropyLoss()(scores1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1


def words_loss(img_features, words_emb, labels, cap_lens, class_ids, batch_size):
    """
        words_emb(query): batch x nef x seq_len
        img_features(context): batch x nef x 17 x 17
    """
    masks = []
    att_maps = []
    similarities = []
    cap_lens = cap_lens.data.tolist()
    for i in range(batch_size):
        if class_ids is not None:
            mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        # Get the i-th text description
        words_num = cap_lens[i]  # + 2
        # -> 1 x nef x words_num
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()
        # -> batch_size x nef x words_num
        word = word.repeat(batch_size, 1, 1)
        # batch x nef x 14 * 14
        context = img_features
        """
            word(query): batch x nef x words_num
            context: batch x nef x 14 * 14
            weiContext: batch x nef x words_num
            attn: batch x words_num x 14 * 14
        """
        weiContext, attn = func_attention(word, context, cfg.TRAIN.SMOOTH.GAMMA1)
        att_maps.append(attn[i].unsqueeze(0).contiguous())
        # --> batch_size x words_num x nef
        word = word.transpose(1, 2).contiguous()
        weiContext = weiContext.transpose(1, 2).contiguous()
        # --> batch_size*words_num x nef
        word = word.view(batch_size * words_num, -1)
        weiContext = weiContext.view(batch_size * words_num, -1)
        #
        # -->batch_size*words_num
        row_sim = cosine_similarity(word, weiContext)
        # --> batch_size x words_num
        row_sim = row_sim.view(batch_size, words_num)

        # Eq. (10)
        row_sim.mul_(cfg.TRAIN.SMOOTH.GAMMA2).exp_()
        row_sim = row_sim.sum(dim=1, keepdim=True)
        row_sim = torch.log(row_sim)
        # --> 1 x batch_size
        # similarities(i, j): the similarity between the i-th image and the j-th text description
        similarities.append(row_sim)

    # batch_size x batch_size
    similarities = torch.cat(similarities, 1)
    if class_ids is not None:
        masks = np.concatenate(masks, 0)
        # masks: batch_size x batch_size
        masks = torch.ByteTensor(masks)
        if cfg.CUDA:
            masks = masks.cuda()
    similarities = similarities * cfg.TRAIN.SMOOTH.GAMMA3
    if class_ids is not None:
        similarities.data.masked_fill_(masks, -float('inf'))
    similarities1 = similarities.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(similarities, labels)
        loss1 = nn.CrossEntropyLoss()(similarities1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1, att_maps


# ##################Loss for G and Ds##############################
def discriminator_loss(netDs, real_imgs, fake_imgs, conditions,
                       real_labels, fake_labels, words_embs):
    errD = []
    # Forward layer1 for unconditional loss
    real_features1 = netDs[0](real_imgs[0])
    fake_features1 = netDs[0](fake_imgs[0].detach())
    real_logits1 = netDs[0].UNCOND_DNET(real_features1)
    fake_logits1 = netDs[0].UNCOND_DNET(fake_features1)
    real_errD1 = nn.BCELoss()(real_logits1, real_labels)
    fake_errD1 = nn.BCELoss()(fake_logits1, fake_labels)
    errD1 = (real_errD1 + fake_errD1) / 2.
    errD.append(errD1)

    # Forward layer2 for sentence conditional loss
    real_features2 = netDs[1](real_imgs[1])
    fake_features2 = netDs[1](fake_imgs[1].detach())
    cond_real_logits2 = netDs[1].COND_DNET(real_features2, conditions)
    cond_real_errD2 = nn.BCELoss()(cond_real_logits2, real_labels)
    cond_fake_logits2 = netDs[1].COND_DNET(fake_features2, conditions)
    cond_fake_errD2 = nn.BCELoss()(cond_fake_logits2, fake_labels)
    #
    batch_size = real_features2.size(0)
    cond_wrong_logits2 = netDs[1].COND_DNET(real_features2[:(batch_size - 1)], conditions[1:batch_size])
    cond_wrong_errD2 = nn.BCELoss()(cond_wrong_logits2, fake_labels[1:batch_size])
    errD2 = cond_real_errD2 + (cond_fake_errD2 + cond_wrong_errD2) / 2.
    errD.append(errD2)

    # Forward layer3 for word conditional loss
    condition_word = torch.mean(words_embs, dim=2)
    real_features3 = netDs[2](real_imgs[2])
    fake_features3 = netDs[2](fake_imgs[2].detach())
    cond_real_logits3 = netDs[2].COND_DNET(real_features3, condition_word)
    cond_real_errD3 = nn.BCELoss()(cond_real_logits3, real_labels)
    cond_fake_logits3 = netDs[2].COND_DNET(fake_features3, condition_word)
    cond_fake_errD3 = nn.BCELoss()(cond_fake_logits3, fake_labels)
    #
    batch_size = real_features3.size(0)
    cond_wrong_logits3 = netDs[2].COND_DNET(real_features3[:(batch_size - 1)], condition_word[1:batch_size])
    cond_wrong_errD3 = nn.BCELoss()(cond_wrong_logits3, fake_labels[1:batch_size])
    errD3 = cond_real_errD3 + (cond_fake_errD3 + cond_wrong_errD3) / 2.
    errD.append(errD3)

    return errD


def generator_loss(netsD, ITM, fake_imgs, real_labels,
                   words_embs, sent_emb, match_labels,
                   cap_lens, class_ids):
    numDs = len(netsD)
    batch_size = real_labels.size(0)
    logs = ''
    # Forward
    errG_total = 0
    clip_loss = 0

    # Forward layer1 for unconditional loss
    features1 = netsD[0](fake_imgs[0])
    logits1 = netsD[0].UNCOND_DNET(features1)
    errG1 = nn.BCELoss()(logits1, real_labels)
    s = 'g_uncond_loss'
    g_loss1 = errG1
    logs += '%s%d: %.2f ' % (s, 1, g_loss1.item())
    errG_total += g_loss1

    # Forward layer2 for sentence level conditional loss
    features2 = netsD[1](fake_imgs[1])
    cond_logits2 = netsD[1].COND_DNET(features2, sent_emb)
    cond_errG2 = nn.BCELoss()(cond_logits2, real_labels)
    # logits2 = netsD[1].UNCOND_DNET(features2)
    # errG2 = nn.BCELoss()(logits2, real_labels)
    s = 'g_cond_loss'
    g_loss2 = cond_errG2
    logs += '%s%d: %.2f ' % (s, 2, g_loss2.item())
    errG_total += g_loss2

    # Forward layer3 for word level conditional loss
    features3 = netsD[2](fake_imgs[2])
    condition = torch.mean(words_embs, dim=2)
    cond_logits3 = netsD[2].COND_DNET(features3, condition)
    cond_errG3 = nn.BCELoss()(cond_logits3, real_labels)
    # logits3 = netsD[2].UNCOND_DNET(features3)
    # errG3 = nn.BCELoss()(logits3, real_labels)
    s = 'g_cond_loss'
    g_loss3 = cond_errG3
    logs += '%s%d: %.2f ' % (s, 3, g_loss3.item())
    errG_total += g_loss3

    # ITM loss
    _, cnn_code_1 = ITM.image_encode(fake_imgs[1])
    s_loss0_1, s_loss1_1 = sent_loss(cnn_code_1, sent_emb,
                                          match_labels, class_ids, batch_size)
    s_loss_1 = (s_loss0_1 + s_loss1_1) * cfg.TRAIN.SMOOTH.LAMBDA2
    errG_total += s_loss_1
    logs += 's_loss_1: %.2f ' % (s_loss_1.item())
    # words_features: batch_size x nef x 14 x 14
    # sent_code: batch_size x nef
    region_features, cnn_code = ITM.image_encode(fake_imgs[-1])
    w_loss0, w_loss1, attn_maps = words_loss(region_features,
                                                  words_embs, match_labels, cap_lens, class_ids, batch_size)
    w_loss = (w_loss0 + w_loss1) * \
             cfg.TRAIN.SMOOTH.LAMBDA2

    # err_words = err_words + w_loss.data[0]
    s_loss0, s_loss1 = sent_loss(cnn_code, sent_emb,
                                      match_labels, class_ids, batch_size)
    s_loss = (s_loss0 + s_loss1) * \
             cfg.TRAIN.SMOOTH.LAMBDA
    errG_total += w_loss + s_loss
    clip_loss = w_loss + s_loss
    logs += 'w_loss: %.2f s_V3_loss: %.2f clip_loss: %.2f ' \
            % (w_loss.item(), s_loss.item(), clip_loss.item())
    return errG_total, logs, clip_loss


##################################################################
def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD
