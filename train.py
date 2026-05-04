import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np, argparse, time
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import IEMOCAPDataset, MELDDataset
from model import MaskedNLLLoss, MaskedKLDivLoss, Transformer_Based_Model
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
import pickle as pk
import datetime

def compute_ulgm_lambda(epoch, lambda_min, lambda_max, e_delay, e_ramp):
    if epoch < e_delay:
        return lambda_min
    if e_ramp <= 0:
        return lambda_max
    progress = float(epoch - e_delay + 1) / float(e_ramp)
    progress = max(0.0, min(1.0, progress))
    return lambda_min + (lambda_max - lambda_min) * progress


def build_ulgm_alphas(args):
    raw = {
        't': args.ulgm_alpha_t,
        'a': args.ulgm_alpha_a,
        'v': args.ulgm_alpha_v,
        'r': args.ulgm_alpha_r,
    }
    used_keys = ['t', 'a', 'v', 'r'] if args.use_rppg else ['t', 'a', 'v']
    total = sum(raw[k] for k in used_keys)
    if args.ulgm_normalize_alpha:
        norm = {}
        for k in raw:
            norm[k] = raw[k] / total if (k in used_keys and total > 0) else 0.0
        return norm
    return raw


def validate_ulgm_args(args):
    alpha_pairs = [
        ('ulgm_alpha_t', args.ulgm_alpha_t),
        ('ulgm_alpha_a', args.ulgm_alpha_a),
        ('ulgm_alpha_v', args.ulgm_alpha_v),
        ('ulgm_alpha_r', args.ulgm_alpha_r),
    ]
    for name, value in alpha_pairs:
        if value < 0:
            raise ValueError(f'{name} must be >= 0, got {value}')

    if args.ulgm_lambda_min < 0 or args.ulgm_lambda_max < 0:
        raise ValueError('ulgm_lambda_min and ulgm_lambda_max must be >= 0')
    if args.ulgm_lambda_min > args.ulgm_lambda_max:
        raise ValueError('ulgm_lambda_min must be <= ulgm_lambda_max')
    if args.ulgm_e_delay < 0 or args.ulgm_e_ramp < 0:
        raise ValueError('ulgm_e_delay and ulgm_e_ramp must be >= 0')

    used_keys = ['ulgm_alpha_t', 'ulgm_alpha_a', 'ulgm_alpha_v']
    used_vals = [args.ulgm_alpha_t, args.ulgm_alpha_a, args.ulgm_alpha_v]
    if args.use_rppg:
        used_keys.append('ulgm_alpha_r')
        used_vals.append(args.ulgm_alpha_r)
    if args.ulgm_normalize_alpha and sum(used_vals) <= 0:
        raise ValueError('sum of ULGM alphas must be > 0 when ulgm_normalize_alpha is enabled')
    if not args.ulgm_normalize_alpha and sum(used_vals) <= 0:
        raise ValueError('sum of active ULGM alphas must be > 0')


def get_train_valid_sampler(trainset, valid=0.1, dataset='MELD'):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

def get_MELD_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = MELDDataset('data/meld_multimodal_features.pkl')
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid, 'MELD')
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = MELDDataset('data/meld_multimodal_features.pkl', train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader


def get_IEMOCAP_loaders(iemocap_pkl_path='data/iemocap_multimodal_features.pkl',
                        session_prefixes=None,
                        use_rppg=False,
                        rppg_npz_path='data/iemocap_rppg_features_ses01_v3.npz',
                        batch_size=32, valid=0.1, test=0.1, split_seed=42, split_max_tries=200,
                        n_classes=6, num_workers=0, pin_memory=False,
                        target_label_map=None):
    # Build on the same filtered pool so all modalities (including rPPG) share identical split.
    fullset = IEMOCAPDataset(path=iemocap_pkl_path, split='all', session_prefixes=session_prefixes,
                              use_rppg=use_rppg, rppg_npz_path=rppg_npz_path,
                              target_label_map=target_label_map)
    size = len(fullset)
    all_idx = list(range(size))
    valid_size = int(valid * size)
    test_size = int(test * size)
    if test > 0 and size > 0 and test_size == 0:
        test_size = 1
    if valid > 0 and size > 1 and valid_size == 0:
        valid_size = 1
    if valid_size + test_size >= size and size > 0:
        max_reserved = max(size - 1, 0)
        reserved = min(valid_size + test_size, max_reserved)
        if reserved < (valid_size + test_size):
            overflow = (valid_size + test_size) - reserved
            trim_valid = min(valid_size, overflow)
            valid_size -= trim_valid
            overflow -= trim_valid
            if overflow > 0:
                test_size = max(0, test_size - overflow)

    idx2label_set = {i: set(fullset.videoLabels[fullset.keys[i]]) for i in all_idx}

    def count_covered_classes(indices):
        c = set()
        for ii in indices:
            c.update(idx2label_set[ii])
        return len(c)

    best = None
    for t in range(split_max_tries):
        rng = np.random.RandomState(split_seed + t)
        idx = all_idx.copy()
        rng.shuffle(idx)

        test_idx = idx[:test_size]
        valid_idx = idx[test_size:test_size + valid_size]
        train_idx = idx[test_size + valid_size:]

        c_train = count_covered_classes(train_idx)
        c_valid = count_covered_classes(valid_idx)
        c_test = count_covered_classes(test_idx)
        score = (min(c_train, c_valid, c_test), c_train + c_valid + c_test)

        if best is None or score > best['score']:
            best = {
                'seed': split_seed + t,
                'train_idx': train_idx,
                'valid_idx': valid_idx,
                'test_idx': test_idx,
                'c_train': c_train,
                'c_valid': c_valid,
                'c_test': c_test,
                'score': score,
            }
        if c_train >= n_classes and c_valid >= n_classes and c_test >= n_classes:
            break

    train_idx = best['train_idx']
    valid_idx = best['valid_idx']
    test_idx = best['test_idx']

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    train_loader = DataLoader(fullset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=fullset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(fullset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=fullset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    test_loader = DataLoader(fullset,
                             batch_size=batch_size,
                             sampler=test_sampler,
                             collate_fn=fullset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    print('IEMOCAP split size -> train: {}, valid: {}, test: {}'.format(
        len(train_idx), len(valid_idx), len(test_idx)))
    print('IEMOCAP split class coverage -> train: {}, valid: {}, test: {}, seed: {}'.format(
        best['c_train'], best['c_valid'], best['c_test'], best['seed']))
    return train_loader, valid_loader, test_loader


def build_dynamic_class_weights(dataset, train_indices, n_classes):
    counts = np.zeros(n_classes, dtype=np.float64)
    for idx in train_indices:
        vid = dataset.keys[idx]
        labels = dataset.videoLabels[vid]
        for y in labels:
            if 0 <= y < n_classes:
                counts[y] += 1.0
    counts = np.maximum(counts, 1.0)
    total = np.sum(counts)
    weights = total / (n_classes * counts)
    return torch.FloatTensor(weights)


def count_iemocap_utterance_class_counts(dataset, train_indices, n_classes):
    counts = np.zeros(n_classes, dtype=np.float64)
    for idx in train_indices:
        vid = dataset.keys[idx]
        for y in dataset.videoLabels[vid]:
            if 0 <= int(y) < n_classes:
                counts[int(y)] += 1.0
    counts = np.maximum(counts, 1.0)
    return counts


def build_auto_class_weights_paper(counts, clip_min=0.4, clip_max=2.5):
    counts = np.asarray(counts, dtype=np.float64)
    n_classes = counts.shape[0]
    total = float(np.sum(counts))
    w_init = total / (n_classes * counts)
    w = w_init / (np.mean(w_init) + 1e-12)
    w = np.clip(w, clip_min, clip_max)
    w_hat = w / (np.mean(w) + 1e-12)
    return torch.FloatTensor(w_hat)


def compute_beta_schedule(epoch, beta_init, beta_target, e_delay, e_warmup):
    if epoch < e_delay:
        return beta_init
    if e_warmup <= 0:
        return beta_target
    if epoch < e_delay + e_warmup:
        t = float(epoch - e_delay + 1) / float(e_warmup)
        t = max(0.0, min(1.0, t))
        return beta_init + t * (beta_target - beta_init)
    return beta_target


def compute_gamma2_schedule(epoch, gamma2, gamma2_start, warmup_epochs):
    """Linear ramp of outer ULGM weight γ₂ from gamma2_start to gamma2 over warmup_epochs."""
    if warmup_epochs <= 0:
        return gamma2
    if warmup_epochs == 1:
        return gamma2
    if epoch >= warmup_epochs:
        return gamma2
    denom = float(max(warmup_epochs - 1, 1))
    t = float(epoch) / denom
    t = max(0.0, min(1.0, t))
    return gamma2_start + t * (gamma2 - gamma2_start)


def compute_happy_boost(epoch, gamma_max, e_boost):
    if e_boost <= 0:
        return 1.0
    val = gamma_max - (float(epoch) / float(e_boost)) * (gamma_max - 1.0)
    return float(max(1.0, val))


def build_class_boost_weights(labels_flat, happy_class_idx, happy_boost,
                              sad_class_idx, sad_boost,
                              neutral_class_idx, neutral_boost):
    labels_flat = labels_flat.long()
    w = torch.ones_like(labels_flat, dtype=torch.float32, device=labels_flat.device)
    if happy_boost > 1.0:
        w = w + (float(happy_boost) - 1.0) * (labels_flat == int(happy_class_idx)).float()
    if sad_boost > 1.0:
        w = w + (float(sad_boost) - 1.0) * (labels_flat == int(sad_class_idx)).float()
    if neutral_boost > 1.0:
        w = w + (float(neutral_boost) - 1.0) * (labels_flat == int(neutral_class_idx)).float()
    return w


def masked_mean(x, mask):
    mask = mask.view(-1).float()
    denom = torch.sum(mask) + 1e-8
    return torch.sum(x.view(-1) * mask) / denom


def build_pseudo_targets_for_modality(log_prob_m, labels_flat, n_classes, happy_class,
                                      tau_conf, omega_true_major, omega_true_happy):
    # log_prob_m: [N, C] (flattened time*batch)
    probs_m = torch.exp(log_prob_m)
    conf_m, pred_m = torch.max(probs_m, dim=-1)

    y = labels_flat.long()
    y_onehot = F.one_hot(y, num_classes=n_classes).float()

    pred_onehot = F.one_hot(pred_m, num_classes=n_classes).float()

    is_happy = (y == int(happy_class))
    high_conf = conf_m >= float(tau_conf)
    use_mix = is_happy & high_conf

    omega = torch.where(
        is_happy,
        torch.full_like(y, float(omega_true_happy), dtype=torch.float32, device=log_prob_m.device),
        torch.full_like(y, float(omega_true_major), dtype=torch.float32, device=log_prob_m.device),
    )

    omega = torch.where(use_mix, omega, torch.ones_like(omega))
    target = omega.unsqueeze(-1) * y_onehot + (1.0 - omega.unsqueeze(-1)) * pred_onehot
    return target, use_mix


def unimodal_pseudo_loss_per_pos(log_prob_m, target_dist, class_weights_vec, labels_flat):
    # cross-entropy with soft targets: -sum_k target_k log p_k  (per position)
    ce = -(target_dist * log_prob_m).sum(dim=-1)
    if class_weights_vec is not None:
        w = class_weights_vec[labels_flat]
        ce = ce * w
    return ce


def train_or_eval_model(model, loss_function, kl_loss, dataloader, epoch, optimizer=None, train=False,
                        gamma_1=1.0, gamma_2=1.0, gamma_3=1.0, use_rppg=False,
                        beta_e=1.0, ulgm_alphas=None,
                        n_classes=6, happy_class=1, tau_conf=0.8,
                        omega_true_major=0.3, omega_true_happy=0.7,
                        gamma_boost=1.0, happy_class_idx=1,
                        sad_class_idx=1, sad_boost=1.0,
                        neutral_class_idx=2, neutral_boost=1.0,
                        class_boost_lambda=0.0,
                        alpha_gate=0.01,
                        use_pseudo_ulgm=True, use_aux_losses=True):
    losses, preds, labels, masks = [], [], [], []
    skipped_non_finite = 0

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()
        
        if ulgm_alphas is None:
            ulgm_alphas = {'t': 1.0, 'a': 1.0, 'v': 1.0, 'r': 1.0}

        if use_rppg:
            textf, visuf, acouf, rppgf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        else:
            textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        qmask = qmask.permute(1, 0, 2)
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

        if use_rppg:
            if use_aux_losses:
                log_prob1, log_prob2, log_prob3, log_prob4, all_log_prob, all_prob, \
                kl_log_prob1, kl_log_prob2, kl_log_prob3, kl_log_prob4, kl_all_prob, \
                gate_entropy = model(textf, visuf, acouf, umask, qmask, lengths, rppgf=rppgf, return_aux_losses=True)
            else:
                log_prob1, log_prob2, log_prob3, log_prob4, all_log_prob, all_prob, \
                kl_log_prob1, kl_log_prob2, kl_log_prob3, kl_log_prob4, kl_all_prob = model(textf, visuf, acouf, umask, qmask, lengths, rppgf=rppgf)
        else:
            if use_aux_losses:
                log_prob1, log_prob2, log_prob3, all_log_prob, all_prob, \
                kl_log_prob1, kl_log_prob2, kl_log_prob3, kl_all_prob, \
                gate_entropy = model(textf, visuf, acouf, umask, qmask, lengths, return_aux_losses=True)
            else:
                log_prob1, log_prob2, log_prob3, all_log_prob, all_prob, \
                kl_log_prob1, kl_log_prob2, kl_log_prob3, kl_all_prob = model(textf, visuf, acouf, umask, qmask, lengths)
        
        lp_1 = log_prob1.view(-1, log_prob1.size()[2])
        lp_2 = log_prob2.view(-1, log_prob2.size()[2])
        lp_3 = log_prob3.view(-1, log_prob3.size()[2])
        lp_all = all_log_prob.view(-1, all_log_prob.size()[2])
        labels_ = label.view(-1)
        class_boost_w = build_class_boost_weights(
            labels_, happy_class_idx, gamma_boost,
            sad_class_idx, sad_boost,
            neutral_class_idx, neutral_boost
        )

        kl_lp_1 = kl_log_prob1.view(-1, kl_log_prob1.size()[2])
        kl_lp_2 = kl_log_prob2.view(-1, kl_log_prob2.size()[2])
        kl_lp_3 = kl_log_prob3.view(-1, kl_log_prob3.size()[2])
        kl_p_all = kl_all_prob.view(-1, kl_all_prob.size()[2])
        if use_rppg:
            lp_4 = log_prob4.view(-1, log_prob4.size()[2])
            kl_lp_4 = kl_log_prob4.view(-1, kl_log_prob4.size()[2])
            if use_pseudo_ulgm:
                cw = getattr(loss_function, 'weight', None)
                tgt1, _ = build_pseudo_targets_for_modality(lp_1, labels_, n_classes, happy_class, tau_conf, omega_true_major, omega_true_happy)
                tgt2, _ = build_pseudo_targets_for_modality(lp_2, labels_, n_classes, happy_class, tau_conf, omega_true_major, omega_true_happy)
                tgt3, _ = build_pseudo_targets_for_modality(lp_3, labels_, n_classes, happy_class, tau_conf, omega_true_major, omega_true_happy)
                tgt4, _ = build_pseudo_targets_for_modality(lp_4, labels_, n_classes, happy_class, tau_conf, omega_true_major, omega_true_happy)

                l1 = unimodal_pseudo_loss_per_pos(lp_1, tgt1, cw, labels_)
                l2 = unimodal_pseudo_loss_per_pos(lp_2, tgt2, cw, labels_)
                l3 = unimodal_pseudo_loss_per_pos(lp_3, tgt3, cw, labels_)
                l4 = unimodal_pseudo_loss_per_pos(lp_4, tgt4, cw, labels_)

                l1 = l1 * class_boost_w
                l2 = l2 * class_boost_w
                l3 = l3 * class_boost_w
                l4 = l4 * class_boost_w

                ulgm_loss = ulgm_alphas['t'] * masked_mean(l1, umask) + \
                            ulgm_alphas['v'] * masked_mean(l2, umask) + \
                            ulgm_alphas['a'] * masked_mean(l3, umask) + \
                            ulgm_alphas['r'] * masked_mean(l4, umask)
            else:
                ulgm_loss = ulgm_alphas['t'] * loss_function(lp_1, labels_, umask) + \
                            ulgm_alphas['v'] * loss_function(lp_2, labels_, umask) + \
                            ulgm_alphas['a'] * loss_function(lp_3, labels_, umask) + \
                            ulgm_alphas['r'] * loss_function(lp_4, labels_, umask)
            loss = gamma_1 * loss_function(lp_all, labels_, umask) + \
                    gamma_2 * beta_e * ulgm_loss + \
                    gamma_3 * (kl_loss(kl_lp_1, kl_p_all, umask) + kl_loss(kl_lp_2, kl_p_all, umask) + kl_loss(kl_lp_3, kl_p_all, umask) + kl_loss(kl_lp_4, kl_p_all, umask))
        else:
            if use_pseudo_ulgm:
                cw = getattr(loss_function, 'weight', None)
                tgt1, _ = build_pseudo_targets_for_modality(lp_1, labels_, n_classes, happy_class, tau_conf, omega_true_major, omega_true_happy)
                tgt2, _ = build_pseudo_targets_for_modality(lp_2, labels_, n_classes, happy_class, tau_conf, omega_true_major, omega_true_happy)
                tgt3, _ = build_pseudo_targets_for_modality(lp_3, labels_, n_classes, happy_class, tau_conf, omega_true_major, omega_true_happy)

                l1 = unimodal_pseudo_loss_per_pos(lp_1, tgt1, cw, labels_)
                l2 = unimodal_pseudo_loss_per_pos(lp_2, tgt2, cw, labels_)
                l3 = unimodal_pseudo_loss_per_pos(lp_3, tgt3, cw, labels_)

                l1 = l1 * class_boost_w
                l2 = l2 * class_boost_w
                l3 = l3 * class_boost_w

                ulgm_loss = ulgm_alphas['t'] * masked_mean(l1, umask) + \
                            ulgm_alphas['v'] * masked_mean(l2, umask) + \
                            ulgm_alphas['a'] * masked_mean(l3, umask)
            else:
                ulgm_loss = ulgm_alphas['t'] * loss_function(lp_1, labels_, umask) + \
                            ulgm_alphas['v'] * loss_function(lp_2, labels_, umask) + \
                            ulgm_alphas['a'] * loss_function(lp_3, labels_, umask)
            loss = gamma_1 * loss_function(lp_all, labels_, umask) + \
                    gamma_2 * beta_e * ulgm_loss + \
                    gamma_3 * (kl_loss(kl_lp_1, kl_p_all, umask) + kl_loss(kl_lp_2, kl_p_all, umask) + kl_loss(kl_lp_3, kl_p_all, umask))

        if use_aux_losses:
            gate_term = alpha_gate * masked_mean(gate_entropy.view(-1), umask)
            loss = loss + gate_term
        if class_boost_lambda > 0:
            fused_ce = F.nll_loss(lp_all, labels_.long(), reduction='none')
            boosted_fused_ce = fused_ce * class_boost_w
            loss = loss + float(class_boost_lambda) * masked_mean(boosted_fused_ce, umask)

        lp_ = all_prob.view(-1, all_prob.size()[2])

        pred_ = torch.argmax(lp_,1)
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        if not torch.isfinite(loss):
            skipped_non_finite += 1
            continue

        losses.append(loss.item()*masks[-1].sum())
        if train:
            loss.backward()
            if args.grad_clip > 0:
                clip_grad_norm_(model.parameters(), args.grad_clip)
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()

    if preds!=[]:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan')

    if len(losses) == 0:
        return float('nan'), float('nan'), labels, preds, masks, float('nan')

    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels,preds, sample_weight=masks)*100, 2)
    avg_fscore = round(f1_score(labels,preds, sample_weight=masks, average='weighted')*100, 2)  
    if skipped_non_finite > 0:
        mode = 'train' if train else 'eval'
        print('[Warn][{}][epoch {}] skipped {} non-finite batches'.format(mode, epoch + 1, skipped_non_finite))
    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.00005, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.3, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=2, metavar='BS', help='batch size')
    parser.add_argument('--hidden_dim', type=int, default=128, metavar='hidden_dim', help='output hidden size')
    parser.add_argument('--n_head', type=int, default=4, metavar='n_head', help='number of heads')
    parser.add_argument('--epochs', type=int, default=60, metavar='E', help='number of epochs')
    parser.add_argument('--temp', type=int, default=1, metavar='temp', help='temp')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')
    parser.add_argument('--class-weight', action='store_true', default=True, help='use class weights')
    parser.add_argument('--Dataset', default='IEMOCAP', help='dataset to train and test')
    parser.add_argument('--iemocap-pkl-path', type=str, default='data/iemocap_multimodal_features.pkl',
                        help='path of IEMOCAP multimodal feature pkl')
    parser.add_argument('--iemocap-session-prefixes', type=str, default='Ses01',
                        help='comma-separated prefixes for IEMOCAP sessions, e.g. Ses01 or Ses01,Ses02')
    parser.add_argument('--use-rppg', action='store_true', default=False,
                        help='enable rPPG branch as the 4th modality')
    parser.add_argument('--iemocap-rppg-npz-path', type=str, default='data/iemocap_rppg_features_ses01_v3.npz',
                        help='path of extracted rPPG npz (expects key videoRppg342)')
    parser.add_argument('--valid-ratio', type=float, default=0.1,
                        help='validation split ratio for filtered IEMOCAP pool')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                        help='test split ratio for filtered IEMOCAP pool')
    parser.add_argument('--split-seed', type=int, default=42,
                        help='random seed for filtered IEMOCAP split')
    parser.add_argument('--split-max-tries', type=int, default=300,
                        help='max resampling tries to improve class coverage in each split')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='gradient clipping max norm; <=0 disables clipping')
    parser.add_argument('--ulgm-alpha-t', type=float, default=1.0,
                        help='ULGM modality weight alpha_t (text)')
    parser.add_argument('--ulgm-alpha-a', type=float, default=1.0,
                        help='ULGM modality weight alpha_a (audio)')
    parser.add_argument('--ulgm-alpha-v', type=float, default=1.0,
                        help='ULGM modality weight alpha_v (visual)')
    parser.add_argument('--ulgm-alpha-r', type=float, default=1.0,
                        help='ULGM modality weight alpha_r (rPPG); ignored when --use-rppg is off')
    parser.add_argument('--ulgm-lambda-max', type=float, default=1.0,
                        help='ULGM target weight lambda_ulgm_max')
    parser.add_argument('--ulgm-lambda-min', type=float, default=1.0,
                        help='ULGM initial weight lambda_ulgm_min')
    parser.add_argument('--ulgm-e-delay', type=int, default=0,
                        help='ULGM warm-up delay epochs E_delay')
    parser.add_argument('--ulgm-e-ramp', type=int, default=0,
                        help='ULGM linear ramp epochs E_ramp')
    parser.add_argument('--ulgm-normalize-alpha', action='store_true', default=False,
                        help='normalize active ULGM alpha weights to sum to 1')

    parser.add_argument('--gamma-1', type=float, default=1.0, help='weight for fused NLL task loss')
    parser.add_argument('--gamma-2', type=float, default=1.0, help='weight for ULGM term (multiplied by beta schedule)')
    parser.add_argument('--gamma-3', type=float, default=1.0, help='weight for KL self-distillation term (set 0 to disable)')
    parser.add_argument('--gamma-2-warmup-epochs', type=int, default=0,
                        help='linear ramp epochs for gamma_2 from gamma-2-start to gamma-2; 0 disables')
    parser.add_argument('--gamma-2-start', type=float, default=0.0,
                        help='initial gamma_2 at epoch 0 when gamma-2-warmup-epochs > 0')

    parser.add_argument('--class-weight-mode', type=str, default='paper',
                        choices=['none', 'legacy', 'paper'],
                        help='IEMOCAP class weight strategy for NLL losses')

    parser.add_argument('--cw-clip-min', type=float, default=0.4, help='paper class weight clip min lambda_min')
    parser.add_argument('--cw-clip-max', type=float, default=2.5, help='paper class weight clip max lambda_max')

    parser.add_argument('--no-pseudo-ulgm', action='store_true', default=False,
                        help='disable pseudo-label ULGM; use standard label NLL on unimodal heads')
    parser.add_argument('--pseudo-tau-conf', type=float, default=0.8, help='confidence threshold for happy pseudo mixing')
    parser.add_argument('--pseudo-omega-major', type=float, default=0.3, help='omega_true for non-happy rows in mixing')
    parser.add_argument('--pseudo-omega-happy', type=float, default=0.7, help='omega_true for happy rows when mixing triggers')
    parser.add_argument('--iemocap-happy-class', type=int, default=0,
                        help='IEMOCAP label id treated as Happy for pseudo/boosting (dataset-specific)')
    parser.add_argument('--iemocap-six-class', action='store_true', default=False,
                        help='disable 4-class focus and keep original IEMOCAP 6-class setup')
    parser.add_argument('--iemocap-no-valid-split', action='store_true', default=False,
                        help='set valid split to 0 and train with all non-test samples')

    parser.add_argument('--happy-boost-max', type=float, default=1.5, help='gamma_max for happy early boosting')
    parser.add_argument('--happy-boost-epochs', type=int, default=10, help='E_boost for happy early boosting')
    parser.add_argument('--iemocap-sad-class', type=int, default=1,
                        help='IEMOCAP label id treated as Sad for class-focused boosting')
    parser.add_argument('--sad-boost-max', type=float, default=1.2, help='gamma_max for sad early boosting')
    parser.add_argument('--sad-boost-epochs', type=int, default=12, help='E_boost for sad early boosting')
    parser.add_argument('--iemocap-neutral-class', type=int, default=2,
                        help='IEMOCAP label id treated as Neutral for class-focused boosting')
    parser.add_argument('--neutral-boost-max', type=float, default=1.2, help='gamma_max for neutral early boosting')
    parser.add_argument('--neutral-boost-epochs', type=int, default=12, help='E_boost for neutral early boosting')
    parser.add_argument('--class-boost-lambda', type=float, default=0.2,
                        help='weight for fused-head class-focused boosting term')

    parser.add_argument('--alpha-gate', type=float, default=0.01, help='weight for multimodal gate entropy regularizer')
    parser.add_argument('--no-aux-losses', action='store_true', default=False,
                        help='disable multimodal gate entropy auxiliary loss')

    args = parser.parse_args()
    validate_ulgm_args(args)
    if (args.Dataset == 'IEMOCAP') and (not args.class_weight):
        args.class_weight_mode = 'none'
    today = datetime.datetime.now()
    print(args)
    if args.Dataset == 'IEMOCAP':
        print('IEMOCAP feature pkl: {}'.format(args.iemocap_pkl_path))
        print('IEMOCAP sessions: {}'.format(args.iemocap_session_prefixes))
        print('Use rPPG branch: {}'.format(args.use_rppg))
        print('IEMOCAP 4-class focus: {}'.format(not args.iemocap_six_class))
        if args.iemocap_no_valid_split:
            print('IEMOCAP valid split disabled: train uses all non-test samples')
        if not args.iemocap_six_class:
            print('IEMOCAP class map: 0->0, 1->1, 2->2, 3->3 (keep original ids)')
        if args.use_rppg:
            print('IEMOCAP rPPG npz: {}'.format(args.iemocap_rppg_npz_path))
    report_labels = None
    report_target_names = None
    if args.Dataset == 'IEMOCAP' and (not args.iemocap_six_class):
        report_labels = [0, 1, 2, 3]
        report_target_names = ['happy', 'sad', 'neutral', 'angry']

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()

    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size
    feat2dim = {'IS10':1582, 'denseface':342, 'MELD_audio':300}
    D_audio = feat2dim['IS10'] if args.Dataset=='IEMOCAP' else feat2dim['MELD_audio']
    D_visual = feat2dim['denseface']
    D_rppg = 342
    D_text = 1024

    D_m = D_audio + D_visual + D_text

    n_speakers = 9 if args.Dataset=='MELD' else 2
    iemocap_four_class_map = {0: 0, 1: 1, 2: 2, 3: 3}  # keep original 0/1/2/3 ids
    if args.Dataset == 'IEMOCAP' and (not args.iemocap_six_class):
        n_classes = 4
        iemocap_target_label_map = iemocap_four_class_map
        effective_happy_class = iemocap_four_class_map.get(args.iemocap_happy_class, 0)
        effective_sad_class = iemocap_four_class_map.get(args.iemocap_sad_class, 1)
        effective_neutral_class = iemocap_four_class_map.get(args.iemocap_neutral_class, 2)
    else:
        n_classes = 7 if args.Dataset == 'MELD' else 6 if args.Dataset == 'IEMOCAP' else 1
        iemocap_target_label_map = None
        effective_happy_class = args.iemocap_happy_class
        effective_sad_class = args.iemocap_sad_class
        effective_neutral_class = args.iemocap_neutral_class

    print('temp {}'.format(args.temp))
    ulgm_alphas = build_ulgm_alphas(args)
    print('ULGM alphas: t={:.4f}, a={:.4f}, v={:.4f}, r={:.4f} (normalized={})'.format(
        ulgm_alphas['t'], ulgm_alphas['a'], ulgm_alphas['v'], ulgm_alphas['r'], args.ulgm_normalize_alpha))
    print('ULGM beta schedule: init={}, target={}, delay={}, warmup={}'.format(
        args.ulgm_lambda_min, args.ulgm_lambda_max, args.ulgm_e_delay, args.ulgm_e_ramp))

    model = Transformer_Based_Model(args.Dataset, args.temp, D_text, D_visual, D_audio, args.n_head,
                                        n_classes=n_classes,
                                        hidden_dim=args.hidden_dim,
                                        n_speakers=n_speakers,
                                        dropout=args.dropout,
                                        use_rppg=args.use_rppg,
                                        D_rppg=D_rppg)

    total_params = sum(p.numel() for p in model.parameters())
    print('total parameters: {}'.format(total_params))
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('training parameters: {}'.format(total_trainable_params))

    if cuda:
        model.cuda()
        
    kl_loss = MaskedKLDivLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    if args.Dataset == 'MELD':
        loss_function = MaskedNLLLoss()
        train_loader, valid_loader, test_loader = get_MELD_loaders(valid=0.0,
                                                                    batch_size=batch_size,
                                                                    num_workers=0)
    elif args.Dataset == 'IEMOCAP':
        iemocap_session_prefixes = [x.strip() for x in args.iemocap_session_prefixes.split(',') if x.strip()]
        valid_ratio_effective = 0.0 if args.iemocap_no_valid_split else args.valid_ratio
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(iemocap_pkl_path=args.iemocap_pkl_path,
                                                                      session_prefixes=iemocap_session_prefixes,
                                                                      use_rppg=args.use_rppg,
                                                                      rppg_npz_path=args.iemocap_rppg_npz_path,
                                                                      valid=valid_ratio_effective,
                                                                      test=args.test_ratio,
                                                                      split_seed=args.split_seed,
                                                                      split_max_tries=args.split_max_tries,
                                                                      n_classes=n_classes,
                                                                      target_label_map=iemocap_target_label_map,
                                                                      batch_size=batch_size,
                                                                      num_workers=0)
        if args.class_weight_mode == 'none':
            loss_function = MaskedNLLLoss()
            print('IEMOCAP class weights: disabled')
        elif args.class_weight_mode == 'legacy':
            loss_weights = torch.FloatTensor([1/0.086747,
                                            1/0.144406,
                                            1/0.227883,
                                            1/0.160585,
                                            1/0.127711,
                                            1/0.252668])
            lw = loss_weights.cuda() if cuda else loss_weights
            loss_function = MaskedNLLLoss(lw)
            print('IEMOCAP class weights (legacy fixed):', lw.tolist())
        else:
            counts = count_iemocap_utterance_class_counts(
                train_loader.dataset, train_loader.sampler.indices, n_classes
            )
            w_hat = build_auto_class_weights_paper(counts, clip_min=args.cw_clip_min, clip_max=args.cw_clip_max)
            lw = w_hat.cuda() if cuda else w_hat
            loss_function = MaskedNLLLoss(lw)
            print('IEMOCAP class counts (train split utterances):', counts.tolist())
            print('IEMOCAP class weights (paper):', lw.tolist())
    else:
        print("There is no such dataset")

    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []

    for e in range(n_epochs):
        start_time = time.time()
        beta_e = compute_beta_schedule(
            e, args.ulgm_lambda_min, args.ulgm_lambda_max, args.ulgm_e_delay, args.ulgm_e_ramp
        )
        gamma2_e = compute_gamma2_schedule(
            e, args.gamma_2, args.gamma_2_start, args.gamma_2_warmup_epochs
        )
        gamma_boost_e = compute_happy_boost(e, args.happy_boost_max, args.happy_boost_epochs)
        sad_boost_e = compute_happy_boost(e, args.sad_boost_max, args.sad_boost_epochs)
        neutral_boost_e = compute_happy_boost(e, args.neutral_boost_max, args.neutral_boost_epochs)
        use_pseudo_ulgm = (args.Dataset == 'IEMOCAP') and (not args.no_pseudo_ulgm)
        use_aux_losses = (not args.no_aux_losses)
        train_loss, train_acc, _, _, _, train_fscore = train_or_eval_model(
            model, loss_function, kl_loss, train_loader, e, optimizer, True,
            gamma_1=args.gamma_1, gamma_2=gamma2_e, gamma_3=args.gamma_3,
            use_rppg=args.use_rppg, beta_e=beta_e, ulgm_alphas=ulgm_alphas,
            n_classes=n_classes, happy_class=effective_happy_class,
            tau_conf=args.pseudo_tau_conf,
            omega_true_major=args.pseudo_omega_major,
            omega_true_happy=args.pseudo_omega_happy,
            gamma_boost=gamma_boost_e,
            happy_class_idx=effective_happy_class,
            sad_class_idx=effective_sad_class,
            sad_boost=sad_boost_e,
            neutral_class_idx=effective_neutral_class,
            neutral_boost=neutral_boost_e,
            class_boost_lambda=args.class_boost_lambda,
            alpha_gate=args.alpha_gate,
            use_pseudo_ulgm=use_pseudo_ulgm,
            use_aux_losses=use_aux_losses,
        )
        valid_indices = getattr(getattr(valid_loader, 'sampler', None), 'indices', None)
        if valid_indices is not None and len(valid_indices) == 0:
            valid_loss, valid_acc, valid_fscore = train_loss, train_acc, train_fscore
        else:
            valid_loss, valid_acc, _, _, _, valid_fscore = train_or_eval_model(
                model, loss_function, kl_loss, valid_loader, e,
                gamma_1=args.gamma_1, gamma_2=gamma2_e, gamma_3=args.gamma_3,
                use_rppg=args.use_rppg, beta_e=beta_e, ulgm_alphas=ulgm_alphas,
                n_classes=n_classes, happy_class=effective_happy_class,
                tau_conf=args.pseudo_tau_conf,
                omega_true_major=args.pseudo_omega_major,
                omega_true_happy=args.pseudo_omega_happy,
                gamma_boost=gamma_boost_e,
                happy_class_idx=effective_happy_class,
                sad_class_idx=effective_sad_class,
                sad_boost=sad_boost_e,
                neutral_class_idx=effective_neutral_class,
                neutral_boost=neutral_boost_e,
                class_boost_lambda=args.class_boost_lambda,
                alpha_gate=args.alpha_gate,
                use_pseudo_ulgm=use_pseudo_ulgm,
                use_aux_losses=use_aux_losses,
            )
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore = train_or_eval_model(
            model, loss_function, kl_loss, test_loader, e,
            gamma_1=args.gamma_1, gamma_2=gamma2_e, gamma_3=args.gamma_3,
            use_rppg=args.use_rppg, beta_e=beta_e, ulgm_alphas=ulgm_alphas,
            n_classes=n_classes, happy_class=effective_happy_class,
            tau_conf=args.pseudo_tau_conf,
            omega_true_major=args.pseudo_omega_major,
            omega_true_happy=args.pseudo_omega_happy,
            gamma_boost=gamma_boost_e,
            happy_class_idx=effective_happy_class,
            sad_class_idx=effective_sad_class,
            sad_boost=sad_boost_e,
            neutral_class_idx=effective_neutral_class,
            neutral_boost=neutral_boost_e,
            class_boost_lambda=args.class_boost_lambda,
            alpha_gate=args.alpha_gate,
            use_pseudo_ulgm=use_pseudo_ulgm,
            use_aux_losses=use_aux_losses,
        )
        all_fscore.append(test_fscore)

        if (test_label is not None) and (len(test_label) > 0) and (best_fscore == None or best_fscore < test_fscore):
            best_fscore = test_fscore
            best_label, best_pred, best_mask = test_label, test_pred, test_mask

        if args.tensorboard:
            writer.add_scalar('test: accuracy', test_acc, e)
            writer.add_scalar('test: fscore', test_fscore, e)
            writer.add_scalar('train: accuracy', train_acc, e)
            writer.add_scalar('train: fscore', train_fscore, e)

        print('epoch: {}, beta_ulgm: {:.6f}, gamma2: {:.4f}, happy_boost: {:.4f}, sad_boost: {:.4f}, neutral_boost: {:.4f}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.\
                format(e+1, beta_e, gamma2_e, gamma_boost_e, sad_boost_e, neutral_boost_e, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc, test_fscore, round(time.time()-start_time, 2)))
        if (e+1)%10 == 0 and best_label is not None and len(best_label) > 0:
            print(classification_report(
                best_label, best_pred, labels=report_labels, target_names=report_target_names,
                sample_weight=best_mask, digits=4
            ))
            print(confusion_matrix(best_label,best_pred,sample_weight=best_mask))


    if args.tensorboard:
        writer.close()

    print('Test performance..')
    print('F-Score: {}'.format(max(all_fscore)))
    print('F-Score-index: {}'.format(all_fscore.index(max(all_fscore)) + 1))
    
    if not os.path.exists("record_{}_{}_{}.pk".format(today.year, today.month, today.day)):
        with open("record_{}_{}_{}.pk".format(today.year, today.month, today.day),'wb') as f:
            pk.dump({}, f)
    with open("record_{}_{}_{}.pk".format(today.year, today.month, today.day), 'rb') as f:
        record = pk.load(f)
    key_ = 'name_'
    if record.get(key_, False):
        record[key_].append(max(all_fscore))
    else:
        record[key_] = [max(all_fscore)]
    if best_label is not None and len(best_label) > 0:
        best_report = classification_report(
            best_label, best_pred, labels=report_labels, target_names=report_target_names,
            sample_weight=best_mask, digits=4
        )
    else:
        best_report = 'No non-empty test split after filtering; classification report skipped.'
    if record.get(key_+'record', False):
        record[key_+'record'].append(best_report)
    else:
        record[key_+'record'] = [best_report]
    with open("record_{}_{}_{}.pk".format(today.year, today.month, today.day),'wb') as f:
        pk.dump(record, f)

    if best_label is not None and len(best_label) > 0:
        print(classification_report(
            best_label, best_pred, labels=report_labels, target_names=report_target_names,
            sample_weight=best_mask, digits=4
        ))
        print(confusion_matrix(best_label,best_pred,sample_weight=best_mask))
    else:
        print('No non-empty test split after filtering; skipped classification report and confusion matrix.')


