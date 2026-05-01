import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np, argparse, time
import torch
import torch.optim as optim
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
                        n_classes=6, num_workers=0, pin_memory=False):
    # Build on the same filtered pool so all modalities (including rPPG) share identical split.
    fullset = IEMOCAPDataset(path=iemocap_pkl_path, split='all', session_prefixes=session_prefixes,
                              use_rppg=use_rppg, rppg_npz_path=rppg_npz_path)
    size = len(fullset)
    all_idx = list(range(size))
    valid_size = int(valid * size)
    test_size = int(test * size)
    if size > 0 and test_size == 0:
        test_size = 1
    if size > 1 and valid_size == 0:
        valid_size = 1
    if valid_size + test_size >= size and size > 2:
        test_size = max(1, min(test_size, size - 2))
        valid_size = max(1, min(valid_size, size - test_size - 1))

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


def train_or_eval_model(model, loss_function, kl_loss, dataloader, epoch, optimizer=None, train=False, gamma_1=1.0, gamma_2=1.0, gamma_3=1.0, use_rppg=False, ulgm_lambda=1.0, ulgm_alphas=None):
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
            log_prob1, log_prob2, log_prob3, log_prob4, all_log_prob, all_prob, \
            kl_log_prob1, kl_log_prob2, kl_log_prob3, kl_log_prob4, kl_all_prob = model(textf, visuf, acouf, umask, qmask, lengths, rppgf=rppgf)
        else:
            log_prob1, log_prob2, log_prob3, all_log_prob, all_prob, \
            kl_log_prob1, kl_log_prob2, kl_log_prob3, kl_all_prob = model(textf, visuf, acouf, umask, qmask, lengths)
        
        lp_1 = log_prob1.view(-1, log_prob1.size()[2])
        lp_2 = log_prob2.view(-1, log_prob2.size()[2])
        lp_3 = log_prob3.view(-1, log_prob3.size()[2])
        lp_all = all_log_prob.view(-1, all_log_prob.size()[2])
        labels_ = label.view(-1)

        kl_lp_1 = kl_log_prob1.view(-1, kl_log_prob1.size()[2])
        kl_lp_2 = kl_log_prob2.view(-1, kl_log_prob2.size()[2])
        kl_lp_3 = kl_log_prob3.view(-1, kl_log_prob3.size()[2])
        kl_p_all = kl_all_prob.view(-1, kl_all_prob.size()[2])
        if use_rppg:
            lp_4 = log_prob4.view(-1, log_prob4.size()[2])
            kl_lp_4 = kl_log_prob4.view(-1, kl_log_prob4.size()[2])
            ulgm_loss = ulgm_alphas['t'] * loss_function(lp_1, labels_, umask) + \
                        ulgm_alphas['v'] * loss_function(lp_2, labels_, umask) + \
                        ulgm_alphas['a'] * loss_function(lp_3, labels_, umask) + \
                        ulgm_alphas['r'] * loss_function(lp_4, labels_, umask)
            loss = gamma_1 * loss_function(lp_all, labels_, umask) + \
                    gamma_2 * ulgm_lambda * ulgm_loss + \
                    gamma_3 * (kl_loss(kl_lp_1, kl_p_all, umask) + kl_loss(kl_lp_2, kl_p_all, umask) + kl_loss(kl_lp_3, kl_p_all, umask) + kl_loss(kl_lp_4, kl_p_all, umask))
        else:
            ulgm_loss = ulgm_alphas['t'] * loss_function(lp_1, labels_, umask) + \
                        ulgm_alphas['v'] * loss_function(lp_2, labels_, umask) + \
                        ulgm_alphas['a'] * loss_function(lp_3, labels_, umask)
            loss = gamma_1 * loss_function(lp_all, labels_, umask) + \
                    gamma_2 * ulgm_lambda * ulgm_loss + \
                    gamma_3 * (kl_loss(kl_lp_1, kl_p_all, umask) + kl_loss(kl_lp_2, kl_p_all, umask) + kl_loss(kl_lp_3, kl_p_all, umask))

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

    args = parser.parse_args()
    validate_ulgm_args(args)
    today = datetime.datetime.now()
    print(args)
    if args.Dataset == 'IEMOCAP':
        print('IEMOCAP feature pkl: {}'.format(args.iemocap_pkl_path))
        print('IEMOCAP sessions: {}'.format(args.iemocap_session_prefixes))
        print('Use rPPG branch: {}'.format(args.use_rppg))
        if args.use_rppg:
            print('IEMOCAP rPPG npz: {}'.format(args.iemocap_rppg_npz_path))

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
    n_classes = 7 if args.Dataset=='MELD' else 6 if args.Dataset=='IEMOCAP' else 1

    print('temp {}'.format(args.temp))
    ulgm_alphas = build_ulgm_alphas(args)
    print('ULGM alphas: t={:.4f}, a={:.4f}, v={:.4f}, r={:.4f} (normalized={})'.format(
        ulgm_alphas['t'], ulgm_alphas['a'], ulgm_alphas['v'], ulgm_alphas['r'], args.ulgm_normalize_alpha))
    print('ULGM lambda schedule: min={}, max={}, delay={}, ramp={}'.format(
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
        loss_weights = torch.FloatTensor([1/0.086747,
                                        1/0.144406,
                                        1/0.227883,
                                        1/0.160585,
                                        1/0.127711,
                                        1/0.252668])
        loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(iemocap_pkl_path=args.iemocap_pkl_path,
                                                                      session_prefixes=iemocap_session_prefixes,
                                                                      use_rppg=args.use_rppg,
                                                                      rppg_npz_path=args.iemocap_rppg_npz_path,
                                                                      valid=args.valid_ratio,
                                                                      test=args.test_ratio,
                                                                      split_seed=args.split_seed,
                                                                      split_max_tries=args.split_max_tries,
                                                                      n_classes=n_classes,
                                                                      batch_size=batch_size,
                                                                      num_workers=0)
        dynamic_loss_weights = build_dynamic_class_weights(
            train_loader.dataset, train_loader.sampler.indices, n_classes
        )
        print('Dynamic class weights:', dynamic_loss_weights.tolist())
        loss_function = MaskedNLLLoss(dynamic_loss_weights.cuda() if cuda else dynamic_loss_weights)
    else:
        print("There is no such dataset")

    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []

    for e in range(n_epochs):
        start_time = time.time()
        ulgm_lambda_e = compute_ulgm_lambda(
            e, args.ulgm_lambda_min, args.ulgm_lambda_max, args.ulgm_e_delay, args.ulgm_e_ramp
        )
        train_loss, train_acc, _, _, _, train_fscore = train_or_eval_model(
            model, loss_function, kl_loss, train_loader, e, optimizer, True,
            use_rppg=args.use_rppg, ulgm_lambda=ulgm_lambda_e, ulgm_alphas=ulgm_alphas
        )
        valid_loss, valid_acc, _, _, _, valid_fscore = train_or_eval_model(
            model, loss_function, kl_loss, valid_loader, e,
            use_rppg=args.use_rppg, ulgm_lambda=ulgm_lambda_e, ulgm_alphas=ulgm_alphas
        )
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore = train_or_eval_model(
            model, loss_function, kl_loss, test_loader, e,
            use_rppg=args.use_rppg, ulgm_lambda=ulgm_lambda_e, ulgm_alphas=ulgm_alphas
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

        print('epoch: {}, ulgm_lambda: {:.6f}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.\
                format(e+1, ulgm_lambda_e, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc, test_fscore, round(time.time()-start_time, 2)))
        if (e+1)%10 == 0 and best_label is not None and len(best_label) > 0:
            print(classification_report(best_label, best_pred, sample_weight=best_mask,digits=4))
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
        best_report = classification_report(best_label, best_pred, sample_weight=best_mask,digits=4)
    else:
        best_report = 'No non-empty test split after filtering; classification report skipped.'
    if record.get(key_+'record', False):
        record[key_+'record'].append(best_report)
    else:
        record[key_+'record'] = [best_report]
    with open("record_{}_{}_{}.pk".format(today.year, today.month, today.day),'wb') as f:
        pk.dump(record, f)

    if best_label is not None and len(best_label) > 0:
        print(classification_report(best_label, best_pred, sample_weight=best_mask,digits=4))
        print(confusion_matrix(best_label,best_pred,sample_weight=best_mask))
    else:
        print('No non-empty test split after filtering; skipped classification report and confusion matrix.')


