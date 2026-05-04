import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd
import numpy as np


class IEMOCAPDataset(Dataset):
    def __init__(self, path='data/iemocap_multimodal_features.pkl', train=True, split='train', session_prefixes=None,
                 use_rppg=False, rppg_npz_path='data/iemocap_rppg_features_ses01_v3.npz',
                 target_label_map=None):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.roberta2, self.roberta3, self.roberta4, \
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open(path, 'rb'), encoding='latin1')
        if split == 'train':
            all_keys = [x for x in self.trainVid]
        elif split == 'test':
            all_keys = [x for x in self.testVid]
        elif split == 'all':
            all_keys = [x for x in self.trainVid] + [x for x in self.testVid]
        else:
            all_keys = [x for x in (self.trainVid if train else self.testVid)]
        if session_prefixes is None:
            self.keys = all_keys
        else:
            self.keys = [x for x in all_keys if any(x.startswith(prefix) for prefix in session_prefixes)]

        self.target_label_map = target_label_map
        self._kept_utter_indices = {}
        if self.target_label_map is not None:
            filtered_keys = []
            remapped_labels = {}
            remapped_text = {}
            remapped_audio = {}
            remapped_visual = {}
            remapped_speakers = {}
            remapped_sentences = {}
            for vid in self.keys:
                old_labels = self.videoLabels[vid]
                keep_idx = [i for i, y in enumerate(old_labels) if int(y) in self.target_label_map]
                if len(keep_idx) == 0:
                    continue
                filtered_keys.append(vid)
                self._kept_utter_indices[vid] = keep_idx
                remapped_labels[vid] = [int(self.target_label_map[int(old_labels[i])]) for i in keep_idx]
                remapped_text[vid] = [self.videoText[vid][i] for i in keep_idx]
                remapped_audio[vid] = [self.videoAudio[vid][i] for i in keep_idx]
                remapped_visual[vid] = [self.videoVisual[vid][i] for i in keep_idx]
                remapped_speakers[vid] = [self.videoSpeakers[vid][i] for i in keep_idx]
                remapped_sentences[vid] = [self.videoSentence[vid][i] for i in keep_idx]
            self.keys = filtered_keys
            self.videoLabels = remapped_labels
            self.videoText = remapped_text
            self.videoAudio = remapped_audio
            self.videoVisual = remapped_visual
            self.videoSpeakers = remapped_speakers
            self.videoSentence = remapped_sentences

        self.use_rppg = use_rppg
        self.videoRppg = {}
        if self.use_rppg:
            pack = np.load(rppg_npz_path, allow_pickle=True)
            raw_rppg = pack['videoRppg342'].item()
            if self.target_label_map is not None:
                for vid in self.keys:
                    keep_idx = self._kept_utter_indices[vid]
                    full_rppg = np.asarray(
                        raw_rppg.get(vid, np.zeros((max(keep_idx) + 1, 342), dtype=np.float32)),
                        dtype=np.float32
                    )
                    selected = []
                    for idx in keep_idx:
                        if idx < len(full_rppg):
                            selected.append(full_rppg[idx])
                        else:
                            selected.append(np.zeros((342,), dtype=np.float32))
                    self.videoRppg[vid] = np.asarray(selected, dtype=np.float32)
            else:
                self.videoRppg = raw_rppg

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        base = (
            torch.FloatTensor(self.videoText[vid]),
            torch.FloatTensor(self.videoVisual[vid]),
            torch.FloatTensor(self.videoAudio[vid]),
        )
        if self.use_rppg:
            rppg = torch.FloatTensor(self.videoRppg.get(vid, np.zeros((len(self.videoLabels[vid]), 342), dtype=np.float32)))
            return base[0], base[1], base[2], rppg, \
                   torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in self.videoSpeakers[vid]]),\
                   torch.FloatTensor([1]*len(self.videoLabels[vid])),\
                   torch.LongTensor(self.videoLabels[vid]),\
                   vid
        return base[0], base[1], base[2],\
               torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in\
                                  self.videoSpeakers[vid]]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        if self.use_rppg:
            # text, visual, audio, rppg, qmask, umask, label, vid
            return [pad_sequence(dat[i]) if i<5 else pad_sequence(dat[i], True) if i<7 else dat[i].tolist() for i in dat]
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]


class MELDDataset(Dataset):
    def __init__(self, path, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
        self.roberta2, self.roberta3, self.roberta4, \
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid, _ = pickle.load(open(path, 'rb'))

        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor(self.videoSpeakers[vid]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid

    def __len__(self):
        return self.len

    def return_labels(self):
        return_label = []
        for key in self.keys:
            return_label+=self.videoLabels[key]
        return return_label

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]