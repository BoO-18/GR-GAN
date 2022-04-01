import os
import torch
from torch.utils import data
import torchvision.transforms as transforms
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer


class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data_dir, split='train'):
        'Initialization'
        self.data_dir = data_dir
        self.captions, self.cap_lens, self.n_words = self.load_text_data(data_dir, split)
        self.embeddings_num = 5

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.captions)

    def __getitem__(self, index):
        # sent_ix = np.random.randint(0, self.embeddings_num)
        # new_sent_ix = index * self.embeddings_num + sent_ix
        caps, cap_len = self.get_caption(index)
        return caps

    def get_caption(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        x_len = self.cap_lens[sent_ix]

        return sent_caption, x_len

    def load_text_data(self, data_dir, split):
        filepath = os.path.join(data_dir, 'captions_clip.pickle')
        with open(filepath, 'rb') as f:
            x = pickle.load(f)
            train_captions, test_captions = x[0], x[1]
            train_lens, test_lens = x[2], x[3]
            # ixtoword, wordtoix = x[2], x[3]
            del x
            n_words = 49408
            print('Load from: ', filepath)
        if split == 'train':
            # a list of list: each list contains
            # the indices of words in a sentence
            cap_lens = train_lens
        else:  # split=='test'
            captions = test_captions
            cap_lens = test_lens
        return captions, cap_lens, n_words


if __name__ == '__main__':
    DATA_DIR = '/raid_sdc/dataset/coco'
    split = 'test'
    batch_size = 16
    dataset = Dataset(DATA_DIR, split)
    print(dataset.__len__())
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    for i, batch in enumerate(dataloader):
        print(batch)
        break
