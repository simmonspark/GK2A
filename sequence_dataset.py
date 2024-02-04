import os
import numpy as np
from torch.utils.data import Dataset, Sampler, DataLoader
from utils import get_date_list
import torch
import cv2 as cv
import netCDF4 as nc
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_sampler_idx(total_list, missing_list, sequence_len=3):
    remain = len(total_list) % sequence_len

    '''if remain != 0:
        total_list = total_list[:-(remain)]
        print(f'remain num : {remain} is deleted!')'''

    miss_idx = [total_list.index(i) for i in missing_list]
    store = [i for i, _ in enumerate(total_list)]
    for_del_idx_store = []

    while miss_idx:
        idx = miss_idx.pop(0)
        R = range(idx - sequence_len, idx + 1, 1)
        for i in R:
            if (i >= 0):
                for_del_idx_store.append(i)
    A = set(for_del_idx_store)
    B = set(store)
    C = B - A
    C = list(C)

    return C[:-(sequence_len)]


class Sampler_Dataset(Dataset):
    def __init__(self, root_data_path, date_from, date_to, interval, img_size, sequence_len, ir_only=True):
        self.img_list = []
        self.missing_date = []
        self.seq_len = sequence_len
        self.ir_only = ir_only

        date_list = get_date_list(date_from, date_to, interval_minutes=interval)

        for date in tqdm(date_list, desc='Loading Dataset..'):
            ir_path = root_data_path + '/GK2A/IR/' + date + '.nc'
            rr_path = root_data_path + '/GK2A/RR/' + date + '.nc'
            if ir_only:
                path = root_data_path + '/GK2A/IR/' + date + '.nc'
            else:
                path = root_data_path + '/GK2A/RR/' + date + '.nc'

            if os.path.isfile(path):
                data = nc.Dataset(path)

                if ir_only:
                    img = data.variables['image_pixel_values'][:]
                else:
                    img = data.variables['RR'][:]

                self.img_list.append(cv.resize(img, (img_size, img_size), interpolation=cv.INTER_AREA))
                data.close()

            else:
                self.img_list.append(np.zeros((img_size, img_size), dtype=np.float32))
                self.missing_date.append(date)

        self.img_list = np.array(self.img_list, dtype=np.float32)

        self.img_list = (self.img_list - self.img_list.min()) / (self.img_list.max() - self.img_list.min())

        self.sampler_idx = get_sampler_idx(total_list=date_list, missing_list=self.missing_date,
                                           sequence_len=self.seq_len)

        # print(f'total data len with missing data is {len(date_list)}, after replace 짱구 data is {len(self.img_list)}')

        print(f'Dataset Size: {len(self.img_list)}')
        print(f'Number of Missing Date: {len(self.missing_date)}')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        imgs = self.img_list[idx:idx + self.seq_len]

        imgs = torch.Tensor(imgs)

        return imgs


class SEQ_Sampler(Sampler):
    def __init__(self, index):
        super(SEQ_Sampler, self).__init__()
        self.index = index

    def __iter__(self):
        for i in self.index:
            yield i

    def __len__(self):
        return len(self.index)


if __name__ == '__main__':
    root_data_path = '/media/sien/DATA/DATA/dataset'
    date_from = '20220602'
    date_to = '20220603'
    interval = 10
    img_size = 224

    dataset = Sampler_Dataset(root_data_path, date_from, date_to, interval, img_size, sequence_len=3)
    sampler = SEQ_Sampler(dataset.sampler_idx)
    # loader = DataLoader(dataset=dataset, sampler=sampler, shuffle=False, batch_size=8,drop_last=True)
    cnt = 0
    for img in next(iter(dataset)):
        print(img.shape)
        cnt += 1
        print(cnt)

    dataset.__len__()
    len(dataset.missing_date)

    print('END!')
