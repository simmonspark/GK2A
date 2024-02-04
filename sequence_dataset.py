import os
import numpy as np
from torch.utils.data import Dataset, Sampler, DataLoader
from utils import get_date_list
import torch
import cv2 as cv
import netCDF4 as nc
from tqdm import tqdm


def get_sampler_idx(total_list, missing_list, sequence_len=3):
    miss_idx = [total_list.index(i) for i in missing_list]
    store = [i for i, _ in enumerate(total_list)]
    for_del_idx_store = []

    for idx in miss_idx:
        R = range(idx - sequence_len, idx + 1, 1)
        for i in R:
            if i >= 0:
                for_del_idx_store.append(i)

    A = set(for_del_idx_store)
    B = set(store)
    C = B - A
    C = list(C)

    return C[:-sequence_len]


class Sequence_Dataset(Dataset):
    def __init__(self, root_data_path, date_from, date_to, interval, img_size, sequence_len, ir_only=True):
        self.img_list = []
        self.missing_date = []
        self.seq_len = sequence_len
        self.ir_only = ir_only

        date_list = get_date_list(date_from, date_to, interval_minutes=interval)

        for date in tqdm(date_list, desc='Loading Dataset..'):
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
        self.sampler = SeqSampler(self.sampler_idx)

        print(f'Dataset Size: {len(self.img_list)}')
        print(f'Number of Missing Date: {len(self.missing_date)}')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        inputs = self.img_list[idx:idx + self.seq_len]
        target = self.img_list[idx + self.seq_len]
        inputs = torch.Tensor(inputs).to('cuda')
        target = torch.Tensor(target).to('cuda')

        return inputs, target


class SeqSampler(Sampler):
    def __init__(self, data_source):
        super().__init__(data_source)
        self.data_source = data_source

    def __iter__(self):
        return iter(self.data_source)

    def __len__(self):
        return len(self.data_source)


if __name__ == '__main__':
    root_data_path = "/home/jh/data/"
    date_from = '20220601'
    date_to = '20220603'
    interval = 10
    img_size = 224

    dataset = Sequence_Dataset(root_data_path, date_from, date_to, interval, img_size, sequence_len=3)
    loader = DataLoader(dataset=dataset, sampler=dataset.sampler, shuffle=False, batch_size=8)

    cnt = 0
    for img in next(iter(dataset)):
        print(img.shape)
        cnt += 1
        print(cnt)

    dataset.__len__()
    len(dataset.missing_date)

    print('END!')
