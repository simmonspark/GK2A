import os
import numpy as np
from torch.utils.data import Dataset
from utils import create_one_img_label, get_date_list
import torch
import cv2 as cv
import netCDF4 as nc
from torch.utils.data import DataLoader
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def get_sampler_idx(total_list, missing_list, sequence_len=3):
    remain = len(total_list) % sequence_len

    if remain != 0:
        total_list = total_list[:-(remain)]
        print(f'remain num : {remain} is deleted!')

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
    return list(C)


def process_images(ir_path, rr_path):
    ir_img, rr_img = create_one_img_label(ir_path, rr_path)
    ir_resized = cv.resize(ir_img, (224, 224))
    rr_resized = cv.resize(rr_img, (224, 224))
    return ir_resized, rr_resized


class Dataset(Dataset):
    def __init__(self, ir_list, rr_list):
        self.ir_list = ir_list
        self.rr_list = rr_list
        self.ir_store = []
        self.rr_store = []
        with ProcessPoolExecutor(max_workers=15) as executor:
            future_to_images = {executor.submit(process_images, ir, rr): (ir, rr) for ir, rr in
                                zip(self.ir_list, self.rr_list)}
            for future in as_completed(future_to_images):
                try:
                    ir_resized, rr_resized = future.result()
                    self.ir_store.append(ir_resized)
                    self.rr_store.append(rr_resized)
                except Exception as exception:
                    print(f'멀티프로세싱 또 오류남 ㅋㅋ-> {exception}')

    def __len__(self):
        return len(self.ir_list)

    def __getitem__(self, idx):

        ir = self.ir_store[idx]
        rr = self.rr_store[idx]

        ir = torch.Tensor(ir).to('cuda')
        rr = torch.Tensor(rr).to('cuda')

        return ir.unsqueeze(0), rr.unsqueeze(0)


class IR2RR_Dataset(Dataset):
    def __init__(self, root_data_path, date_from, date_to, interval, img_size):
        self.ir_img_list = []
        self.rr_img_list = []
        self.missing_date = []

        date_list = get_date_list(date_from, date_to, interval_minutes=interval)

        for date in tqdm(date_list, desc='Loading Dataset..'):
            ir_path = root_data_path + '/GK2A/IR/' + date + '.nc'
            rr_path = root_data_path + '/GK2A/RR/' + date + '.nc'
            if os.path.isfile(ir_path) and os.path.isfile(rr_path):
                ir_data = nc.Dataset(ir_path)
                rr_data = nc.Dataset(rr_path)

                ir_img = ir_data.variables['image_pixel_values'][:]
                rr_img = rr_data.variables['RR'][:]

                self.ir_img_list.append(cv.resize(ir_img, (img_size, img_size), interpolation=cv.INTER_AREA))
                self.rr_img_list.append(cv.resize(rr_img, (img_size, img_size), interpolation=cv.INTER_AREA))

                ir_data.close()
                rr_data.close()

            else:
                self.missing_date.append(date)

        self.ir_img_list = np.array(self.ir_img_list, dtype=np.float32)
        self.rr_img_list = np.array(self.rr_img_list, dtype=np.float32)

        # self.ir_img_list = (self.ir_img_list - self.ir_img_list.mean()) / (self.ir_img_list.std())
        # self.rr_img_list = (self.rr_img_list - self.rr_img_list.mean()) / (self.rr_img_list.std())

        self.ir_img_list = (self.ir_img_list - self.ir_img_list.min()) / (self.ir_img_list.max() - self.ir_img_list.min())
        self.rr_img_list = (self.rr_img_list - self.rr_img_list.min()) / (self.rr_img_list.max() - self.rr_img_list.min())

        self.sampler_idx = get_sampler_idx(total_list=date_list, missing_list=self.missing_date)

        print(f'Dataset Size: {len(self.ir_img_list)}')
        print(f'Number of Missing Date: {len(self.missing_date)}')

    def __len__(self):
        return len(self.ir_img_list)

    def __getitem__(self, idx):
        ir = self.ir_img_list[idx]
        rr = self.rr_img_list[idx]

        ir = torch.Tensor(ir).to('cuda')
        rr = torch.Tensor(rr).to('cuda')

        return ir.unsqueeze(0), rr.unsqueeze(0)


if __name__ == '__main__':
    root_data_path = '/media/sien/DATA/DATA/dataset/'
    date_from = '20230601'
    date_to = '20230605'
    interval = 10
    img_size = 256

    dataset = IR2RR_Dataset(root_data_path, date_from, date_to, interval, img_size)

    data_loader = DataLoader(dataset, batch_size=4, num_workers=4)

    print('END!')
