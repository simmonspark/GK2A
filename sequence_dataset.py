import os
import numpy as np
from torch.utils.data import Dataset,Sampler, DataLoader
from utils import get_date_list
import torch
import cv2 as cv
import netCDF4 as nc
from tqdm import tqdm
import matplotlib.pyplot as plt

debug_img = np.array(cv.resize(cv.cvtColor(plt.imread('./pred_example/debug_img.jpeg'),cv.COLOR_BGR2GRAY),dsize=(256,256)),dtype=float)

def get_sampler_idx(total_list,missing_list,sequence_len=3):
    remain = len(total_list) % sequence_len

    if remain != 0:
        total_list = total_list[:-(remain)]
        print(f'remain num : {remain} is deleted!')

    miss_idx = [total_list.index(i) for i in missing_list]
    store = [i for i , _ in enumerate(total_list)]
    for_del_idx_store = []


    while miss_idx:
        idx = miss_idx.pop(0)
        R = range(idx-sequence_len,idx+1,1)
        for i in R:
            if(i>=0):
                for_del_idx_store.append(i)
    A = set(for_del_idx_store)
    B = set(store)
    C = B - A
    return list(C)





class Sampler_Dataset(Dataset):
    def __init__(self, root_data_path, date_from, date_to, interval, img_size, sequence_len, ir_only = False):
        self.ir_img_list = []
        self.rr_img_list = []
        self.missing_date = []
        self.seq_len = sequence_len
        self.ir_only = ir_only

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
                self.ir_img_list.append(debug_img)
                self.rr_img_list.append(debug_img)

        self.ir_img_list=np.array(self.ir_img_list,dtype=float)
        self.rr_img_list=np.array(self.rr_img_list,dtype=float)

        self.sampler_idx = get_sampler_idx(total_list=date_list,missing_list=self.missing_date,sequence_len=self.seq_len)
        print(f'total_data_len with missing data is {len(date_list)}, after replace 짱구 data len is {len(self.ir_img_list)}')

        print(f'Dataset Size: {len(self.ir_img_list)}')
        print(f'Number of Missing Date: {len(self.missing_date)}')

    def __len__(self):
        return len(self.ir_img_list)

    def __getitem__(self, idx):
        ir = self.ir_img_list[idx:idx+self.seq_len]
        rr = self.rr_img_list[idx:idx+self.seq_len]

        ir = torch.Tensor(ir)
        rr = torch.Tensor(rr)

        if self.ir_only is True :
            return ir
        elif self.ir_only is False :
            return rr
class SEQ_Sampler(Sampler):
    def __init__(self,index):
        super(SEQ_Sampler,self).__init__()
        self.index = index
    def __iter__(self):
        for i in self.index:
            yield i
    def __len__(self):
        return len(self.index)



if __name__ == '__main__':
    root_data_path = '/media/sien/DATA/DATA/dataset'
    date_from = '20230701'
    date_to = '20230702'
    interval = 10
    img_size = 256

    dataset = Sampler_Dataset(root_data_path, date_from, date_to, interval, img_size, sequence_len=4)
    sampler = SEQ_Sampler(dataset.sampler_idx)
    loader = DataLoader(dataset=dataset,sampler=sampler,shuffle=False,batch_size=3)

    for img in loader:
        dummy_img = np.array(img[0].permute(1,2,0))
        img_1 = dummy_img[...,0]
        img_2 = dummy_img[...,1]
        img_3 = dummy_img[...,2]
        img_4 = dummy_img[...,3]
        print('for debug')




    dataset.__len__()
    len(dataset.missing_date)

    print('END!')
