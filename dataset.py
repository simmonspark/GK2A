import numpy as np
from torch.utils.data import Dataset
from utils import create_one_img_label
import torch
import cv2 as cv
import torchvision.transforms as transforms


class Dataset(Dataset):
    def __init__(self,ir_list,rr_list):
        self.ir_list = ir_list
        self.rr_list = rr_list
        self.ir_store = []
        self.rr_store = []
        for ir,rr in zip(self.ir_list,self.rr_list):
            ir_img,rr_img = create_one_img_label(ir,rr)
            self.ir_store.append(cv.resize(ir_img,(224,224)))
            self.rr_store.append(cv.resize(rr_img,(224,224)))

    def __len__(self):
        return len(self.ir_list)

    def __getitem__(self, idx):

        ir = self.ir_store[idx]
        rr = self.rr_store[idx]

        ir = torch.Tensor(ir)
        rr = torch.Tensor(rr)

        return ir.unsqueeze(0), rr.unsqueeze(0)
