import numpy as np
from torch.utils.data import Dataset
from utils import create_one_img_label
import torch
import cv2 as cv
import torchvision.transforms as transforms
from concurrent.futures import ProcessPoolExecutor, as_completed


def process_images(ir_path, rr_path):
    ir_img, rr_img = create_one_img_label(ir_path, rr_path)
    ir_resized = cv.resize(ir_img, (224, 224))
    rr_resized = cv.resize(rr_img, (224, 224))
    return ir_resized, rr_resized
class Dataset(Dataset):
    def __init__(self,ir_list,rr_list):
        self.ir_list = ir_list
        self.rr_list = rr_list
        self.ir_store = []
        self.rr_store = []
        with ProcessPoolExecutor(max_workers=15) as executor:
            future_to_images = {executor.submit(process_images, ir, rr): (ir, rr) for ir, rr in zip(self.ir_list, self.rr_list)}
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

        ir = torch.Tensor(ir)
        rr = torch.Tensor(rr)

        return ir.unsqueeze(0), rr.unsqueeze(0)
