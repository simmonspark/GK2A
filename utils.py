import os
import datetime as dt
import numpy as np
import netCDF4 as nc
from sklearn.model_selection import train_test_split
import torch
from glob import glob
import os
import gc
import cv2 as cv
import pandas as pd


def get_date_list(yyyymmdd_from, yyyymmdd_to, interval_minutes=10):
    from_date = dt.datetime.strptime(yyyymmdd_from, '%Y%m%d')
    to_date = dt.datetime.strptime(yyyymmdd_to, '%Y%m%d')
    to_date += dt.timedelta(days=1)

    delta = dt.timedelta(minutes=interval_minutes)

    date_list = []
    while from_date < to_date:
        date_list.append(from_date.strftime('%Y%m%d%H%M'))
        from_date += delta

    return date_list


def find_missing_date_from_nc(root_data_path, yyyymmdd_from, yyyymmdd_to, interval_minutes=10):
    ref_date_list = get_date_list(yyyymmdd_from, yyyymmdd_to, interval_minutes)

    ir_date_list = [os.path.basename(x)[:-3] for x in glob(root_data_path + '/GK2A/IR/*.nc')]
    rr_date_list = [os.path.basename(x)[:-3] for x in glob(root_data_path + '/GK2A/RR/*.nc')]
    date_list = sorted(set(ir_date_list) & set(rr_date_list))

    missing_date_list = list(set(ref_date_list) - set(date_list))

    return missing_date_list


def get_nc_list(DIR='/media/sien/DATA/DATA/dataset/GK2A'):
    IR = os.path.join(DIR, 'IR/')
    RR = os.path.join(DIR, 'RR/')
    IR_STORE = []
    RR_STORE = []

    for dir, _, name in os.walk(IR):
        IR_STORE = [os.path.join(dir, file_name) for file_name in name]
    for dir, _, name in os.walk(RR):
        RR_STORE = [os.path.join(dir, file_name) for file_name in name]
    train_ir, test_ir, train_rr, test_rr = train_test_split(IR_STORE, RR_STORE, train_size=0.9, shuffle=False)
    sample_ir, test_ir, sample_rr, test_rr = train_test_split(test_ir, test_rr, shuffle=False, train_size=0.1)

    return train_ir, train_rr, sample_ir, sample_rr, test_ir, test_rr


def create_one_img_label(IR_PATH, RR_PATH):
    ir_data = nc.Dataset(IR_PATH)
    rr_data = nc.Dataset(RR_PATH)

    ir_img = ir_data.variables['image_pixel_values'][:]
    rr_img = rr_data.variables['RR'][:]

    return (np.array(ir_img).reshape(900, 900, 1)) / 12000.0, (np.array(rr_img).reshape(900, 900, 1)) / 100.0


def mkdir_p(directory):
    """Like mkdir -p ."""
    if not directory:
        return
    if directory.endswith("/"):
        mkdir_p(directory[:-1])
        return
    if os.path.isdir(directory):
        return
    mkdir_p(os.path.dirname(directory))
    os.mkdir(directory)


'''
https://www.kma.go.kr/kma/biz/forecast05.jsp

약한 비 : 1이상  ~ 3미만  mm/h
비 : 3이상 ~ 15미만  mm/h
강한 : 15이상 ~ 30미만  mm/h
매우 강한 비 : 30이상  mm/h

비의 종류는 4개입니다. label = mask1(class1) mask2(class2) mask3(class3) mask4(class4)

질문 1.  모델의 출력이 달라져야 하는지... classification이면 batch_size, width, height, depth(class 갯수) 
질문 2. 그럼 각 채널별로 softmax 해야함? 아니면 레이블 인코딩 해야하는지 
질문 3. 아니면 모델의 강우량 출력에 그냥 마스크를 씌워서 각 각 구하는 방식으로 하는지... 
#
'''


def create_classification_mask(tensor):
    '''
    레이블 텐서 또는 모델의 출력을 받아 4개의 mask를 생성합니다.
    노말라이즈 시킨 값은 잘못된 mask를 생성합니다 주의하세욤!
    :param tensor:
    :return: mask(tuple)|num-> 4
    '''
    if (len(tensor.shape) != 4):
        tensor = tensor.reshape(-1, 1, 224, 224)
    tensor = tensor.to('cpu').detach()
    mask1 = torch.where((tensor >= 1) & (tensor < 3), 1, 0)
    mask2 = torch.where((tensor >= 3) & (tensor < 15), 1, 0)
    mask3 = torch.where((tensor >= 15) & (tensor < 30), 1, 0)
    mask4 = torch.where((tensor >= 30), 1, 0)
    return np.array(mask1), np.array(mask2), np.array(mask3), np.array(mask4)


def get_ir_total_min_max(data_dir='/media/sien/DATA/DATA/dataset/GK2A'):
    ir_path = os.path.join(data_dir, 'IR/')
    rr_path = os.path.join(data_dir, 'RR/')
    re_save_path = '/media/sien/DATA/DATA/dataset/GK2A_224'
    file_name_list = [file_name for file_dir, _, file_name in os.walk(ir_path)][0]

    ir_value = [
        nc.Dataset(os.path.join(ir_path, i)).variables['image_pixel_values'][:]
        for i in file_name_list
    ]
    ir_max = np.array([i.max() for i in ir_value]).max()
    ir_min = np.array([i.min() for i in ir_value]).min()

    del ir_value
    gc.collect()
    rr_value = [
        nc.Dataset(os.path.join(rr_path, i)).variables['RR'][:]
        for i in file_name_list
    ]
    ir_max = np.array([i.max() for i in rr_value]).max()
    ir_min = np.array([i.min() for i in rr_value]).min()
    print()


def get_rr_total_min_max(data_dir='/media/sien/DATA/DATA/dataset/GK2A'):
    ir_path = os.path.join(data_dir, 'IR/')
    rr_path = os.path.join(data_dir, 'RR/')
    file_name_list = [file_name for file_dir, _, file_name in os.walk(ir_path)][0]
    max = -10000
    min = 10000

    for i in file_name_list:
        rr = nc.Dataset(os.path.join(rr_path, i))
        img = np.array(rr.variables['RR'][:])
        if img.max() > max:
            max = img.max()
        if img.min() < min:
            min = img.min()
        rr.close()
    print()


def EDA(data_dir='/media/sien/DATA/DATA/dataset/GK2A'):
    ir_path = os.path.join(data_dir, 'IR/')
    rr_path = os.path.join(data_dir, 'RR/')
    ir_name_list = [file_name for file_dir, _, file_name in os.walk(ir_path)][0]
    rr_name_list = [file_name for file_dir, _, file_name in os.walk(rr_path)][0]
    ir_path_list = [os.path.join(ir_path, i) for i in ir_name_list]
    rr_path_list = [os.path.join(rr_path, i) for i in rr_name_list]
    ir_df = []
    for idx, i in enumerate(ir_path_list):
        ds = nc.Dataset(i)
        img = np.array(ds.variables['image_pixel_values'][:])
        mean = img.mean()
        std = img.std()
        max = img.max()
        min = img.min()
        ir_df.append({'file_name': ir_name_list[idx], 'mean': mean, 'std': std, 'min': min, 'max': max})
    ir_df = pd.DataFrame(ir_df)
    ir_df.to_csv('/media/sien/Samsung_T5/와따시 파일/plot/ir_df.csv')

    rr_df = []
    for idx, i in enumerate(rr_path_list):
        ds = nc.Dataset(i)
        img = np.array(ds.variables['RR'][:])
        mean = img.mean()
        std = img.std()
        max = img.max()
        min = img.min()
        rr_df.append({'file_name': ir_name_list[idx], 'mean': mean, 'std': std, 'min': min, 'max': max})
    rr_df = pd.DataFrame(rr_df)
    rr_df.to_csv('/media/sien/Samsung_T5/와따시 파일/plot/rr_df.csv')
    print()
def show():
    pass



if __name__ == "__main__":
    print('for debug')
