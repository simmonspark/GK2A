import os
import datetime as dt
import numpy as np
import netCDF4 as nc
from sklearn.model_selection import train_test_split


# 안녕하세용
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

    return (np.array(ir_img).reshape(900, 900, 1)) / 12000.0, (np.array(rr_img).reshape(900, 900, 1)) / 10.0


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


def get_date_list(yyyymmdd_from, yyyymmdd_to, interval_hours=1):
    from_date = dt.datetime.strptime(yyyymmdd_from, '%Y%m%d')
    to_date = dt.datetime.strptime(yyyymmdd_to, '%Y%m%d')
    to_date += dt.timedelta(days=1)

    delta = dt.timedelta(hours=interval_hours)

    date_list = []
    while from_date < to_date:
        date_list.append(from_date.strftime('%Y%m%d%H%M'))
        from_date += delta

    return date_list
