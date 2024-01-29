import os
import requests
from tqdm import tqdm
from utils import mkdir_p, get_date_list


def download_data(save_root_path, date_from, date_to, interval_minutes, key):
    url_root = 'https://apihub.kma.go.kr/api/typ05/api/GK2A'
    raw_ir_data_dir = save_root_path + '/GK2A/IR'
    raw_rr_data_dir = save_root_path + '/GK2A/RR'

    if not os.path.exists(raw_ir_data_dir):
        mkdir_p(raw_ir_data_dir)

    if not os.path.exists(raw_rr_data_dir):
        mkdir_p(raw_rr_data_dir)

    date_list = get_date_list(date_from, date_to, interval_minutes=interval_minutes)
    for date in tqdm(date_list):
        ir_url = url_root + f'/LE1B/IR105/KO/data?date={date}&authKey={key}'
        rr_url = url_root + f'/LE2/RR/KO/data?date={date}&authKey={key}'

        ir_response = requests.get(ir_url)
        rr_response = requests.get(rr_url)

        save_ir_path = raw_ir_data_dir + f'/{date}.nc'
        save_rr_path = raw_rr_data_dir + f'/{date}.nc'

        if (ir_response.status_code == 200) and (rr_response.status_code == 200):
            with open(save_ir_path, 'wb') as ir_f:
                ir_f.write(ir_response.content)
            with open(save_rr_path, 'wb') as rr_f:
                rr_f.write(rr_response.content)
        else:
            print(f'Missing Date - {date}')


if __name__ == '__main__':
    save_root_path = '/media/sien/DATA/DATA/dataset'
    date_from = '20230626'  # yyyyMMdd
    date_to = '20230720'  # yyyyMMdd
    interval_minutes = 10  # minutes
    key = 'HHrdTz4JQhW63U8-CSIV9Q'  # KMA API Hub Authorization Key

    download_data(save_root_path, date_from, date_to, interval_minutes, key)

    print('END!')