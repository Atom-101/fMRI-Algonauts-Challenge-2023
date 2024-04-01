import pandas as pd 
from multiprocessing import Pool
import tarfile
import numpy as np

import mne
import io
import cv2
from tqdm import tqdm
import os
import random
import string

from multiprocessing import Manager
manager = Manager()
lock = manager.Lock()

random.seed(0)

def generate_random_string(length):
    # Define the characters that can be used in the random string
    characters = string.ascii_letters + string.digits  # Letters and digits

    # Generate the random string by selecting characters randomly
    random_string = ''.join(random.choice(characters) for _ in range(length))

    return random_string

def done_callback(result):
    if isinstance(result, int):  # simple check to see if it's a progress update
        progress_bar.update(1)
        print(f'Done chunk: {result}')
    else:
        print(result)

def process_chunk(chunk, output_tar, chunk_num):
    try:
        # Process chunk
        print(f'Starting chunk: {chunk_num}')
        for row_idx, row in enumerate(chunk.itertuples(index=False)):
            eeg = np.array(row[:64000], dtype=np.float32).reshape(128, -1)
            label = int(row[64000])
            mnist_image = np.array(row[64003:64003+784], dtype=np.uint8).reshape(28,28)

            # random_names = ['bSP', 'Uf5', 'hT5', 'Dfp', 'd8H', 'Ry0', 'vkw', 'V1i', '06z', 'Cjt', 'vLP', 'JI8', 'ikk', 'fVQ', 'psd', 'yHd', 'n0f', 'K09', '3SU', '7yW', '7Uw', 'Slb', 'YUP', '1Fy', 'v0T', '0fc', 'Ii9', 'A6t', 'xmR', 'g7x', 'FMb', 'oD5', 'aYk', 'Qze', 'b6a', 'CPw', 'QNA', 'KLM', '7Pg', 'mso', 'Zfd', 'njv', 'gg7', 'Ng8', 'q9G', '5yf', 'aw8', 'tEG', '0l0', 'tya', '9Pe', 'Kus', 'F8x', 'xZq', 'b4C', 'Vuw', '7tL', 'Gzw', 'PSp', 'XAW', '80l', 'yiP', 'ThM', 'XTA', 'ccC', 'dAd', '9y2', '4gt', 'O9D', 'lOk', 'iu9', 'nlM', 'wQs', 'gcw', 'kyL', 'V2d', 'Aqm', 'oMX', '0s7', 'W1C', 'JPv', 'xMc', 'Jz4', 'ucu', '8UD', 'ZGE', 'BcS', '7mR', 'Ex8', 'rpz', 'U6x', 'vk5', '7ML', 'KHh', '4L1', 'QOY', 'H0b', 'kX6', 'za2', '7vE', '3I6', 'FQW', '9dX', 'IzO', 'LA4', 'u3C', '0p5', 'SZW', 'TBR', 'TG4', 'YzP', 'heQ', 'PXd', 'A4F', '8Yz', '4Oz', 'YOK', 'hpI', '2oy', '1hp', 'ZKa', 'KRz', 'd4G', 'tSb', 'NSs', 'e8R', 'mU9', 'G0R']
            # raw = mne.io.RawArray(eeg, mne.create_info(ch_names=random_names, sfreq=256, ch_types='eeg'), verbose=False)
            # low_cut = 5
            # hi_cut  = 95
            # eeg = raw.copy().filter(low_cut, hi_cut, verbose=False)._data
            eeg = eeg/1000

            eeg_buffer = io.BytesIO()
            mnist_buffer = io.BytesIO()
            label_buffer = io.BytesIO()
            
            np.save(eeg_buffer, eeg)
            np.save(label_buffer, np.array([label], dtype=np.int8))
            success, encoded_image = cv2.imencode('.png', mnist_image)
            if success:
                mnist_buffer.write(encoded_image.tobytes())
            else:
                raise ValueError()

            with lock:
                # Acquire lock
                with tarfile.open(output_tar, 'a') as tar:
                    file_name = f'{chunk_num}_{row_idx}_{generate_random_string(5)}'
                    # eeg dump
                    tar_info = tarfile.TarInfo(name=file_name +f'_{label:02d}.eeg.npy')
                    tar_info.size = len(eeg_buffer.getvalue())
                    eeg_buffer.seek(0)  # Reset buffer position to the beginning
                    tar.addfile(tar_info, eeg_buffer)

                    # mnist dump
                    tar_info = tarfile.TarInfo(name=file_name +f'_{label:02d}.png')
                    tar_info.size = len(mnist_buffer.getvalue())
                    mnist_buffer.seek(0)  # Reset buffer position to the beginning
                    tar.addfile(tar_info, mnist_buffer)

                    # label dump
                    tar_info = tarfile.TarInfo(name=file_name +f'_{label:02d}.label.npy')
                    tar_info.size = len(label_buffer.getvalue())
                    label_buffer.seek(0)  # Reset buffer position to the beginning
                    tar.addfile(tar_info, label_buffer)
    except Exception as e:
        return f"Error in process_chunk({chunk_num}): {str(e)}"
    return chunk_num
    
            
def parallel_process(csv_file, output_h5):
    global progress_bar
    progress_bar = tqdm(total=150000//1000)

    chunks = pd.read_csv(csv_file, chunksize=1000)

    with Pool() as pool:
        for i, chunk in enumerate(chunks):
            pool.apply_async(process_chunk, args=(chunk, output_h5, i), callback=done_callback)
    
        print(f'Max I: {i}')
        pool.close()
        pool.join()

if __name__ == '__main__':

    csv_file = '/fsx/proj-fmri/shared/eeg_mnist_raw/MindBigData2023_MNIST-8B/train.csv'
    output_h5 = '/fsx/proj-fmri/shared/eeg_mnist_train_1.tar'
    
    # csv_file = '/fsx/proj-fmri/shared/eeg_mnist_raw/MindBigData2023_MNIST-8B/test.csv'
    # output_h5 = '/fsx/proj-fmri/shared/eeg_mnist_test_1.tar'
    
    if os.path.exists(output_h5):
        os.remove(output_h5)
    
    parallel_process(csv_file, output_h5)