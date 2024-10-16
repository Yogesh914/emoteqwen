import numpy as np
from scipy.fft import fft
import pandas as pd
import os
from tqdm import tqdm

def process_landmarks(landmarks_folder, output_file, max_length=3000):
    """
    Process landmark files and create a padded numpy array.
    
    :param landmarks_folder: folder containing landmark npy files
    :param output_file: file to save the processed data
    :param reduction_method: method to reduce landmarks ('random' or 'every_other')
    :param target_num: number of landmarks to keep if using random sampling
    :param max_length: maximum length for padding (if None, will use the length of the longest sequence)
    """
    npy_files = [f for f in os.listdir(landmarks_folder) if f.endswith('.npy')]

    running_mean = None
    i = 0

    for file_name in tqdm(npy_files, desc="Processing files", unit="file"):
        all_landmarks = []
        file_path = os.path.join(landmarks_folder, file_name)
        landmarks_data = np.load(file_path, allow_pickle=True)
        
        for entry in landmarks_data:
            landmarks = entry['landmarks'].astype(np.float32)
            landmarks = fft(landmarks, n=max_length, axis=0)
            landmarks = np.abs(landmarks).astype(np.float32)
            all_landmarks.append(landmarks)

        all_landmarks = np.stack(all_landmarks)
        np.save(f"../data/facial_landmarks_dataset/landmark_fft_{i}.npy", all_landmarks)

        all_landmarks = np.mean(all_landmarks, axis=0)

        if i == 0:
            running_mean = all_landmarks
        else:
            running_mean = np.mean(np.stack([all_landmarks, running_mean]), axis=0)
        
        i+=1

        del landmarks_data, all_landmarks
    
    np.save(output_file, running_mean)

    

# Example usage
landmarks_folder = "../data/facial_landmarks/"
output_file = "../data/facial_landmarks_dataset/mean_fft.npy"
process_landmarks(landmarks_folder, output_file)
