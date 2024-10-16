import numpy as np
from scipy.fft import fft
import os
from tqdm import tqdm

def process_fft_landmarks(magnitude_fft, mean_magnitudes, top_n=10):

    num_vids, max_length, num_landmarks, num_coords = magnitude_fft.shape

    # Initialize the final matrix
    final_matrix = np.zeros((num_vids, top_n, num_landmarks, num_coords))

    for coord in range(num_coords):
        coord_magnitudes = mean_magnitudes[:, :, coord]  # Shape: (max_length, 478)
        top_indices = np.argsort(coord_magnitudes, axis=0)[-top_n:][::-1]  # Shape: (top_n, 478)

        for vid in range(num_vids):
            for i in range(top_n):
                final_matrix[vid, i, :, coord] = magnitude_fft[
                    vid, top_indices[i], np.arange(num_landmarks), coord
                ]

    return final_matrix

def process_all_files(landmarks_folder, mean_magnitudes_file, top_n=10):
    mean_magnitudes = np.load(mean_magnitudes_file)
    final_results = []
    npy_files = [f for f in os.listdir(landmarks_folder) if f.endswith('.npy')]

    for file_name in tqdm(npy_files, desc="Processing data", unit="file"):
            landmarks_array = np.load(os.path.join(folder_path, file_name))
            result = process_fft_landmarks(landmarks_array, mean_magnitudes, top_n)
            final_results.append(result)

    final_results = np.stack(final_results, axis=0)
    return final_results


folder_path = "../data/facial_landmarks_dataset/landmark_fft/"
top_n = 30
mean_magnitudes_file = "../data/facial_landmarks_dataset/mean_fft.npy"
result = process_all_files(folder_path, mean_magnitudes_file, top_n)
print(result.shape)  # Should be (num_vids, top_n, 478, 3)
np.save("../data/facial_landmarks_dataset/top30_fft.npy", result)
