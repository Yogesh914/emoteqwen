import numpy as np
import pywt
import os
from tqdm import tqdm
import pandas as pd

def calculate_optimal_level(signal_length, target_coeffs=30):
    """Calculate optimal decomposition level for target number of coefficients."""
    level = round(np.log2(signal_length / target_coeffs))
    return max(1, min(level, pywt.dwt_max_level(signal_length, 'haar')))

def adjust_to_target_length(coeffs, target_length=30, mode='symmetric'):
    """Adjust coefficient array to exact target length."""
    current_length = len(coeffs)
    
    if current_length == target_length:
        return coeffs
    elif current_length < target_length:
        pad_length = target_length - current_length
        
        if mode == 'symmetric':
            pad_left = pad_length // 2
            pad_right = pad_length - pad_left
            return np.pad(coeffs, (pad_left, pad_right), mode='symmetric')
        elif mode == 'edge':
            return np.pad(coeffs, (0, pad_length), mode='edge')
        else:  # 'zero' mode
            return np.pad(coeffs, (0, pad_length), mode='constant', constant_values=0)
    else:
        start_idx = (current_length - target_length) // 2
        return coeffs[start_idx:start_idx + target_length]

def normalize_coefficients(coeffs, level):
    """Normalize coefficients to account for different wavelet decomposition levels."""
    scale_factor = np.sqrt(2 ** level)
    return coeffs / scale_factor

def get_approx_coefficients(signal, n_coeffs=30, padding_mode='symmetric'):
    """Get fixed number of approximation coefficients for a signal."""
    wavelet = 'haar'
    level = calculate_optimal_level(len(signal), n_coeffs)
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    approx_coeffs = coeffs[0]
    normalized_coeffs = normalize_coefficients(approx_coeffs, level)
    return adjust_to_target_length(normalized_coeffs, n_coeffs, mode=padding_mode)

def process_video_landmarks(landmarks, n_coeffs=30):
    """Process landmarks for a single video, returning wavelet coefficients."""
    # Create lists to hold coefficients for each coordinate
    x_coeffs = []
    y_coeffs = []
    z_coeffs = []
    
    # Process each landmark
    for landmark_idx in range(landmarks.shape[1]):  # 478 landmarks
        # Get coefficients for each coordinate
        x_signal = landmarks[:, landmark_idx, 0]
        y_signal = landmarks[:, landmark_idx, 1]
        z_signal = landmarks[:, landmark_idx, 2]
        
        x_coeffs.append(get_approx_coefficients(x_signal, n_coeffs))
        y_coeffs.append(get_approx_coefficients(y_signal, n_coeffs))
        z_coeffs.append(get_approx_coefficients(z_signal, n_coeffs))
    
    # Stack into final shape (30, 478, 3)
    x_coeffs = np.stack(x_coeffs, axis=1)  # Shape: (30, 478)
    y_coeffs = np.stack(y_coeffs, axis=1)  # Shape: (30, 478)
    z_coeffs = np.stack(z_coeffs, axis=1)  # Shape: (30, 478)
    
    return np.stack([x_coeffs, y_coeffs, z_coeffs], axis=-1)  # Shape: (30, 478, 3)

def main():
    # Load your data
    fps_data = pd.read_csv("../data/all_metadata.csv")
    landmarks_folder = "../data/facial_landmarks/"
    
    all_wavelet_coeffs = []
    all_file_paths = []
    
    npy_files = [f for f in os.listdir(landmarks_folder) if f.endswith('.npy')]
    
    for file_name in tqdm(npy_files, desc="Processing videos", unit="file"):
        file_path = os.path.join(landmarks_folder, file_name)
        landmarks_data = np.load(file_path, allow_pickle=True)
        
        for entry in landmarks_data:
            video_path = entry["file_path"]
            landmarks = entry["landmarks"]
            
            if video_path in fps_data["file_path"].values:
                fps = fps_data[fps_data["file_path"] == video_path]['FPS'].values[0]
                
                video_coeffs = process_video_landmarks(landmarks, n_coeffs=10)
                
                all_wavelet_coeffs.append(video_coeffs)
                # all_file_paths.append(video_path)
    
    # Convert to numpy arrays
    all_wavelet_coeffs = np.array(all_wavelet_coeffs)  # Shape: (num_videos, 30, 478, 3)
    # all_file_paths = np.array(all_file_paths)
    
    # Save the results
    output_dir = "../data/facial_landmarks_dataset/wavelet_coeffs/"
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(f"{output_dir}/wl_coeffs_approx_10.npy", all_wavelet_coeffs)
    #np.save(f"{output_dir}/filepaths_wl.npy", all_file_paths)
    
    print(f"Processed {len(all_file_paths)} videos")
    print(f"Final shape of wavelet coefficients: {all_wavelet_coeffs.shape}")

if __name__ == "__main__":
    main()