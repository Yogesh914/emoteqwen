class Config:
    seed = 42

    model_name = "../models/gemma-2-2b"
    n_landmarks = 478
    embedding_type = 'mlp'

    device = "cuda"
    eval_split = 0.1
    prompt = ""

    learning_rate = 2e-4
    num_epochs = 40
    batch_size = 4

    landmarks_file = '../data/facial_landmarks_dataset/wavelet_coeffs/wl_coeffs_approx_30.npy'
    transcriptions_file = '../data/facial_landmarks_dataset/transcriptions_fft.npy'
    labels_file = '../data/facial_landmarks_dataset/labels_fft.npy'

    log_dir = "./tensorboard_logs/wl_approx_30_us"
    model_save_dir = "./saved_models"
    best_model = "./saved_models/best_wl_30us_model.pth"