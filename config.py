class Config:
    seed = 42

    model_name = "../models/gemma-2-2b"
    n_landmarks = 478
    embedding_type = 'mlp'

    landmarks_file = '../data/facial_landmarks_dataset/top30_fft.npy'
    transcriptions_file = '../data/facial_landmarks_dataset/transcriptions_fft.npy'
    labels_file = '../data/facial_landmarks_dataset/labels_fft.npy'

    eval_split = 0.1
    prompt = ""

    batch_size = 4
    learning_rate = 2e-4
    num_epochs = 30

    device = "cuda"
    
    log_dir = "./tensorboard_logs"
    model_save_dir = "./saved_models"
    best_model = "./saved_models/best_top30_model.pth"
