class Config:
    seed = 42

    model_name = "../models/gemma-2-2b"
    n_landmarks = 478
    embedding_type = 'mlp'

    landmarks_file = '../data/facial_landmarks_dataset/window30_step10/landmarks.npy'
    transcriptions_file = '../data/facial_landmarks_dataset/window30_step10/transcriptions.npy'
    labels_file = '../data/facial_landmarks_dataset/window30_step10/labels.npy'
    eval_split = 0.1
    prompt = "Classify emotion from the features above.\n"

    batch_size = 32
    learning_rate = 2e-5
    num_epochs = 10

    device = "cuda"
    
    log_dir = "./tensorboard_logs"
    model_save_dir = "./saved_models"
