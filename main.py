from models import CustomModelWithPrefix
from dataset import load_emotion_dataset
from train import train_model
from evaluate import evaluate_model
from utils import set_seeds
from config import Config
from sklearn.model_selection import train_test_split
import os

def train_and_save_model(config):
    set_seeds(config.seed)
    model = CustomModelWithPrefix(config.model_name, config.n_landmarks, config.embedding_type)

    full_dataset = load_emotion_dataset(
        landmarks_file=config.landmarks_file,
        transcriptions_file=config.transcriptions_file,
        labels_file=config.labels_file,
        tokenizer=model.tokenizer,
        prompt=config.prompt
    )

    train_dataset, _ = train_test_split(full_dataset, test_size=config.eval_split, random_state=config.seed)

    trained_model = train_model(model, train_dataset, config)

    save_path = os.path.join(config.model_save_dir, 'trained_model.pth')
    torch.save(trained_model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def load_and_evaluate_model(config):
    set_seeds(config.seed)
    model = CustomModelWithPrefix(config.model_name, config.n_landmarks, config.embedding_type)
    
    load_path = os.path.join(config.model_save_dir, 'trained_model.pth')
    model.load_state_dict(torch.load(load_path))
    print(f"Model loaded from {load_path}")

    full_dataset = load_emotion_dataset(
        landmarks_file=config.landmarks_file,
        transcriptions_file=config.transcriptions_file,
        labels_file=config.labels_file,
        tokenizer=model.tokenizer,
        prompt=config.prompt
    )

    _, eval_dataset = train_test_split(full_dataset, test_size=config.eval_split, random_state=config.seed)

    eval_loss, avg_errors = evaluate_model(model, eval_dataset, config)

    

if __name__ == "__main__":
    config = Config()
    train_and_save_model(config)
    # load_and_evaluate_model(config)