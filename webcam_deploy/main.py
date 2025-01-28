import os
import sys
import torch
import mlflow
import torch.nn as nn
from trainer import Trainer
from model import SimpleCNN
from utils import build_transforms
from utils import create_experiment
from dataloader import KaggleDataset
from mlflow.models import infer_signature
from torch.utils.data import DataLoader, random_split
from visualize import plot_confusion_matrix
from torchinfo import summary
from sklearn.metrics import confusion_matrix

def main():
    image_dirs = {
        "barbell_biceps_curl": "data/barbell_biceps_curl/image",
        "hammer_curl": "data/hammer_curl/image"
    }
    transform = build_transforms(is_train=True)
    dataset = KaggleDataset(image_dirs=image_dirs,transform=transform)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    # Chia tập train và valid (80% train, 20% valid)
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    # Tạo DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    # Tạo model
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = Trainer(model, criterion,optimizer, scheduler=None)

    checkpoint_path = "./checkpoint/checkpoint.pt"
    exp_id = create_experiment(
        name="TheSis25_Experiment",
        artifact_location="TheSis25_Artifact",
        tags={"env": "dev", "version": "1.0.0"}
    )

    if os.path.exists(checkpoint_path):
        print("[+] Found checkpoint. Loading model...")
        trainer.load_checkpoint(checkpoint_path)

    with mlflow.start_run(run_name="Pytorch_test", experiment_id=exp_id,log_system_metrics=True) as run:
        params = {
            "epochs": 5,
            "learning_rate": 0.001,
            "batch_size": 32,
            "loss_function": criterion.__class__.__name__,
            "optimizer": optimizer.__class__.__name__,
        }

        sample_inputs, _ = next(iter(train_loader))
        X = sample_inputs.to(device)
        signature = infer_signature(X.cpu().numpy(), model(X).detach().cpu().numpy())


        mlflow.log_params(params)
        with open("model_summary.txt", "w", encoding="utf-8") as f:
            f.write(str(summary(model)))
        mlflow.log_artifact("model_summary.txt")
        
    
        try:
            trainer.fit(train_loader, valid_loader, epochs=5)
            # cm = confusion_matrix(trainer.all_labels, trainer.all_preds) #note về tên class mình chỉnh cần đúng hoặc sai thôi Sẽ thảo luận sau.
            # plot_confusion_matrix(cm, dataset)

            # # Log confusion matrix vào MLflow
            # mlflow.log_artifact("confusion_matrix.png")

            mlflow.pytorch.log_model(model, "models",signature=signature)
        except KeyboardInterrupt:
            sys.exit()




if __name__ == "__main__":
    main()