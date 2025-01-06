import os
import sys
import yaml
import torch
import mlflow
import torchvision
import torch.nn as nn
from trainer import Trainer
import torch.optim as optim
from model import ImprovedNet
from torchinfo import summary
from utils import create_experiment
from mlflow.models import infer_signature
import torchvision.transforms as transforms
from visualize import plot_confusion_matrix
from sklearn.metrics import confusion_matrix


def load_params_from_yaml(file_path):
    with open(file_path, 'r') as f:
        params = yaml.safe_load(f)
    return params


def main():
    # Data loading with augmentation
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params = load_params_from_yaml('hyper_params.yaml')
    
    # Các tham số mô hình và huấn luyện
    num_workers = params['data_loader']['num_workers']
    batch_size =  params['data_loader']['batch_size']
    
    num_epochs = params['training']['num_epochs']

    learning_rate = params['optimizer']['lr']
    momentum = params['optimizer']['momentum']

    weight_decay = float(params['optimizer']['weight_decay'])
    checkpoint_path= params['training']['checkpoint_path']

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10 specific stats
    ])

      # Increased batch size
    trainset = torchvision.datasets.CIFAR10(root='./test', train=True,
                                        download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                        shuffle=True, num_workers=num_workers)
    
    validset = torchvision.datasets.CIFAR10(root='./test', train=False,
                                        download=False, transform=transform_train)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size,
                                        shuffle=False, num_workers=num_workers)
    
    exp_id = create_experiment(
        name="pytorch_test",
        artifact_location="pytorch_test_artifact",
        tags={"env": "dev", "version": "1.0.0"}
    )
    

    model = ImprovedNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, criterion, device=device)
    
    if os.path.exists(checkpoint_path):
        print("[+] Found checkpoint. Loading model...")
        trainer.load_checkpoint(checkpoint_path)

    with mlflow.start_run(run_name="Pytorch_test", experiment_id=exp_id,log_system_metrics=True) as run:
        params = {
            "epochs": num_epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "loss_function": criterion.__class__.__name__,
            "optimizer": optimizer.__class__.__name__,
        }

        sample_inputs, _ = next(iter(trainloader))
        X = sample_inputs.to(device)
        signature = infer_signature(X.cpu().numpy(), model(X).detach().cpu().numpy())


        mlflow.log_params(params)
        with open("model_summary.txt", "w", encoding="utf-8") as f:
            f.write(str(summary(model)))
        mlflow.log_artifact("model_summary.txt")
        
    
        try:
            trainer.fit(trainloader, validloader, epochs=num_epochs)
            cm = confusion_matrix(trainer.all_labels, trainer.all_preds) #note về tên class mình chỉnh cần đúng hoặc sai thôi Sẽ thảo luận sau.
            plot_confusion_matrix(cm, trainset)

            # Log confusion matrix vào MLflow
            mlflow.log_artifact("confusion_matrix.png")

            mlflow.pytorch.log_model(model, "models",signature=signature)
        except KeyboardInterrupt:
            sys.exit()

if __name__ == "__main__":
    main()