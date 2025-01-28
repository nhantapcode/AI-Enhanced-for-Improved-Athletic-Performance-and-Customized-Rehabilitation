from datetime import timedelta
import sys
import time
import torch
import sklearn.metrics as metrics


class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler=None):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_acc = 0
        self.cache = {'train_loss': [], 
                      'train_acc': [], 
                      'val_loss': [], 
                      'val_acc': [],
                      'lr': []}

    def compute_matrix(self, preds, labels):
        accuracy = sum([1 for i, j in zip(preds, labels) if i == j]) / len(preds)
        precision = metrics.precision_score(labels, preds, average='binary')
        recall = metrics.recall_score(labels, preds, average='binary')
        f1 = metrics.f1_score(labels, preds, average='binary')

        return {
            'accuracy': round(accuracy, 7),
            'precision': round(precision, 7),
            'recall': round(recall, 7),
            'f1': round(f1, 7)
        }

    def save_checkpoint(self, path):
        acc = self.cache['val_acc'][-1]['accuracy']
        if acc >= self.best_acc:
            self.best_acc = acc
            params = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                "scheduler": None if self.scheduler is None else self.scheduler.state_dict(),
                'cache': self.cache
            }
            torch.save(params, path)
        print("[+] Save checkpoint successfully")

    def load_checkpoint(self, path):
        params = torch.load(path, weights_only=False)
        self.model.load_state_dict(params['model'])
        self.optimizer.load_state_dict(params["optimizer"])
        if self.scheduler:
            self.scheduler.load_state_dict(params["scheduler"])
        self.cache = params["cache"]
        print("[+] Load checkpoint successfully")

    def forward(self, dataloader, fw_model='train'):
        if fw_model == 'train':
            self.model.train()
        else:
            self.model.eval()

        cache = {'loss': [], 'acc': []}
        N = len(dataloader)

        for i, data in enumerate(dataloader, 1):
            if fw_model == 'train':
                self.optimizer.zero_grad()

            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            labels = labels.long()

            with torch.set_grad_enabled(fw_model == 'train'):
                outputs = self.model(inputs).squeeze(1)  # Đầu ra của mô hình (batch_size,)
                loss = self.criterion(outputs, labels)  # Dùng BCE Loss
                preds = torch.argmax(outputs, dim=1)  # Ngưỡng 0.5 cho nhị phân

                acc = self.compute_matrix(preds.tolist(), labels.tolist())
                cache['loss'].append(loss.item())
                cache['acc'].append(acc)

                if fw_model == 'train':
                    loss.backward()
                    self.optimizer.step()

            print("\r", end="")
            print(f"{fw_model.capitalize()} step: {i} / {N} - Acc: {acc['accuracy']}", end="" if i != N else "\n")

        loss = sum(cache['loss']) / len(cache['loss'])
        acc = [[i["accuracy"] for i in cache["acc"]], [i["precision"] for i in cache["acc"]], 
               [i["recall"] for i in cache["acc"]], [i["f1"] for i in cache["acc"]]]
        acc = {
            "accuracy": round(sum(acc[0]) / len(acc[0]), 7),
            "precision": round(sum(acc[1]) / len(acc[1]), 7),
            "recall": round(sum(acc[2]) / len(acc[2]), 7),
            "f1": round(sum(acc[3]) / len(acc[3]), 7)
        }

        self.cache[f"{fw_model}_loss"].append(loss)
        self.cache[f"{fw_model}_acc"].append(acc)

    def fit(self, train_loader, val_loader, epochs, checkpoint="./checkpoint/checkpoint.pt"):
        print(f"Running on: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        print(f"Total update step: {len(train_loader) * epochs}")

        for epoch in range(1, epochs + 1):
            start = time.time()
            print(f"Epoch: {epoch}")
            logs = []
            current_lr = f": {self.optimizer.param_groups[0]['lr']:e}"

            try:
                self.forward(train_loader, 'train')
                train_loss = round(self.cache['train_loss'][-1], 5)
                train_acc = [f"{k}: {v}" for k, v in self.cache['train_acc'][-1].items()]
                logs.append(f"\t=> Train epoch: loss: {train_loss} - " + " - ".join(train_acc))
            except KeyboardInterrupt:
                sys.exit()

            if val_loader is not None:
                try:
                    self.forward(val_loader, 'val')
                    val_loss = round(self.cache['val_loss'][-1], 5)
                    val_acc = [f"{k}: {v}" for k, v in self.cache['val_acc'][-1].items()]
                    logs.append(f"\t=> Val epoch: loss: {val_loss} - " + " - ".join(val_acc))
                except KeyboardInterrupt:
                    sys.exit()

            total_time = round(time.time() - start, 1)
            logs.append(f"\t=> Learning Rate: {current_lr} - Time: {timedelta(seconds=int(total_time))}/step\n")
            print("\n".join(logs))
            self.cache['lr'].append(current_lr)
            self.save_checkpoint(checkpoint)
