import copy
import torch
from tqdm import tqdm


class Trainer:

    def __init__(self, model, device, criterion, optimizer, scheduler):

        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.best_weights = copy.deepcopy(model.state_dict())

        self.best_accuracy = 0.0

    def train_epoch(self, loader):

        self.model.train()

        running_loss = 0.0
        running_correct = 0
        total = 0

        loop = tqdm(loader)

        for images, labels in loop:

            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.criterion(outputs, labels)

            loss.backward()

            self.optimizer.step()

            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * images.size(0)

            running_correct += (preds == labels).sum().item()

            total += labels.size(0)

            loop.set_description(f"Loss {loss.item():.4f}")

        loss = running_loss / total
        acc = running_correct / total

        return loss, acc

    def validate(self, loader):

        self.model.eval()

        running_loss = 0.0
        running_correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in loader:

                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)

                loss = self.criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                running_loss += loss.item() * images.size(0)

                running_correct += (preds == labels).sum().item()

                total += labels.size(0)

        loss = running_loss / total
        acc = running_correct / total

        return loss, acc

    def fit(self, train_loader, valid_loader, epochs, model_path):

        patience = 5
        counter = 0

        history = {"train_loss": [], "valid_loss": [], "train_acc": [], "valid_acc": []}

        for epoch in range(epochs):

            print(f"\nEpoch {epoch+1}/{epochs}")

            train_loss, train_acc = self.train_epoch(train_loader)

            valid_loss, valid_acc = self.validate(valid_loader)

            self.scheduler.step()

            history["train_loss"].append(train_loss)
            history["valid_loss"].append(valid_loss)
            history["train_acc"].append(train_acc)
            history["valid_acc"].append(valid_acc)

            print(f"Train Loss : {train_loss:.4f}")
            print(f"Train Acc  : {train_acc:.4f}")
            print(f"Valid Loss : {valid_loss:.4f}")
            print(f"Valid Acc  : {valid_acc:.4f}")

            if valid_acc > self.best_accuracy:

                self.best_accuracy = valid_acc

                self.best_weights = copy.deepcopy(self.model.state_dict())

                torch.save(self.best_weights, model_path)

                counter = 0

                print("Best model saved.")

            else:

                counter += 1

                print(f"Early stopping counter: {counter}/{patience}")

            if counter >= patience:

                print("\nEarly stopping.")

                break

        self.model.load_state_dict(self.best_weights)

        return history
