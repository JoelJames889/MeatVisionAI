from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import torch


class Evaluator:

    def __init__(self, model, device, class_names):

        self.model = model
        self.device = device
        self.class_names = class_names

    def evaluate(self, loader):

        self.model.eval()

        predictions = []
        labels_list = []

        with torch.no_grad():

            for images, labels in loader:

                images = images.to(self.device)

                outputs = self.model(images)

                _, preds = torch.max(outputs, 1)

                predictions.extend(preds.cpu().numpy())

                labels_list.extend(labels.numpy())

        accuracy = accuracy_score(labels_list, predictions)

        precision = precision_score(
            labels_list, predictions, average="weighted", zero_division=0
        )

        recall = recall_score(
            labels_list, predictions, average="weighted", zero_division=0
        )

        f1 = f1_score(labels_list, predictions, average="weighted", zero_division=0)

        cm = confusion_matrix(labels_list, predictions)

        report = classification_report(
            labels_list, predictions, target_names=self.class_names, zero_division=0
        )

        print("\n" + "=" * 80)
        print("TEST RESULTS")
        print("=" * 80)

        print(f"Accuracy : {accuracy*100:.2f}%")
        print(f"Precision: {precision*100:.2f}%")
        print(f"Recall   : {recall*100:.2f}%")
        print(f"F1 Score : {f1*100:.2f}%")

        print("\nClassification Report\n")
        print(report)

        print("\nConfusion Matrix\n")
        print(cm)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm,
            "report": report,
        }
