import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix


def _plot_confusion_matrix(matrix: np.ndarray, class_names: list[str], out_dir: str) -> None:
    plt.figure(figsize=(10, 8))
    # Normalize by row (True Labels) to see percentages
    matrix_perc = matrix.astype("float") / matrix.sum(axis=1)[:, np.newaxis]

    sns.heatmap(
        matrix_perc,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Normalized Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/confusion_matrix.png")
    plt.show()


class Evaluator:
    """Evaluator for VAE training using a classifier."""

    def __init__(
        self,
        vae: torch.nn.Module,
        classifier: torch.nn.Module,
        dataloader: DataLoader,
        classes: list[str],
        out_dir: str,
        gpu_id: int,
    ) -> None:
        """Initialize the evaluator.

        Arguments:
            vae: The trained VAE model.
            classifier: The trained classifier model.
            dataloader: DataLoader for evaluation data.
            classes: List of class names.
            out_dir: Output directory for saving results.
            gpu_id: GPU identifier for computation.

        """
        self.__vae = vae
        self.__classifier = classifier
        self.__dataloader = dataloader
        self.__classes = classes
        self.__n_classes = len(classes)
        self.__out_dir = out_dir
        self.__gpu_id = gpu_id

    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate the VAE and classifier on the test dataset."""
        self.__classifier.eval()
        self.__vae.eval()

        accuracy_metric = MulticlassAccuracy(num_classes=self.__n_classes).to(self.__gpu_id)
        confusion_matrix_metric = MulticlassConfusionMatrix(num_classes=self.__n_classes).to(self.__gpu_id)

        for x, y in self.__dataloader:
            _, z_hard_vae, _ = self.__vae(x.to(self.__gpu_id))
            z_hard_vae = z_hard_vae.view(z_hard_vae.size(0), -1)
            logits = self.__classifier(z_hard_vae)
            preds = torch.argmax(logits, dim=1)

            accuracy_metric.update(preds, y.to(self.__gpu_id))
            confusion_matrix_metric.update(preds, y.to(self.__gpu_id))

        conf_matrix = confusion_matrix_metric.compute().cpu().numpy()
        _plot_confusion_matrix(conf_matrix, self.__classes, self.__out_dir)

        return accuracy_metric.compute().item()
