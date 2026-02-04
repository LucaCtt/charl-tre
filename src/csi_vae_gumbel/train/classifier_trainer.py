import torch
from torch import nn, optim
from torch.utils.data import DataLoader


class ClassifierTrainer:
    """Trainer class for classifier model using Distributed Data Parallel (DDP)."""

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        vae: nn.Module,
        gpu_id: int,
    ) -> None:
        """Initialize the Classifier Trainer."""
        self.__model = model.to(gpu_id)
        self.__dataloader = dataloader
        self.__vae = vae.to(gpu_id)  # No need to DDP the VAE as it's frozen
        self.__gpu_id = gpu_id
        self.__criterion = nn.CrossEntropyLoss()

        self.__optimizer = optim.Adam(self.__model.parameters())

    def __run_epoch(self) -> tuple[float, float]:
        epoch_loss = 0.0
        epoch_accuracy = 0.0

        for x, y in self.__dataloader:
            self.__optimizer.zero_grad()

            with torch.no_grad():
                # Go from (batch_size, n_antennas, window_size, n_subcarriers)
                # to (batch_size / 3, 3, n_antennas, window_size, n_subcarriers) and get z
                x_temp = x.view(x.size(0) // 3, 3, x.size(1), x.size(2), x.size(3))
                z_hard = []
                for i in range(3):
                    _, z_hard_partial, _ = self.__vae(x_temp[:, i].to(self.__gpu_id))
                    z_hard_partial = z_hard_partial.view(z_hard_partial.size(0), -1)
                    z_hard.append(z_hard_partial)

            z_hard = torch.cat(z_hard, dim=0)

            logits = self.__model(z_hard)
            loss = self.__criterion(logits, y.to(self.__gpu_id))
            accuracy = (logits.argmax(dim=1) == y.to(self.__gpu_id)).float().mean()

            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()

            loss.backward()
            self.__optimizer.step()

        epoch_loss /= len(self.__dataloader)
        epoch_accuracy /= len(self.__dataloader)

        return epoch_loss, epoch_accuracy

    def train(self, epochs: int) -> tuple[float, float]:
        """Train the classifier model for a specified number of epochs."""
        self.__model.train()
        self.__vae.eval()

        total_loss = 0.0
        total_accuracy = 0.0

        for _ in range(epochs):
            epoch_loss, epoch_accuracy = self.__run_epoch()

            total_loss += epoch_loss
            total_accuracy += epoch_accuracy

        total_loss /= epochs
        total_accuracy /= epochs

        return total_loss, total_accuracy
