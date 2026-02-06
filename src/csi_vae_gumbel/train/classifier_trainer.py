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
        test_window_factor: int,
        gpu_id: int,
    ) -> None:
        """Initialize the Classifier Trainer."""
        self.__model = model.to(gpu_id)
        self.__dataloader = dataloader
        self.__vae = vae.to(gpu_id)  # No need to DDP the VAE as it's frozen
        self.__test_window_factor = test_window_factor
        self.__gpu_id = gpu_id
        self.__criterion = nn.CrossEntropyLoss()

        self.__optimizer = optim.Adam(self.__model.parameters())

    def __run_epoch(self) -> tuple[float, float]:
        metrics = torch.zeros(2, device=self.__gpu_id)

        for x, y in self.__dataloader:
            self.__optimizer.zero_grad()

            batch_size = x.shape[0]

            with torch.no_grad():
                _, z_hard, _ = self.__vae(x.to(self.__gpu_id))

                # (B, latent_dim, n_categories) → (B, latent_dim * n_categories)
                z_hard = z_hard.view(batch_size, -1)

                # (B, latent_dim * n_categories) → (B/factor, factor * latent_dim * n_categories)
                z_hard = z_hard.view(batch_size // self.__test_window_factor, -1)

            # Take one label every test_window_factor samples
            y_trimmed = y[:: self.__test_window_factor].to(self.__gpu_id)

            logits = self.__model(z_hard)
            loss = self.__criterion(logits, y_trimmed)

            with torch.no_grad():
                accuracy = (logits.argmax(dim=1) == y_trimmed).float().mean()

            metrics += torch.tensor([loss.detach(), accuracy], device=self.__gpu_id)

            loss.backward()
            self.__optimizer.step()

        metrics /= len(self.__dataloader)

        return tuple(metrics.tolist())

    def train(self, epochs: int) -> tuple[float, float]:
        """Train the classifier model for a specified number of epochs."""
        self.__model.train()
        self.__vae.eval()

        total_metrics = torch.zeros(2, device=self.__gpu_id)

        for _ in range(epochs):
            epoch_loss, epoch_accuracy = self.__run_epoch()
            total_metrics += torch.tensor([epoch_loss, epoch_accuracy], device=self.__gpu_id)

        total_metrics /= epochs

        return tuple(total_metrics.tolist())
