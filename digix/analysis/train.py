from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from torch import Tensor
    from torch.utils.data import DataLoader
    from torch.nn import Module
    from torch.optim import Optimizer


def train(
    model: "Module",
    train: "DataLoader[Any]",
    criterion: "Module",
    optimizer: "Optimizer",
    epochs: int,
    silent: bool = False
):
    if epochs < 1:
        raise ValueError("Must specify at least 1 epoch")
    
    training_loss: list[float] = []

    for epoch in range(epochs):
        num_iters: int = 1
        epoch_loss: float = 0.0

        for data in train:
            inputs, labels = data
            optimizer.zero_grad()

            outputs: Tensor = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_iters += 1
        epoch_loss /= num_iters
        training_loss.append(epoch_loss)
        
        if not silent:
            print(f'Epoch {epoch + 1} training loss: {epoch_loss}')

    return training_loss