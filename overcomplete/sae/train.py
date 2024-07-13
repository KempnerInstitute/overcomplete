import time
from collections import defaultdict

import torch

from ..metrics import l2, sparsity_eps


def train_sae(model, dataloader, criterion, optimizer, scheduler=None,
              nb_epochs=20, clip_grad=1.0, monitoring=True, device="cpu"):
    """
    Train a Sparse Autoencoder (SAE) model.

    Parameters
    ----------
    model : nn.Module
        The SAE model to train.
    dataloader : DataLoader
        DataLoader providing the training data.
    criterion : callable
        Loss function.
    optimizer : optim.Optimizer
        Optimizer for updating model parameters.
    scheduler : callable, optional
        Learning rate scheduler. If None, no scheduler is used, by default None.
    nb_epochs : int, optional
        Number of training epochs, by default 20.
    clip_grad : float, optional
        Gradient clipping value, by default 1.0.
    monitoring : bool, optional
        Whether to monitor and log training statistics, by default True.
    device : str, optional
        Device to run the training on, by default 'cpu'.

    Returns
    -------
    defaultdict
        Logs of training statistics.
    """
    num_batches = len(dataloader)
    logs = defaultdict(list)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(nb_epochs):
        start_time = time.time()
        epoch_loss = 0
        epoch_error = 0

        for batch in dataloader:
            if device != "cpu":
                x = batch[0].cuda(non_blocking=True)
            else:
                x = batch[0]

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                z, x_hat = model(x)
                loss = criterion(x, x_hat, z, model.get_dictionary())

            epoch_loss += loss.item()
            epoch_error += torch.mean(l2(x, x_hat, -1)).item()
            scaler.scale(loss).backward()

            if clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None:
                scheduler.step()

            if monitoring:
                logs['z'].append(z.detach()[::10])
                logs['z_l2'].append(l2(z).item())
                logs['z_sparsity'].append((z == 0.0).float().mean().item())
                logs['dictionary_sparsity'].append(sparsity_eps(model.get_dictionary(), threshold=1e-6).item())
                logs['dictionary_norm'].append(l2(model.get_dictionary()).item())
                logs['lr'].append(optimizer.param_groups[0]['lr'])
                logs['step_loss'].append(loss.item())

                for name, param in model.named_parameters():
                    if param.grad is not None:
                        logs[f'params_norm_{name}'].append(l2(param).item())
                        logs[f'params_grad_norm_{name}'].append(l2(param.grad).item())

        end_time = time.time()
        epoch_duration = end_time - start_time
        avg_loss = epoch_loss / num_batches
        avg_error = epoch_error / num_batches

        if monitoring:
            logs['avg_loss'].append(avg_loss)
            logs['time_epoch'].append(epoch_duration)
            logs['avg_loss'].append(avg_loss)
            print(f'Epoch [{epoch+1}/{nb_epochs}], Loss: {avg_loss:.4f}, Error: {avg_error}, Time: {epoch_duration:.4f} seconds')

    return logs
