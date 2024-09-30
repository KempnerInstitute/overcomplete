"""
Module for training Sparse Autoencoder (SAE) models.
"""

import time
from collections import defaultdict

import torch
from einops import rearrange

from ..metrics import l2, sparsity_eps


def _compute_reconstruction_error(x, x_hat):
    """
    Try to match the shapes of x and x_hat to compute the reconstruction error.
    If the input (x) shape is 4D assume it is (n, c, w, h), if it is 3D assume
    it is (n, t, c). Else, assume it is already flattened.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    x_hat : torch.Tensor
        Reconstructed tensor.

    Returns
    -------
    float
        Reconstruction error.
    """
    if len(x.shape) == 4 and len(x_hat.shape) == 2:
        x_flatten = rearrange(x, 'n c w h -> (n w h) c')
    elif len(x.shape) == 3 and len(x_hat.shape) == 2:
        x_flatten = rearrange(x, 'n t c -> (n t) c')
    else:
        assert x.shape == x_hat.shape, "Input and output shapes must match."
        x_flatten = x

    return torch.mean(l2(x_flatten - x_hat, -1)).item()


def _log_metrics(logs, model, z, loss, optimizer):
    """
    Log training metrics for the current training step.

    Parameters
    ----------
    logs : defaultdict
        Logs of training statistics.
    model : nn.Module
        The SAE model.
    z : torch.Tensor
        Encoded tensor.
    loss : torch.Tensor
        Loss value.
    optimizer : optim.Optimizer
        Optimizer for updating model parameters.
    """
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
    logs = defaultdict(list)

    for epoch in range(nb_epochs):
        model.train()
        start_time = time.time()
        epoch_loss = 0.0
        epoch_error = 0.0

        for batch in dataloader:
            x = batch[0].to(device, non_blocking=True)
            optimizer.zero_grad()

            z_pre, z, x_hat = model(x)
            loss = criterion(x, x_hat, z_pre, z, model.get_dictionary())

            # tfel: monitoring of NaNs in loss could be added here
            loss.backward()

            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            epoch_loss += loss.item()
            epoch_error += _compute_reconstruction_error(x, x_hat)

            if monitoring:
                _log_metrics(logs, model, z, loss, optimizer)

        avg_loss = epoch_loss / len(dataloader)
        avg_error = epoch_error / len(dataloader)
        epoch_duration = time.time() - start_time

        if monitoring:
            logs['avg_loss'].append(avg_loss)
            logs['time_epoch'].append(epoch_duration)
            print(f"Epoch[{epoch+1}/{nb_epochs}], Loss: {avg_loss:.4f}, "
                  f"Error: {avg_error:.4f}, Time: {epoch_duration:.4f} seconds")

    return logs


def train_sae_amp(model, dataloader, criterion, optimizer, scheduler=None,
                  nb_epochs=20, clip_grad=1.0, monitoring=True, device="cpu",
                  max_nan_fallbacks=5):
    """
    Train a Sparse Autoencoder (SAE) model with AMP and NaN fallback.
    Training in fp16 with AMP and fallback to fp32 for the rest of the current
    epoch if NaNs are detected 'max_nan_fallbacks' times. Next epoch will start
    again in fp16.

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
    max_nan_fallbacks : int, optional
        Maximum number of NaN fallbacks per epoch before disabling AMP, by default 5.

    Returns
    -------
    defaultdict
        Logs of training statistics.
    """
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    logs = defaultdict(list)

    for epoch in range(nb_epochs):
        model.train()
        start_time = time.time()
        epoch_loss = 0.0
        epoch_error = 0.0
        nan_fallback_count = 0

        for batch in dataloader:
            x = batch[0].to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=True):
                z_pre, z, x_hat = model(x)
                loss = criterion(x, x_hat, z_pre, z, model.get_dictionary())

            if torch.isnan(loss) or torch.isinf(loss):
                nan_fallback_count += 1
                if monitoring:
                    print(f"[Warning] NaN detected in loss at Epoch {epoch+1}, "
                          f"Iteration {len(logs['step_loss'])+1}. Retrying in full precision.")

                with torch.amp.autocast('cuda', enabled=False):
                    z, x_hat = model(x)
                    loss = criterion(x, x_hat, z, model.get_dictionary())

                if torch.isnan(loss) or torch.isinf(loss):
                    if monitoring:
                        print(f"[Error] Loss is NaN even in full precision at Epoch {epoch+1}, "
                              f"Iteration {len(logs['step_loss'])+1}. Skipping this batch.")
                    continue  # skip the current batch if NaN persists

                if nan_fallback_count >= max_nan_fallbacks:
                    if monitoring:
                        print(f"[Info] Maximum NaN fallbacks reached at Epoch {epoch+1}. "
                              f"Disabling AMP for the rest of this epoch.")
                    break

            scaler.scale(loss).backward()

            if clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None:
                scheduler.step()

            epoch_loss += loss.item()
            epoch_error += _compute_reconstruction_error(x, x_hat)

            if monitoring:
                _log_metrics(logs, model, z, loss, optimizer)

        avg_loss = epoch_loss / len(dataloader)
        avg_error = epoch_error / len(dataloader)
        epoch_duration = time.time() - start_time

        if monitoring:
            logs['avg_loss'].append(avg_loss)
            logs['time_epoch'].append(epoch_duration)
            logs['epoch_nan_fallbacks'].append(nan_fallback_count)
            print(f"Epoch[{epoch+1}/{nb_epochs}], Loss: {avg_loss:.4f}, "
                  f"Error: {avg_error:.4f}, Time: {epoch_duration:.4f} seconds")

    return logs
