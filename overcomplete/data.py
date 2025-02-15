"""
Module for data loading and conversion utilities.
"""

import os
import random
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Thread

import numpy as np
import torch
from torch.utils.data import IterableDataset
from PIL import Image


def load_directory(directory):
    """
    Load all images from a directory.

    Parameters
    ----------
    dir : str
        Directory path.

    Returns
    -------
    list
        List of PIL images.
    """
    paths = os.listdir(directory)
    paths = [path for path in paths if path.endswith(('.jpg', '.jpeg', '.png'))]
    paths = sorted(paths)

    images = []
    for path in paths:
        try:
            img = Image.open(os.path.join(directory, path)).convert('RGB')
            images.append(img)
        except OSError:
            # skip files that are not images
            continue
    return images


def to_npf32(tensor):
    """
    Check if tensor is torch, ensure it is on CPU and convert to NumPy.

    Parameters
    ----------
    tensor : torch.Tensor or np.ndarray
        Input tensor.
    """
    # return as is if already npf32
    if isinstance(tensor, np.ndarray) and tensor.dtype == np.float32:
        return tensor
    # torch case
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy().astype(np.float32)
    # pil case (and other)
    return np.array(tensor).astype(np.float32)


def unwrap_dataloader(dataloader):
    """
    Unwrap a DataLoader into a single tensor.

    Parameters
    ----------
    dataloader : DataLoader
        DataLoader object.
    """
    return torch.cat([batch[0] if isinstance(batch, (tuple, list))
                      else batch for batch in dataloader], dim=0)


class AsyncTensorDataset(IterableDataset):
    """
    A dataset that streams tensors from `.pth` and `.pt` files asynchronously.
    This could be use to load sets of precomputed activations to train your SAE.
    This dataset loads tensor files in parallel, prefetches batches into a queue,
    and provides an efficient streaming interface for large-scale training.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing `.pth` or `.pt` tensor files.
    batch_size : int, optional
        Number of tensors to yield per batch.
    shuffle_files : bool, optional
        Whether to shuffle the file order before processing. Default is True.
    file_stride : int, optional
        Interval for selecting files (e.g., `2` processes every second file). Default is None.
    max_prefetch_batches : int, optional
        Maximum number of batches to store in the prefetch queue.
    num_workers : int, optional
        Number of threads for parallel file loading.
    device : str, optional
        Device of the queue, when iterating over the dataset, the tensors will
        be served on this device. Default is 'cpu'.
    """

    def __init__(self, data_dir, batch_size=2048, shuffle_files=True, file_stride=None,
                 max_prefetch_batches=20, num_workers=4, device='cpu'):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle_files = shuffle_files
        self.file_stride = file_stride
        self.max_prefetch_batches = max_prefetch_batches
        self.num_workers = num_workers
        self.device = device

        # collect all `.pth` and `.pt` tensor files in the directory
        self.tensor_files = [
            os.path.join(data_dir, fname)
            for fname in os.listdir(data_dir)
            if fname.endswith(('.pth', '.pt'))
        ]

        # optionally shuffle the file order
        if self.shuffle_files:
            random.shuffle(self.tensor_files)

        # apply file stride (skipping files)
        if self.file_stride is not None:
            self.tensor_files = self.tensor_files[::self.file_stride]

        # queue to store prefetched batches
        self.prefetch_queue = Queue(maxsize=self.max_prefetch_batches)

    def _load_tensor_file(self, file_path):
        """
        Loads a tensor file, splits it into batches, and enqueues them.

        Parameters
        ----------
        file_path : str
            Path to the `.pth` or `.pt` file.
        """
        # load and randomly shuffle tensors if needed
        tensors = torch.load(file_path, map_location='cpu')
        if self.shuffle_files:
            tensors = tensors[torch.randperm(tensors.size(0))]

        # serve only full batches
        num_tensors = (len(tensors) // self.batch_size) * self.batch_size
        for i in range(0, num_tensors, self.batch_size):
            batch = tensors[i: i + self.batch_size]
            batch = batch.to(self.device)
            self.prefetch_queue.put(batch)

    def _async_loader(self):
        """
        Loads tensor files asynchronously using multiple threads.

        This method submits file loading tasks to a thread pool and signals
        when all files have been processed by enqueuing a `None` sentinel.
        """
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for tensor_file in self.tensor_files:
                executor.submit(self._load_tensor_file, tensor_file)

        # signal end of loading
        self.prefetch_queue.put(None)

    def __iter__(self):
        """
        Iterates over batches of tensors, asynchronously prefetching them.

        Returns
        -------
        generator
            Yields tensor batches until all files are processed.
        """
        loader_thread = Thread(target=self._async_loader, daemon=True)
        loader_thread.start()

        while True:
            batch = self.prefetch_queue.get()
            if batch is None:
                break
            yield batch

        loader_thread.join()
