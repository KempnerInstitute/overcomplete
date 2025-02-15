import os
import torch
import pytest
import shutil
import random
import tempfile
from torch.utils.data import DataLoader

from overcomplete.data import AsyncTensorDataset


@pytest.fixture
def temporary_data_dir():
    """Creates a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def create_sample_tensors(temporary_data_dir):
    """Creates sample `.pth` and `.pt` files in a temporary directory."""
    num_files = 20
    tensor_size = 1_000
    for i in range(num_files):
        tensor_data = torch.randn(tensor_size)
        torch.save(tensor_data, os.path.join(temporary_data_dir, f"file_{i}.pth"))


def test_initialization(temporary_data_dir):
    """Tests dataset initialization with different parameter configurations."""
    batch_size = 32
    dataset = AsyncTensorDataset(temporary_data_dir, batch_size=batch_size)

    assert dataset.data_dir == temporary_data_dir
    assert dataset.batch_size == batch_size
    assert dataset.shuffle_files is True
    assert dataset.file_stride is None
    assert dataset.max_prefetch_batches == 20
    assert dataset.num_workers == 4


def test_file_collection(temporary_data_dir, create_sample_tensors):
    """Ensures only `.pth` and `.pt` files are collected."""
    dataset = AsyncTensorDataset(temporary_data_dir)

    # Collect files manually
    expected_files = [
        os.path.join(temporary_data_dir, f"file_{i}.pth") for i in range(20)
    ]

    assert sorted(dataset.tensor_files) == sorted(expected_files)


def test_shuffle_behavior(temporary_data_dir, create_sample_tensors):
    """Ensures shuffle is applied correctly."""
    dataset1 = AsyncTensorDataset(temporary_data_dir, shuffle_files=False)
    dataset2 = AsyncTensorDataset(temporary_data_dir, shuffle_files=True)

    assert dataset1.tensor_files != dataset2.tensor_files  # Order should be different


def test_file_stride_behavior(temporary_data_dir, create_sample_tensors):
    """Ensures file skipping works correctly."""
    dataset = AsyncTensorDataset(temporary_data_dir, file_stride=2)

    assert len(dataset.tensor_files) == 10  # Only half of the files should be collected


def test_prefetch_queue_limit(temporary_data_dir, create_sample_tensors):
    """Ensures that the prefetch queue does not exceed the max limit."""
    dataset = AsyncTensorDataset(temporary_data_dir, max_prefetch_batches=3)

    assert dataset.prefetch_queue.maxsize == 3  # Must respect queue limit


def test_batch_sizes(temporary_data_dir, create_sample_tensors):
    """Verifies batches are of correct size."""
    bs = 64
    dataset = AsyncTensorDataset(temporary_data_dir, batch_size=bs)

    for b in dataset:
        assert len(b) == bs


def test_async_loading(temporary_data_dir, create_sample_tensors):
    """Ensures multiple threads correctly load data asynchronously."""
    dataset = AsyncTensorDataset(temporary_data_dir, num_workers=2, batch_size=32)

    # should be able to iterate over dataset
    batch_count = 0
    for batch in dataset:
        batch_count += 1
        if batch is None:
            break

    assert batch_count > 0


def test_invalid_directory():
    """Ensures dataset handles non-existent directory gracefully."""
    with pytest.raises(FileNotFoundError):
        dataset = AsyncTensorDataset("invalid/path")


def test_correct_finalization(temporary_data_dir, create_sample_tensors):
    """Ensures `None` sentinel is used correctly to signal iteration completion."""
    dataset = AsyncTensorDataset(temporary_data_dir, batch_size=16, num_workers=3)

    batch_count = 0
    last_batch = 42
    for batch in dataset:
        batch_count += 1
        if batch is None:
            last_batch = None
            break

    assert batch_count > 0  # ensures data was iterated
