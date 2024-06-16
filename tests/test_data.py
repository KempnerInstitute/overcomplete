import os
import tempfile
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from overcomplete.data import load_directory, to_npf32, unwrap_dataloader


def test_load_directory():
    # ensure we can load all the image in a directory
    # and avoid files that are not images
    with tempfile.TemporaryDirectory() as tempdir:

        image1 = Image.new('RGB', (10, 10), color='red')
        image2 = Image.new('RGB', (10, 10), color='blue')

        image1_path = os.path.join(tempdir, 'image1.png')
        image2_path = os.path.join(tempdir, 'image2.png')

        image1.save(image1_path)
        image2.save(image2_path)

        # create a dummy non-image file
        non_image_path = os.path.join(tempdir, 'non_image.txt')
        with open(non_image_path, 'w') as f:
            f.write("This is a test file and not an image.")

        images = load_directory(tempdir)

        assert len(images) == 2
        assert images[0].size == (10, 10)
        assert images[1].size == (10, 10)


def test_to_npf32():
    # ensure we can convert torch tensor, numpy and pil image to np.float32
    tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float64)
    np_array = to_npf32(tensor)
    assert isinstance(np_array, np.ndarray)
    assert np_array.dtype == np.float32
    assert np.allclose(np_array, [[1, 2], [3, 4]])

    array = np.array([[5, 6], [7, 8]], dtype=np.float64)
    np_array = to_npf32(array)
    assert isinstance(np_array, np.ndarray)
    assert np_array.dtype == np.float32
    assert np.allclose(np_array, [[5, 6], [7, 8]])

    image = Image.new('RGB', (10, 10), color='red')
    np_array = to_npf32(image)
    assert isinstance(np_array, np.ndarray)
    assert np_array.dtype == np.float32
    assert np_array.shape == (10, 10, 3)


def test_unwrap_dataloader():
    # ensure we can unwrap a DataLoader into a single tensor
    tensor = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)
    dataset = TensorDataset(tensor)
    dataloader = DataLoader(dataset, batch_size=2)

    result_tensor = unwrap_dataloader(dataloader)

    assert isinstance(result_tensor, torch.Tensor)
    assert result_tensor.shape == torch.Size([3, 2])
    assert torch.allclose(result_tensor, tensor)
