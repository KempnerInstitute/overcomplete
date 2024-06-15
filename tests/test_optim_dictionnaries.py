import pytest

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from overcomplete.methods import (
    OptimPCA, OptimICA, OptimNMF, OptimKMeans,
    OptimDictionaryLearning, OptimSparsePCA, OptimSVD
)

data_shape = (100, 10)
sample_data = torch.tensor(np.clip(np.random.randn(*data_shape), 0, None), dtype=torch.float32)

dataset = TensorDataset(sample_data)
data_loader = DataLoader(dataset, batch_size=10)


def method_encode_decode(model_class, sample_data, n_components, extra_args=None):
    if extra_args is None:
        extra_args = {}
    model = model_class(n_components=n_components, **extra_args)
    model.fit(sample_data)

    encoded = model.encode(sample_data)
    assert encoded.shape[1] == n_components, "Encoded output should have the correct shape"
    assert isinstance(encoded, torch.Tensor), "Encoded output should be a torch.Tensor"

    decoded = model.decode(encoded)
    assert decoded.shape[1] == data_shape[1], "Decoded output should have the correct shape"
    assert isinstance(decoded, torch.Tensor), "Decoded output should be a torch.Tensor"

    dictionary = model.get_dictionary()
    assert dictionary.shape[0] == n_components, "Dictionary should have the correct shape"
    assert isinstance(dictionary, torch.Tensor), "Dictionary should be a torch.Tensor"


@pytest.mark.parametrize("model_class, extra_args", [
    (OptimPCA, {}),
    (OptimICA, {}),
    (OptimNMF, {}),
    (OptimKMeans, {}),
    (OptimDictionaryLearning, {"sparsity": 1.0}),
    (OptimSparsePCA, {"sparsity": 1.0}),
    (OptimSVD, {}),
])
def test_optim_models_with_tensor(model_class, extra_args):
    n_components = 2
    method_encode_decode(model_class, data_loader, n_components, extra_args)


@pytest.mark.parametrize("model_class, extra_args", [
    (OptimPCA, {}),
    (OptimICA, {}),
    (OptimNMF, {}),
    (OptimKMeans, {}),
    (OptimDictionaryLearning, {"sparsity": 1.0}),
    (OptimSparsePCA, {"sparsity": 1.0}),
    (OptimSVD, {}),
])
def test_optim_models_with_dataloader(model_class, extra_args):
    n_components = 2
    method_encode_decode(model_class, data_loader, n_components, extra_args)
