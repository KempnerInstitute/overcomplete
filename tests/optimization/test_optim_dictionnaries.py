import pytest

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from overcomplete import (
    SkDictionaryLearning,
    SkICA,
    SkKMeans,
    SkNMF,
    SkPCA,
    SkSVD,
    SkSparsePCA,
)

data_shape = (100, 10)
sample_data = torch.tensor(np.clip(np.random.randn(*data_shape), 0, None), dtype=torch.float32)

dataset = TensorDataset(sample_data)
data_loader = DataLoader(dataset, batch_size=10)


def method_encode_decode(model_class, sample_data, nb_concepts, extra_args=None):
    if extra_args is None:
        extra_args = {}
    model = model_class(nb_concepts=nb_concepts, **extra_args)
    model.fit(sample_data)

    encoded = model.encode(sample_data)
    assert encoded.shape[1] == nb_concepts, "Encoded output should have the correct shape"
    assert isinstance(encoded, torch.Tensor), "Encoded output should be a torch.Tensor"

    decoded = model.decode(encoded)
    assert decoded.shape[1] == data_shape[1], "Decoded output should have the correct shape"
    assert isinstance(decoded, torch.Tensor), "Decoded output should be a torch.Tensor"

    dictionary = model.get_dictionary()
    assert dictionary.shape[0] == nb_concepts, "Dictionary should have the correct shape"
    assert isinstance(dictionary, torch.Tensor), "Dictionary should be a torch.Tensor"


@pytest.mark.parametrize("model_class, extra_args", [
    (SkPCA, {}),
    (SkICA, {}),
    (SkNMF, {}),
    (SkKMeans, {}),
    (SkDictionaryLearning, {}),
    (SkSparsePCA, {}),
    (SkSVD, {}),
])
def test_optim_models_with_tensor(model_class, extra_args):
    nb_concepts = 2
    method_encode_decode(model_class, data_loader, nb_concepts, extra_args)


@pytest.mark.parametrize("model_class, extra_args", [
    (SkPCA, {}),
    (SkICA, {}),
    (SkNMF, {}),
    (SkKMeans, {}),
    (SkDictionaryLearning, {}),
    (SkSparsePCA, {}),
    (SkSVD, {}),
])
def test_optim_models_with_dataloader(model_class, extra_args):
    nb_concepts = 2
    method_encode_decode(model_class, data_loader, nb_concepts, extra_args)
