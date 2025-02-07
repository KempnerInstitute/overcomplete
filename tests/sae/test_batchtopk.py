import pytest
import torch
import torch.nn as nn

from overcomplete.sae import BatchTopKSAE

from ..utils import epsilon_equal


@pytest.fixture
def dummy_model():
    # encoder module is identity that return twice the input
    class IdentityModule(nn.Module):
        def forward(self, x):
            return x, x
    identity_module = IdentityModule()
    model = BatchTopKSAE(input_shape=3, nb_concepts=3, top_k=3, encoder_module=identity_module)
    return model


def test_dummy_model(dummy_model):
    assert dummy_model.top_k == 3
    assert dummy_model.threshold_momentum == 0.9
    assert dummy_model.running_threshold is None

    codes = torch.tensor([[1., 2., 3.],
                          [4., 5., 6.]], dtype=torch.float32)
    pre_codes, z = dummy_model.encode(codes)

    gt = torch.tensor([[0., 0., 0.],
                       [4., 5., 6.]], dtype=torch.float32)

    assert epsilon_equal(pre_codes, codes)
    assert epsilon_equal(z, gt)


def test_encode_train_threshold(dummy_model):
    dummy_model.train()
    codes = torch.tensor([[1., 3., 2.],
                          [6., 4., 5.]], dtype=torch.float32)
    pre_codes, z = dummy_model.encode(codes)

    # flattened codes: [1, 3, 2, 6, 4, 5]. top 3 are [6, 5, 4], so threshold is 4.
    expected_threshold = 4.
    expected_mask = (codes >= expected_threshold).float()
    expected_z = codes * expected_mask

    assert epsilon_equal(pre_codes, codes)
    assert epsilon_equal(z, expected_z)
    assert epsilon_equal(dummy_model.running_threshold, torch.tensor(expected_threshold))


def test_running_threshold_update(dummy_model):
    momentum = dummy_model.threshold_momentum  # default is 0.9
    dummy_model.train()

    # first call with one batch
    codes1 = torch.tensor([[1., 3., 2.],
                           [6., 4., 5.]], dtype=torch.float32)
    _ = dummy_model.encode(codes1)
    threshold1 = 4.  # top 3 of [1, 3, 2, 6, 4, 5] gives threshold 4

    # second call with a different batch
    codes2 = torch.tensor([[0., 10., 0.],
                           [0.,  0., 0.]], dtype=torch.float32)
    _ = dummy_model.encode(codes2)
    threshold2 = 0.  # flattened: [0,10,0,0,0,0], top 3 are [10, 0, 0]

    # running threshold updates as: momentum * threshold1 + (1 - momentum) * threshold2
    expected_running_threshold = momentum * threshold1 + (1 - momentum) * threshold2
    assert epsilon_equal(dummy_model.running_threshold,
                         torch.tensor(expected_running_threshold))


def test_encode_eval_mode(dummy_model):
    # first run a training pass to initialize the running threshold
    dummy_model.train()
    codes_train = torch.tensor([[1., 3., 2.],
                                [6., 4., 5.]], dtype=torch.float32)
    _ = dummy_model.encode(codes_train)  # running_threshold becomes 4

    # switch to evaluation mode and supply a new dummy code tensor
    dummy_model.eval()
    codes_eval = torch.tensor([[8., 5., 9.],
                               [7., 10., 6.]], dtype=torch.float32)
    _, z = dummy_model.encode(codes_eval)

    # in eval mode the running threshold is used (here, 4)
    # so all codes are above threshold and the output is the same as the input
    epsilon_equal(z, codes_eval)


def test_threshold_topk_all_elements(dummy_model):
    # when top_k equals the total number of elements, the threshold should be the minimum value
    # for a 2x3 code tensor the total elements are 6; we set top_k to 6.
    model = dummy_model
    model.top_k = 6
    model.train()
    codes = torch.tensor([[1., 2., 3.],
                          [4., 5., 6.]], dtype=torch.float32)
    _, z = model.encode(codes)

    expected_threshold = 1.
    expected_mask = (codes >= expected_threshold).float()
    expected_z = codes * expected_mask
    assert epsilon_equal(z, expected_z)
    assert epsilon_equal(model.running_threshold, torch.tensor(expected_threshold))


def test_threshold_topk_one(dummy_model):
    # when top_k is 1, the threshold should be the maximum value
    model = dummy_model
    model.top_k = 1
    model.train()
    codes = torch.tensor([[1., 2., 3.],
                          [4., 5., 6.]], dtype=torch.float32)
    _, z = model.encode(codes)

    expected_threshold = 6.
    expected_mask = (codes >= expected_threshold).float()
    expected_z = codes * expected_mask
    assert epsilon_equal(z, expected_z)
    assert epsilon_equal(model.running_threshold, torch.tensor(expected_threshold))


def test_gradient_flow(dummy_model):
    # verify that gradients flow through the encoder output even though the mask is detached
    dummy_model.top_k = 2
    dummy_model.train()
    # use an encoder that returns its input as both pre_codes and codes
    x = torch.tensor([[1., 2., 3.]], dtype=torch.float32, requires_grad=True)
    _, z = dummy_model.encode(x)
    loss = z.sum()
    loss.backward()

    # for input [1,2,3] and top_k=2, the top 2 values are [3,2] so threshold is 2 and mask is [0,1,1]
    expected_grad = torch.tensor([[0., 1., 1.]], dtype=torch.float32)
    assert x.grad is not None
    assert epsilon_equal(x.grad, expected_grad)
