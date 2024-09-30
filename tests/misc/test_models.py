import pytest
from PIL import Image

from overcomplete.models import DinoV2, SigLIP, ViT, ResNet, ConvNeXt


def create_test_image():
    image = Image.new('RGB', (224, 224), color='red')
    return image


@pytest.fixture
def test_image():
    return create_test_image()


def test_dinov2_model(test_image):
    model = DinoV2(use_half=False, device='cpu')
    test_image = model.preprocess(test_image).unsqueeze(0)
    output = model.forward_features(test_image)
    assert output.shape == (1, 256, 384)


def test_siglip_model(test_image):
    model = SigLIP(use_half=False, device='cpu')
    test_image = model.preprocess(test_image).unsqueeze(0)
    output = model.forward_features(test_image)
    assert output.shape == (1, 196, 768)


def test_vit_model(test_image):
    model = ViT(use_half=False, device='cpu')
    test_image = model.preprocess(test_image).unsqueeze(0)
    output = model.forward_features(test_image)
    assert output.shape == (1, 196, 768)


def test_resnet_model(test_image):
    model = ResNet(use_half=False, device='cpu')
    test_image = model.preprocess(test_image).unsqueeze(0)
    output = model.forward_features(test_image)
    assert output.shape == (1, 2048, 7, 7)


def test_convnext_model(test_image):
    model = ConvNeXt(use_half=False, device='cpu')
    test_image = model.preprocess(test_image).unsqueeze(0)
    output = model.forward_features(test_image)
    assert output.shape == (1, 768, 7, 7)
