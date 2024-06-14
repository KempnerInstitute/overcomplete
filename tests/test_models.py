import pytest
from PIL import Image

from overcomplete.models import DinoV2Model, SigLIPModel, ViTModel


def create_test_image():
    image = Image.new('RGB', (224, 224), color='red')
    return image


@pytest.fixture
def test_image():
    return create_test_image()


def test_dinov2_model(test_image):
    model = DinoV2Model(use_half=False, device='cpu')
    test_image = model.preprocess(test_image).unsqueeze(0)
    output = model.forward_features(test_image)
    assert output.shape == (1, 256, 384)


def test_siglip_model(test_image):
    model = SigLIPModel(use_half=False, device='cpu')
    test_image = model.preprocess(test_image).unsqueeze(0)
    output = model.forward_features(test_image)
    assert output.shape == (1, 196, 768)


def test_vit_model(test_image):
    model = ViTModel(use_half=False, device='cpu')
    test_image = model.preprocess(test_image).unsqueeze(0)
    output = model.forward_features(test_image)
    assert output.shape == (1, 197, 768)
