import torch
import pytest

import SpongyMothIPM.kernels as kernels
from SpongyMothIPM.config import Config
import SpongyMothIPM.util as util

@pytest.fixture
def config():
    with torch.no_grad():
        yield Config()

@pytest.fixture
def prediapause(config: Config):
    with torch.no_grad():
        return kernels.Prediapause(config)

@pytest.fixture
def diapause(config: Config):
    with torch.no_grad():
        return kernels.Diapause(config)

@pytest.fixture
def postdiapause(config: Config):
    with torch.no_grad():
        return kernels.Postdiapause(config)

@pytest.fixture
def first_instar(config: Config):
    with torch.no_grad():
        return kernels.FirstInstar(config)

@pytest.fixture
def second_instar(config: Config):
    with torch.no_grad():
        return kernels.SecondInstar(config)

@pytest.fixture
def third_instar(config: Config):
    with torch.no_grad():
        return kernels.ThirdInstar(config)

@pytest.fixture
def fourth_instar(config: Config):
    with torch.no_grad():
        return kernels.FourthInstar(config)

@pytest.fixture
def male_late_instar(config: Config):
    with torch.no_grad():
        return kernels.MaleFifthInstar(config)

@pytest.fixture
def female_late_instar(config: Config):
    with torch.no_grad():
        return kernels.FemaleFifthSixthInstar(config)

@pytest.fixture
def male_pupae(config: Config):
    with torch.no_grad():
        return kernels.MalePupae(config)

@pytest.fixture
def female_pupae(config: Config):
    with torch.no_grad():
        return kernels.FemalePupae(config)

@pytest.fixture
def adult(config: Config):
    with torch.no_grad():
        return kernels.Adult(config)

@pytest.fixture(params=[-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0])
def temp(request):
    return request.param

# 1.0 tests if mature individuals are handled properly
@pytest.fixture(params=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]) 
def mean(request):
    return request.param

@pytest.fixture(params=[1.1, 1.5, 2])
def scale(request):
    return request.param