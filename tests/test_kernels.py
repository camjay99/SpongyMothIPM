import torch
import pytest

import SpongyMothIPM.main as main
import SpongyMothIPM.kernels as kernels
from SpongyMothIPM.config import Config
import SpongyMothIPM.util as util

#################
# Define Fixtures
#################

@pytest.fixture
def config():
    return Config()

@pytest.fixture
def prediapause(config: Config):
    return kernels.Prediapause(config)

@pytest.fixture
def diapause(config: Config):
    return kernels.Diapause(config)

@pytest.fixture
def postdiapause(config: Config):
    return kernels.Postdiapause(config)

@pytest.fixture
def first_instar(config: Config):
    return kernels.FirstInstar(config)

@pytest.fixture
def second_instar(config: Config):
    return kernels.SecondInstar(config)

@pytest.fixture
def third_instar(config: Config):
    return kernels.ThirdInstar(config)

@pytest.fixture
def fourth_instar(config: Config):
    return kernels.FourthInstar(config)

@pytest.fixture
def male_late_instar(config: Config):
    return kernels.MaleFifthInstar(config)

@pytest.fixture
def female_late_instar(config: Config):
    return kernels.FemaleFifthSixthInstar(config)

@pytest.fixture
def male_pupae(config: Config):
    return kernels.MalePupae(config)

@pytest.fixture
def female_pupae(config: Config):
    return kernels.FemalePupae(config)

@pytest.fixture
def adult(config: Config):
    return kernels.Adult(config)

@pytest.fixture
def pop_1D(config: Config):
    return util.LnormPDF(config.xs, 
                         torch.tensor(0.2), 
                         torch.tensor(1.1))

@pytest.fixture
def pop_2D(config: Config):
    pop_I = util.LnormPDF(config.from_x, 
                          torch.tensor(0.2), 
                          torch.tensor(1.1))
    pop_D = util.LnormPDF(config.to_x, 
                          torch.tensor(0.2), 
                          torch.tensor(1.1))
    return torch.flatten(pop_I * pop_D)
################
# Tests
################

@pytest.mark.parametrize("life_stage,pop",
                          [('prediapause', 'pop_1D'),
                           ('diapause', 'pop_2D'),
                           ('postdiapause', 'pop_1D'),
                           ('first_instar', 'pop_1D'),
                           ('second_instar', 'pop_1D'),
                           ('third_instar', 'pop_1D'),
                           ('fourth_instar', 'pop_1D'),
                           ('male_late_instar', 'pop_1D'),
                           ('female_late_instar', 'pop_1D'),
                           ('male_pupae', 'pop_1D'),
                           ('female_pupae', 'pop_1D'),
                           ('adult', 'pop_1D')])
def test_eigenvalue(life_stage, pop, request):
    # Get parametrized fixtures
    life_stage = request.getfixturevalue(life_stage)
    pop = request.getfixturevalue(pop)
    # Run test
    kernel = life_stage.build_kernel([15.0]).detach()
    next_pop = kernel @ pop
    print(next_pop)
    assert torch.sum(pop) == pytest.approx(torch.sum(next_pop))

################
# Transfer Tests
################

# def test_buffer_width():
#     # This test assumes that once a distribution is close to zero, 
#     # it does not increase again.
#     kern_rounded = main.kern_prediapause * (main.kern_prediapause > 0.0001)
#     kern_cum_sum = torch.cumsum(kern_rounded, dim=0)
#     kern_arg_max = torch.argmax(kern_cum_sum, dim=0)
#     kern_not_ended = kern_arg_max >= (main.n_bins - 1)
#     total_not_ended = torch.sum(kern_not_ended * ~main.xs_for_transfer)
#     assert total_not_ended == 0

# def test_transfer():
#     pop0 = main.LnormPDF(main.xs, torch.tensor(0.90), torch.tensor(1.1))
#     pop1 = main.LnormPDF(main.xs, torch.tensor(0.4), torch.tensor(1.1))
#     pop0_new, transfers = main.get_transfers(pop0)
#     pop1_new = main.add_transfers(pop1, transfers)
#     old_pop_total = torch.sum(pop0) + torch.sum(pop1)
#     new_pop_total = torch.sum(pop0_new) + torch.sum(pop1_new)
#     assert old_pop_total == pytest.approx(new_pop_total)

# def test_transfer_diapause():
#     # Population 0
#     pop0_I = main.LnormPDF(main.from_x, torch.tensor(0.3), torch.tensor(1.1))
#     pop0_D = main.LnormPDF(main.to_x, torch.tensor(0.9), torch.tensor(1.1))
#     pop0 = torch.flatten(pop0_I * pop0_D)
#     # Population 1
#     pop1_I = main.LnormPDF(main.from_x, torch.tensor(0.4), torch.tensor(1.1))
#     pop1_D = main.LnormPDF(main.to_x, torch.tensor(0.6), torch.tensor(1.1))
#     pop1 = torch.flatten(pop1_I * pop1_D)
#     # Test transfers
#     pop0_new, transfers = main.get_transfers_diapause(pop0)
#     pop1_new = main.add_transfers_diapause(pop1, transfers)
#     old_pop_total = torch.sum(pop0) + torch.sum(pop1)
#     new_pop_total = torch.sum(pop0_new) + torch.sum(pop1_new)
#     assert old_pop_total == pytest.approx(new_pop_total)