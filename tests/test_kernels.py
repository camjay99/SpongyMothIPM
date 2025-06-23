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

@pytest.fixture(params=[0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0])
def temp(request):
    return request.param

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
def test_eigenvalue(life_stage, pop, temp, request):
    """Test if eigenvalue of the projection kernel is 1, indicating
       that population sizes neither grow nor shrink during
       individual growth."""
    # Get parametrized fixtures
    life_stage = request.getfixturevalue(life_stage)
    pop = request.getfixturevalue(pop)
    # Run test
    kernel = life_stage.build_kernel([temp]).detach()
    next_pop = kernel @ pop
    assert torch.sum(pop) == pytest.approx(torch.sum(next_pop))

@pytest.mark.parametrize("life_stage",
                          [('prediapause'),
                           ('diapause'),
                           ('postdiapause'),
                           ('first_instar'),
                           ('second_instar'),
                           ('third_instar'),
                           ('fourth_instar'),
                           ('male_late_instar'),
                           ('female_late_instar'),
                           ('male_pupae'),
                           ('female_pupae'),
                           ('adult')])
def test_lower_triangular(life_stage, temp, request):
    """Test if projection kernels are lower trianguler, indicating
       that development only happens in one direction (i.e., there
       is no regression in development)."""
    # Get parametrized fixtures
    life_stage = request.getfixturevalue(life_stage)
    # Run test
    kernel = life_stage.build_kernel([temp]).detach()
    torch.testing.assert_close(kernel, torch.tril(kernel)) 

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