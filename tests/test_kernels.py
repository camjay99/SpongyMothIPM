import torch
import pytest

import SpongyMothIPM.kernels as kernels
from SpongyMothIPM.config import Config
import SpongyMothIPM.util as util

@pytest.mark.parametrize("life_stage",
                          [('prediapause'),
                           #('diapause'),
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
def test_eigenvalue(life_stage, temp, request):
    """Test if eigenvalue of the projection kernel is 1, indicating
       that population sizes neither grow nor shrink during
       individual growth."""
    # Get parametrized fixtures
    life_stage = request.getfixturevalue(life_stage)
    # Run test
    kernel = life_stage.build_kernel([temp]).detach()
    eigvals = torch.linalg.eigvals(kernel)
    eigvals = eigvals.abs()
    assert eigvals.max() == pytest.approx(1)

# Diapause kernel is tested separately due to potentially long
# eigenvalue calculations. Here we more weakly test that 
# all columns of 2D kernel sum to one.
def test_eigenvalue_diapause(diapause, temp):
    kernel = diapause.build_kernel([temp]).detach()
    col_sums = kernel.sum(dim=0)
    torch.testing.assert_close(col_sums, torch.ones(col_sums.shape)) 
    

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

def test_eigenvector(diapause, temp):
    kernel = diapause.build_kernel([temp]).detach()
    values, vectors = torch.eig(kernel)
    print(values)
    print(vectors)
    assert False

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