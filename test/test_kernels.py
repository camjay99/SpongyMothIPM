import torch
import pytest

import SpongyMothIPM.main as main

##################
# Eigenvalue Tests
##################
def test_prediapause_kernel():
    pop0 = main.LnormPDF(main.xs, torch.tensor(0.2), torch.tensor(1.1))
    pop1 = main.kern_prediapause @ pop0
    pop1 = pop1.detach()
    assert torch.sum(pop0) == pytest.approx(torch.sum(pop1))

def test_diapause_kernel():
    pop0_I = main.LnormPDF(main.from_x, torch.tensor(0.2), torch.tensor(1.1))
    pop0_D = main.LnormPDF(main.to_x, torch.tensor(0.2), torch.tensor(1.1))
    pop0 = torch.flatten(pop0_I * pop0_D)
    pop1 = main.kern_diapause_2D @ pop0
    pop1 = pop1.detach()
    assert torch.sum(pop0) == pytest.approx(torch.sum(pop1))

def test_postdiapause_kernel():
    pop0 = main.LnormPDF(main.xs, torch.tensor(0.2), torch.tensor(1.1))
    pop1 = main.kern_postdiapause @ pop0
    pop1 = pop1.detach()
    assert torch.sum(pop0) == pytest.approx(torch.sum(pop1))

def test_L1_kernel():
    pop0 = main.LnormPDF(main.xs, torch.tensor(0.2), torch.tensor(1.1))
    pop1 = main.kern_L1 @ pop0
    pop1 = pop1.detach()
    assert torch.sum(pop0) == pytest.approx(torch.sum(pop1))

def test_L2_kernel():
    pop0 = main.LnormPDF(main.xs, torch.tensor(0.2), torch.tensor(1.1))
    pop1 = main.kern_L2 @ pop0
    pop1 = pop1.detach()
    assert torch.sum(pop0) == pytest.approx(torch.sum(pop1))

def test_L3_kernel():
    pop0 = main.LnormPDF(main.xs, torch.tensor(0.2), torch.tensor(1.1))
    pop1 = main.kern_L3 @ pop0
    pop1 = pop1.detach()
    assert torch.sum(pop0) == pytest.approx(torch.sum(pop1))

def test_L4_kernel():
    pop0 = main.LnormPDF(main.xs, torch.tensor(0.2), torch.tensor(1.1))
    pop1 = main.kern_L4 @ pop0
    pop1 = pop1.detach()
    assert torch.sum(pop0) == pytest.approx(torch.sum(pop1))

def test_L5__male_kernel():
    pop0 = main.LnormPDF(main.xs, torch.tensor(0.2), torch.tensor(1.1))
    pop1 = main.kern_L5_male @ pop0
    pop1 = pop1.detach()
    assert torch.sum(pop0) == pytest.approx(torch.sum(pop1))

def test_L5_L6_female_kernel():
    pop0 = main.LnormPDF(main.xs, torch.tensor(0.2), torch.tensor(1.1))
    pop1 = main.kern_L5_L6_female @ pop0
    pop1 = pop1.detach()
    assert torch.sum(pop0) == pytest.approx(torch.sum(pop1))

def test_pupae_male_kernel():
    pop0 = main.LnormPDF(main.xs, torch.tensor(0.2), torch.tensor(1.1))
    pop1 = main.kern_pupae_male @ pop0
    pop1 = pop1.detach()
    assert torch.sum(pop0) == pytest.approx(torch.sum(pop1))

def test_pupae_female_kernel():
    pop0 = main.LnormPDF(main.xs, torch.tensor(0.2), torch.tensor(1.1))
    pop1 = main.kern_pupae_female @ pop0
    pop1 = pop1.detach()
    assert torch.sum(pop0) == pytest.approx(torch.sum(pop1))

def test_adult_kernel():
    pop0 = main.LnormPDF(main.xs, torch.tensor(0.2), torch.tensor(1.1))
    pop1 = main.kern_adult @ pop0
    pop1 = pop1.detach()
    assert torch.sum(pop0) == pytest.approx(torch.sum(pop1))

################
# Transfer Tests
################

def test_buffer_width():
    # This test assumes that once a distribution is close to zero, 
    # it does not increase again.
    kern_rounded = main.kern_prediapause * (main.kern_prediapause > 0.0001)
    kern_cum_sum = torch.cumsum(kern_rounded, dim=0)
    kern_arg_max = torch.argmax(kern_cum_sum, dim=0)
    kern_not_ended = kern_arg_max >= (main.n_bins - 1)
    total_not_ended = torch.sum(kern_not_ended * ~main.xs_for_transfer)
    assert total_not_ended == 0

def test_transfer():
    pop0 = main.LnormPDF(main.xs, torch.tensor(0.90), torch.tensor(1.1))
    pop1 = main.LnormPDF(main.xs, torch.tensor(0.4), torch.tensor(1.1))
    pop0_new, transfers = main.get_transfers(pop0)
    pop1_new = main.add_transfers(pop1, transfers)
    old_pop_total = torch.sum(pop0) + torch.sum(pop1)
    new_pop_total = torch.sum(pop0_new) + torch.sum(pop1_new)
    assert old_pop_total == pytest.approx(new_pop_total)

def test_transfer_diapause():
    # Population 0
    pop0_I = main.LnormPDF(main.from_x, torch.tensor(0.3), torch.tensor(1.1))
    pop0_D = main.LnormPDF(main.to_x, torch.tensor(0.9), torch.tensor(1.1))
    pop0 = torch.flatten(pop0_I * pop0_D)
    # Population 1
    pop1_I = main.LnormPDF(main.from_x, torch.tensor(0.4), torch.tensor(1.1))
    pop1_D = main.LnormPDF(main.to_x, torch.tensor(0.6), torch.tensor(1.1))
    pop1 = torch.flatten(pop1_I * pop1_D)
    # Test transfers
    pop0_new, transfers = main.get_transfers_diapause(pop0)
    pop1_new = main.add_transfers_diapause(pop1, transfers)
    old_pop_total = torch.sum(pop0) + torch.sum(pop1)
    new_pop_total = torch.sum(pop0_new) + torch.sum(pop1_new)
    assert old_pop_total == pytest.approx(new_pop_total)