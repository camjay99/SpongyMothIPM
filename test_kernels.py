import torch
import pytest

import SpongyMothIPM

def test_prediapause_kernel():
    pop0 = SpongyMothIPM.LnormPDF(SpongyMothIPM.xs, torch.tensor(0.2), torch.tensor(1.1))
    pop1 = SpongyMothIPM.kern_prediapause @ pop0
    pop1 = pop1.detach()
    assert torch.sum(pop0) == pytest.approx(torch.sum(pop1))

def test_postdiapause_kernel():
    pop0 = SpongyMothIPM.LnormPDF(SpongyMothIPM.xs, torch.tensor(0.2), torch.tensor(1.1))
    pop1 = SpongyMothIPM.kern_postdiapause @ pop0
    pop1 = pop1.detach()
    assert torch.sum(pop0) == pytest.approx(torch.sum(pop1))

def test_L1_kernel():
    pop0 = SpongyMothIPM.LnormPDF(SpongyMothIPM.xs, torch.tensor(0.2), torch.tensor(1.1))
    pop1 = SpongyMothIPM.kern_L1 @ pop0
    pop1 = pop1.detach()
    assert torch.sum(pop0) == pytest.approx(torch.sum(pop1))

def test_L2_kernel():
    pop0 = SpongyMothIPM.LnormPDF(SpongyMothIPM.xs, torch.tensor(0.2), torch.tensor(1.1))
    pop1 = SpongyMothIPM.kern_L2 @ pop0
    pop1 = pop1.detach()
    assert torch.sum(pop0) == pytest.approx(torch.sum(pop1))

def test_L3_kernel():
    pop0 = SpongyMothIPM.LnormPDF(SpongyMothIPM.xs, torch.tensor(0.2), torch.tensor(1.1))
    pop1 = SpongyMothIPM.kern_L3 @ pop0
    pop1 = pop1.detach()
    assert torch.sum(pop0) == pytest.approx(torch.sum(pop1))

def test_L4_kernel():
    pop0 = SpongyMothIPM.LnormPDF(SpongyMothIPM.xs, torch.tensor(0.2), torch.tensor(1.1))
    pop1 = SpongyMothIPM.kern_L4 @ pop0
    pop1 = pop1.detach()
    assert torch.sum(pop0) == pytest.approx(torch.sum(pop1))

def test_L5__male_kernel():
    pop0 = SpongyMothIPM.LnormPDF(SpongyMothIPM.xs, torch.tensor(0.2), torch.tensor(1.1))
    pop1 = SpongyMothIPM.kern_L5_male @ pop0
    pop1 = pop1.detach()
    assert torch.sum(pop0) == pytest.approx(torch.sum(pop1))

def test_L5_L6_female_kernel():
    pop0 = SpongyMothIPM.LnormPDF(SpongyMothIPM.xs, torch.tensor(0.2), torch.tensor(1.1))
    pop1 = SpongyMothIPM.kern_L5_L6_female @ pop0
    pop1 = pop1.detach()
    assert torch.sum(pop0) == pytest.approx(torch.sum(pop1))

def test_pupae_male_kernel():
    pop0 = SpongyMothIPM.LnormPDF(SpongyMothIPM.xs, torch.tensor(0.2), torch.tensor(1.1))
    pop1 = SpongyMothIPM.kern_pupae_male @ pop0
    pop1 = pop1.detach()
    assert torch.sum(pop0) == pytest.approx(torch.sum(pop1))

def test_pupae_female_kernel():
    pop0 = SpongyMothIPM.LnormPDF(SpongyMothIPM.xs, torch.tensor(0.2), torch.tensor(1.1))
    pop1 = SpongyMothIPM.kern_pupae_female @ pop0
    pop1 = pop1.detach()
    assert torch.sum(pop0) == pytest.approx(torch.sum(pop1))

def test_adult_kernel():
    pop0 = SpongyMothIPM.LnormPDF(SpongyMothIPM.xs, torch.tensor(0.2), torch.tensor(1.1))
    pop1 = SpongyMothIPM.kern_adult @ pop0
    pop1 = pop1.detach()
    assert torch.sum(pop0) == pytest.approx(torch.sum(pop1))