import torch
import pytest

import SpongyMothIPM.kernels as kernels
from SpongyMothIPM.config import Config
import SpongyMothIPM.util as util

def test_collect_transfer_1D(first_instar, request):
    # Manually initialize pop with known # of individuals to transfer.
    # Note that we assume max x value is 1 here.
    first_instar.pop = torch.ones(first_instar.config.n_bins,
                                dtype=first_instar.config.dtype)
    transfers = first_instar.get_transfers()
    assert transfers == pytest.approx(1)

def test_collect_transfer_2D(diapause, request):
    # Manually initialize pop with known # of individuals to transfer.
    # Note that we assume max x value is 1 here.
    diapause.pop = torch.ones(diapause.config.shape,
                              dtype=diapause.config.dtype).flatten()
    transfers = diapause.get_transfers()
    assert transfers == pytest.approx(1.0 * diapause.config.n_bins)

# Get and add transfers is identical for all life stages besides
# diapause, therefore one test for non-diapause stages suffices.
@pytest.mark.parametrize('stage1,stage2',
                         [('prediapause','diapause'),
                          ('diapause','postdiapause'),
                          ('first_instar','second_instar')])
def test_transfer(stage1, stage2, mean, scale, request):
    # Get parametrized fixtures
    stage1 = request.getfixturevalue(stage1)
    stage2 = request.getfixturevalue(stage2)
    # Set up populations
    stage1.init_pop(mean, scale)
    stage2.init_pop(mean, scale)
    initial_pop = stage1.pop.sum() + stage2.pop.sum()
    # Transfer Individuals
    transfers = stage1.get_transfers()
    stage2.add_transfers(transfers)
    result_pop = stage1.pop.sum() + stage2.pop.sum()
    # No population should be lost through transferring
    assert initial_pop == pytest.approx(result_pop)

# Ensure that individuals do not go through multiple
# growth stages in one iteration.
# Only one transfer, as this function is shared by all life stages.
def test_growth_order(third_instar, fourth_instar, mean, scale):
    # Set up populations
    third_instar.init_pop(mean, scale)
    fourth_instar.pop = torch.zeros(fourth_instar.config.n_bins)
    # Run a trial time step
    transfers = third_instar.run_one_step([25.0])
    fourth_instar.run_one_step([25.0], transfers)
    # There should be no fourth instars beyond the newly developed ones.
    assert fourth_instar.pop[1:].sum() == pytest.approx(0)