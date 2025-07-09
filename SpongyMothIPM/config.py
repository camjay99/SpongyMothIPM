import torch

class Config():
    def __init__(self,
                 dtype=torch.float,
                 delta_t=1):
        #############
        # Torch Setup
        #############
        # In the future, we can specify what devices we want to use here.
        self.dtype = dtype

        ################
        # IPM Parameters
        ################
        self.delta_t = delta_t