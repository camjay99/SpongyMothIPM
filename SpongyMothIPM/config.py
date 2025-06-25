import torch

class Config():
    def __init__(self,
                 dtype=torch.float,
                 n_bins=100,
                 min_x=0,
                 max_x=1,
                 delta_t=1):
        #############
        # Torch Setup
        #############
        # In the future, we can specify what devices we want to use here.
        self.dtype = dtype

        ################
        # IPM Parameters
        ################
        self.n_bins = n_bins # Resolution of physiological age for each stage
        self.min_x = min_x
        self.max_x = max_x
        self.delta_t = delta_t

        self.init_helpers()

    def init_helpers(self):
        ##################
        # Helper Variables
        ##################
        self.shape = (self.n_bins, self.n_bins)
        self.xs = torch.linspace(self.min_x, self.max_x, self.n_bins)
        self.xs_for_transfer = self.xs >= 1
        self.input_xs = torch.zeros_like(self.xs)
        self.input_xs[0] = 1
        self.from_x = torch.reshape(self.xs, (1, self.n_bins))
        self.to_x = torch.reshape(self.xs, (self.n_bins, 1))
        self.x_dif = torch.maximum(torch.tensor(0), self.to_x - self.from_x)

        # These are used for computing diapause kernel
        self.ys = torch.linspace(0.0001, 1, self.n_bins)
        self.from_I = torch.reshape(self.xs, (self.n_bins, 1, 1, 1))
        self.to_I = torch.reshape(self.xs, (1, 1, self.n_bins, 1))
        self.from_D = torch.reshape(self.ys, (1, self.n_bins, 1, 1))
        self.to_D = torch.reshape(self.ys, (1, 1, 1, self.n_bins))
        self.I_dif = torch.maximum(torch.tensor(0), self.to_I - self.from_I)
        self.D_dif = torch.maximum(torch.tensor(0), self.to_D - self.from_D)
        self.grid2d = torch.squeeze(torch.ones_like(self.from_I)*self.from_D)
        self.grid2d_for_transfer = self.grid2d >= 1
        self.input_grid2d = torch.zeros_like(self.grid2d)
        self.input_grid2d[-1, 0] = 1