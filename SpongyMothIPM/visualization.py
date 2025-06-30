from math import ceil

import matplotlib.pyplot as plt
import pandas as pd
import torch

from SpongyMothIPM.config import Config
import SpongyMothIPM.kernels as kernels

def tensor2d_imshow(tensor, n_bins, xmin, xmax):
    """Takes a 2d tensor and displays it as a heatmap."""
    fig, ax = plt.subplots()

    im = ax.imshow(tensor, cmap='Reds')
    fig.colorbar(im)
    ax.plot([-0.5, n_bins-0.5], [-0.5, n_bins-0.5], color='blue', zorder=2)
    ax.set_ylim(n_bins-0.5, -0.5)
    ax.set_xlim(-0.5, n_bins-0.5)

    positions = list(range(0, n_bins, 20))
    labels = [f"{i/n_bins*(xmax-xmin)+xmin:.2f}" for i in positions]
    ax.set_xticks(positions, labels)

    plt.show()

def tensor4d_to_2d_imshow(tensor, n_bins, sample_dims, sample_rates, dim_names, one_to_one=False):
    """Takes a 4d tensor and displays 2d slices as a heatmap."""
    if len(sample_dims) != len(sample_rates):
        raise Exception("sample_dims and sample_rates "
                        + "must have the same length: "
                        + f"{len(sample_dims)}, {len(sample_rates)}.")
    
    # Compute the number of graphs needed based on
    # sampling rate of each dimension.
    n_rows = n_bins // sample_rates[0]
    n_cols = n_bins // sample_rates[1]

    # Create figure
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)

    for i in range(n_rows):
        for j in range(n_cols):
            # Isolate slice
            slice = torch.index_select(tensor, 
                                       dim=sample_dims[1], 
                                       index=torch.tensor(j*sample_rates[1]))
            slice = torch.index_select(slice,
                                       dim=sample_dims[0],
                                       index=torch.tensor(i*sample_rates[0]))
            slice = slice.reshape((n_bins, n_bins))
            axes[i, j].imshow(slice, cmap='Reds')
            axes[i, j].scatter(j*sample_rates[1], i*sample_rates[0], color='black', s=5)
            # axes[i, j].set_xticks([])
            # axes[i, j].set_yticks([])
            axes[i, j].set_ylim([n_bins-0.5, -0.5])
            if j == 0:
                axes[i, j].set_ylabel(f"{dim_names[0]} = {i*sample_rates[0]}")
            if i == n_rows-1:
                axes[i, j].set_xlabel(f"{dim_names[1]} = {j*sample_rates[1]}")
            if one_to_one:
                axes[i, j].plot([-0.5, n_bins-0.5], [-0.5, n_bins-0.5], color='blue', zorder=2)

    plt.show()

def project_plot(kernel, pop, xs, num_gens):
    fig, ax = plt.subplots()

    ax.plot(xs, pop, label=f'Gen: {0}')

    for n in range(num_gens):
        pop = kernel @ pop
        ax.plot(xs, pop, label=f'Gen: {n+1}')
    
    ax.legend()
    plt.show()

def plot_eigenvector(tensor, n_bins):
    test = torch.ones(tensor.shape[0])
    fig, ax = plt.subplots()
    for i in range(100):
        test = tensor @ test
        if i % 10 == 0:
            D_test = test.reshape((n_bins, n_bins)).sum(dim=1)
            ax.plot(D_test)
    plt.show()

def compute_abundances(df):
    return df.sum(axis=1)
    
def plot_abundances(dfs, names, validation=None):
    fig, ax = plt.subplots()
    for i, df in enumerate(dfs):
        ax.plot(compute_abundances(df), label=names[i])
    if validation is not None:
        ax.plot(validation, label='validation')
    ax.legend()
    plt.show()

def _plot_age_dists_1D(ax, df, times):
    for time in times:
        ax.plot(df.iloc[time], label=time)
    ax.legend()

def _plot_age_dists_2D(ax, df, times, n_bins):
    for time in times:
        row = df.iloc[time].to_numpy().reshape((n_bins, n_bins))
        row = row.sum(axis=0)
        ax.plot(row, label=time)
    ax.legend()

def plot_age_dists(dfs, twoD=None, bins=None, start=0, end=-1, step=1):
    if twoD is None:
        twoD = [False]*len(dfs)
    if end == -1:
        end = len(dfs[0])
    times = list(range(start, end, step))
    ncols = min(3, len(dfs))
    nrows = ceil(len(dfs)/ncols)
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, squeeze=False)

    for i in range(nrows):
        for j in range(ncols):
            if i*ncols + j < len(dfs):
                if twoD[i*ncols + j]:
                    _plot_age_dists_2D(axes[i, j], 
                                       dfs[i*ncols + j], 
                                       times,
                                       bins[i*ncols + j])
                else:
                    _plot_age_dists_1D(axes[i, j],
                                       df[i*ncols + j],
                                       times)
    plt.show()



if __name__ == '__main__':
    config = Config()


    # pop0_I = SpongyMothIPM.LnormPDF(SpongyMothIPM.from_x, torch.tensor(0.2), torch.tensor(1.1))
    # pop0_D = SpongyMothIPM.LnormPDF(SpongyMothIPM.to_x, torch.tensor(0.4), torch.tensor(1.1))
    # kern_test = torch.nan_to_num(SpongyMothIPM.kern_diapause_2D)
    # pop0 = torch.flatten(pop0_I * pop0_D)
    # pop1 = kern_test @ pop0
    # pop0 = torch.reshape(pop0, SpongyMothIPM.shape)
    # pop1 = torch.reshape(pop1, SpongyMothIPM.shape)
    # tensor2d_imshow(pop0.detach(), SpongyMothIPM.n_bins)
    # tensor2d_imshow(pop1.detach(), SpongyMothIPM.n_bins)


    # pop = SpongyMothIPM.LnormPDF(SpongyMothIPM.xs, torch.tensor(0.2), torch.tensor(1.1))
    # project_plot(SpongyMothIPM.kern_postdiapause.detach(), 
    #              pop, 
    #              SpongyMothIPM.xs, 
    #              10)

    # stage = kernels.Diapause(config)
    # kernel = stage.build_kernel([30.0], twoD=False).detach()
    # tensor4d_to_2d_imshow(
    #     kernel,
    #     100,
    #     (0, 1),
    #     (20, 20),
    #     ("I", "D"),
    #     one_to_one=True)
    
    # stage = kernels.Diapause(config)
    # kernel1 = stage.build_kernel([0.0]).detach()
    
    # tensor2d_imshow(kernel1, 
    #                 config.n_bins*config.n_bins,
    #                 config.min_x,
    #                 config.max_x)

    # stage = kernels.Diapause(config)
    # kernel = stage.build_kernel([30.0]).detach()

    # plot_eigenvectors(kernel, config.n_bins)

    df = pd.read_csv('./outputs/test.csv', header=0, index_col=0)
    plot_abundances([df], ['Diapause'])
    plot_age_dists([df], [True], [100], 260, 280, 2)
