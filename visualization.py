import matplotlib.pyplot as plt
import torch

import SpongyMothIPM

def tensor2d_imshow(tensor, n_bins):
    """Takes a 2d tensor and displays it as a heatmap."""
    fig, ax = plt.subplots()

    ax.imshow(tensor)
    ax.colorbar()
    ax.set_ylim(-0.5, n_bins-0.5)

    fig.show()

def tensor4d_to_2d_imshow(tensor, n_bins, sample_dims, sample_rates, dim_names):
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
                                       dim=sample_dims[0], 
                                       index=torch.tensor(j*sample_rates[0]))
            slice = torch.index_select(slice,
                                       dim=sample_dims[1],
                                       index=torch.tensor(i*sample_rates[1]))
            slice = slice.reshape((n_bins, n_bins))
            axes[i, j].imshow(slice, cmap='Reds')
            axes[i, j].scatter(i*sample_rates[1], j*sample_rates[0], color='black', s=5)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            axes[i, j].set_ylim([-0.5, n_bins-0.5])
            if j == 0:
                axes[i, j].set_ylabel(f"{dim_names[0]} = {i*sample_rates[0]}")
            if i == n_rows-1:
                axes[i, j].set_xlabel(f"{dim_names[1]} = {j*sample_rates[1]}")

    plt.show()

def project_plot(kernel, pop, xs, num_gens):
    fig, ax = plt.subplots()

    ax.plot(xs, pop, label=f'Gen: {0}')

    for n in range(num_gens):
        pop = kernel @ pop
        ax.plot(xs, pop, label=f'Gen: {n+1}')
    
    ax.legend()
    plt.show()

if __name__ == '__main__':
    pop = SpongyMothIPM.LnormPDF(SpongyMothIPM.xs, torch.tensor(0.2), torch.tensor(1.1))
    project_plot(SpongyMothIPM.kern_postdiapause.detach(), 
                 pop, 
                 SpongyMothIPM.xs, 
                 10)
    # tensor4d_to_2d_imshow(
    #     SpongyMothIPM.kern_diapause.detach(),
    #     100,
    #     (0, 1),
    #     (20, 20),
    #     ("I", "D"))
    print(SpongyMothIPM.mu_I_diapause)
