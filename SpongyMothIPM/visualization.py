import matplotlib.pyplot as plt
import torch

import SpongyMothIPM.main as main
import trial 

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
                                       dim=sample_dims[0], 
                                       index=torch.tensor(j*sample_rates[0]))
            slice = torch.index_select(slice,
                                       dim=sample_dims[1],
                                       index=torch.tensor(i*sample_rates[1]))
            slice = slice.reshape((n_bins, n_bins))
            axes[i, j].imshow(slice, cmap='Reds')
            axes[i, j].scatter(i*sample_rates[1], j*sample_rates[0], color='black', s=5)
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

if __name__ == '__main__':
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

    # print(SpongyMothIPM.mu_I_diapause)
    # test_dist = trial.kern_test.expand(100, 100, 100,100)
    # tensor4d_to_2d_imshow(
    #     test_dist.detach(),
    #     100,
    #     (1, 3),
    #     (20, 20),
    #     ("I", "D"),
    #     one_to_one=True)

    tensor2d_imshow(main.kern_adult.detach(), main.n_bins,
                    main.min_x,
                    main.max_x)
