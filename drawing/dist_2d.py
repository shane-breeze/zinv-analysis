import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.width', None)

def dist_2d(df, path, cfg):
    all_columns = list(df.index.names)
    columns_bins = [c for c in all_columns if "bin" in c]

    # Drop overflow and underflow
    df = df.reset_index(columns_bins)
    df = df.loc[(~np.isinf(df)).all(axis=1)]\
            .set_index(columns_bins, append=True)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(5.76, 4.8))

    xbin_low = df.index.get_level_values("bin0_low")
    xbin_upp = df.index.get_level_values("bin0_upp")
    ybin_low = df.index.get_level_values("bin1_low")
    ybin_upp = df.index.get_level_values("bin1_upp")

    xbins = (xbin_low + xbin_upp)/2
    ybins = (ybin_low + ybin_upp)/2

    xbins_unique = np.array(list(xbin_low.unique())+[xbin_upp.unique()[-1]])
    ybins_unique = np.array(list(ybin_low.unique())+[ybin_upp.unique()[-1]])

    h = ax.hist2d(
        xbins, ybins,
        bins = [xbins_unique, ybins_unique],
        weights = df["yield"],
        cmap = "Blues",
    )
    cbar = fig.colorbar(h[3], ax=ax)
    cbar.set_label('Events', fontsize='large')

    ax.set_xlabel(cfg.xlabel, fontsize='large')
    ax.set_ylabel(cfg.ylabel, fontsize='large')

    # Add CMS text to top + energy + lumi
    ax.text(0, 1, r'$\mathbf{CMS}\ \mathit{Preliminary}$', ha='left',
            va='bottom', transform=ax.transAxes, fontsize='large')
    ax.text(1, 1, r'$35.9\ \mathrm{fb}^{-1}(13\ \mathrm{TeV})$', ha='right',
            va='bottom', transform=ax.transAxes, fontsize='large')

    print("Creating {}.pdf".format(path))
    #plt.tight_layout()
    fig.savefig(path+".pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

    return df
