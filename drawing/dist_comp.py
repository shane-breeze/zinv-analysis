import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import spline

def dist_comp(df, filepath, cfg):
    allowed_processes = ["DYJetsToLL", "WJetsToLNu", "ZJetsToNuNu"]
    df = df.reset_index("weight", drop=True)

    # Define columns
    all_columns = list(df.index.names)
    columns_noname = [c for c in all_columns if c != "name"]
    columns_noproc = [c for c in all_columns if c != "process"]
    columns_nobins = [c for c in all_columns if "bin" not in c]
    columns_nobins_noproc = [c for c in columns_nobins if c != "process"]

    # Remove under and overflow bins (add overflow into final bin)
    def truncate(indf):
        indf.iloc[-2] += indf.iloc[-1]
        indf = indf.iloc[1:-1]
        indf = indf.reset_index(columns_nobins, drop=True)
        return indf
    df = df.groupby(columns_nobins).apply(truncate)

    df_pivot_name = df.pivot_table(
        values = 'yield',
        index = columns_noname,
        columns = 'name',
        aggfunc = np.sum,
    )

    name_uncorr = [c for c in df_pivot_name.columns if "corrected" not in c].pop()
    name_corr = [c for c in df_pivot_name.columns if "corrected" in c].pop()
    df_uncorr = df_pivot_name[name_uncorr]
    df_corr = df_pivot_name[name_corr]
    df_uncorr.columns = ['yield']
    df_corr.columns = ['yield']

    df_pivot_uncorr = df_uncorr.unstack(level='process')
    df_pivot_corr = df_corr.unstack(level='process')
    df_pivot_ratio = df_pivot_corr / df_pivot_uncorr

    # Get the global bins
    bins_low = list(df_pivot_uncorr.index.get_level_values("bin0_low"))
    bins_upp = list(df_pivot_uncorr.index.get_level_values("bin0_upp"))
    bins = np.array(bins_low[:]+[bins_upp[-1]])
    bin_centers = (bins[1:]+bins[:-1])/2
    bin_widths = (bins[1:]-bins[:-1])

    # Split axis into top and bottom with ratio 3:1
    # Share the x axis, not the y axis
    # Figure size is 4.8 by 6.4 inches
    fig, (axtop, axbot) = plt.subplots(
        nrows=2, ncols=1, sharex='col', sharey=False,
        gridspec_kw={'height_ratios': [3, 1]},
        figsize = (4.8, 6.4),
    )
    if cfg.log: axtop.set_yscale('log')

    # Draw hists
    processes = [p for p in df_pivot_corr.columns if p in allowed_processes]
    axtop.hist(
        [bin_centers]*len(processes),
        bins = bins,
        weights = [df_pivot_uncorr[p] for p in processes],
        histtype = 'step',
        color = [cfg.sample_colours.get(p, "blue") for p in processes],
        label = [cfg.sample_names.get(p, p) for p in processes],
        ls = '--',
    )
    handles, labels = axtop.get_legend_handles_labels()
    axtop.hist(
        [bin_centers]*len(processes),
        bins = bins,
        weights = [df_pivot_corr[p] for p in processes],
        histtype = 'step',
        color = [cfg.sample_colours.get(p, "blue") for p in processes],
        label = [cfg.sample_names.get(p, p) for p in processes],
    )
    handles_corr, labels_corr = axtop.get_legend_handles_labels()
    handles_corr = handles_corr[len(handles):]
    labels_corr = labels_corr[len(labels):]
    labels_corr = [l+" corr." for l in labels_corr]
    labels = reduce(lambda x,y: x+y, list(zip(labels, labels_corr)))
    handles = reduce(lambda x,y: x+y, list(zip(handles, handles_corr)))

    axtop.set_xlim(bins[0], bins[-1])



    # Add CMS text to top + energy + lumi
    axtop.text(0, 1, r'$\mathbf{CMS}\ \mathit{Preliminary}$',
               horizontalalignment='left',
               verticalalignment='bottom',
               transform=axtop.transAxes,
               fontsize='large')
    axtop.text(1, 1, r'$35.9\ \mathrm{fb}^{-1}(13\ \mathrm{TeV})$',
               horizontalalignment='right',
               verticalalignment='bottom',
               transform=axtop.transAxes,
               fontsize='large')

    # Legend - reverse the labels
    axtop.legend(handles, labels)

    # Ratio in bottom panel
    axbot.hist(
        [bin_centers]*len(processes),
        bins = bins,
        weights = [df_pivot_ratio[p] for p in processes],
        histtype = 'step',
        color = [cfg.sample_colours.get(p, 'blue') for p in processes],
    )
    axbot.axhline(1, ls='--', color='gray')

    name = cfg.name
    ymin = min(df_pivot_ratio[processes].min())
    ymax = max(df_pivot_ratio[processes].max())
    padding = (ymax - ymin)*0.05
    if not (np.isinf(ymin) or np.isnan(ymin) or np.isinf(ymax) or np.isnan(ymax)):
        axbot.set_ylim((ymin-padding, ymax+padding))
    axbot.set_xlabel(cfg.axis_label.get(name, name),
                     fontsize='large')
    axbot.set_ylabel(r'$1 + \kappa_{EW}$', fontsize='large')

    # Report
    print("Creating {}.pdf".format(filepath))

    # Actually save the figure
    plt.tight_layout()
    fig.savefig(filepath+".pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

    return df
