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

    # Draw hists
    processes = [p for p in df_pivot_corr.columns if p in allowed_processes]
    axtop.hist(
        [df_pivot_uncorr[p] for p in processes],
        bins = bins,
        log = cfg.log,
        histtype = 'step',
        color = [cfg.sample_colours.get(p, "blue") for p in processes],
        label = [cfg.sample_names.get(p, p) for p in processes],
        ls = '--',
    )
    handles, labels = axtop.get_legend_handles_labels()
    axtop.hist(
        [df_pivot_corr[p] for p in processes],
        bins = bins,
        log = cfg.log,
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
        [df_pivot_ratio[p] for p in processes],
        bins = bins,
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

def taper_and_drop(hist):
    """
    Final bin is an overflow.
    Remove underflow.
    """
    hist["counts"][-2] += hist["counts"][-1]
    hist["yields"][-2] += hist["yields"][-1]
    hist["variance"][-2] += hist["variance"][-1]

    if isinstance(hist["bins"], list):
        hist["bins"] = hist["bins"][0]
    hist["bins"] = hist["bins"][1:-1]
    hist["counts"] = hist["counts"][1:-1]
    hist["yields"] = hist["yields"][1:-1]
    hist["variance"] = hist["variance"][1:-1]

    return hist

def dist_comp_old(hist_pairs, filepath, cfg):
    """
    Draw distributions with a ratio plot beneath.

    Parameters
    ----------

    hist_pairs : List of pairs of dicts (ratio of the pairs shown in subplot).
                 Dicts have the following form:
    {
        "name" : name of the distribution being plotted
        "sample" : name of the sample (used for labelling)
        "bins" : numpy array of the bins edges used (size = (nbins+1,)). The
                 first bin is taken as the underflow and the last bin the
                 overflow
        "counts" : numpy array (size = (nbins,)) of the counts per bin (not
                   used for plotting yet)
        "yields" : numpy array (size = (nbins,)) of the yields per bin (what is
                  plotted)
        "variance" : numpy array (size = (nbins,)) of the variance per bin.
                     Sqrt of this is used for the error on the data. Symmetric
                     errors only so far.
        "function" : optional. numpy array (size = (nbins,)) for the y values
                     in each bin for a normalized (to 1) function to be
                     plotted (which will be smoothed)
    }
    filepath : str for the output file destination (without the extension)
    cfg : object with the following attributes:
        log : boolean. If True then the y-axis will be on a log-scale
        sample_colours : dict. Conversion between sample names and colours
        sample_names : dict. Conversion between sample names and their labels
                       shown in the plot.
        axis_label : dict. Conversion between axis names and their labels shown
                     in the plot

    Returns
    -------
    "Success"
    """

    # Split axis into top and bottom with ratio 3:1
    # Share the x axis, not the y axis
    # Figure size is 4.8 by 6.4 inches
    fig, (axtop, axbot) = plt.subplots(
        nrows=2, ncols=1, sharex='col', sharey=False,
        gridspec_kw={'height_ratios': [3, 1]},
        figsize = (4.8, 6.4),
    )

    # Remove under/overflow bins (add overflow into final bin)
    hist_pairs = [map(taper_and_drop, hs) for hs in hist_pairs]
    bins = hist_pairs[0][0]["bins"]

    # Draw hists
    axtop.hist(
        [hs[0]["yields"] for hs in hist_pairs],
        bins = bins,
        log = cfg.log,
        histtype = 'step',
        color = [cfg.sample_colours.get(hs[0]["sample"], "blue")
                 for hs in hist_pairs],
        label = [cfg.sample_names.get(hs[0]["sample"], hs[0]["sample"])
                 for hs in hist_pairs],
        ls = '--',
    )
    axtop.hist(
        [hs[1]["yields"] for hs in hist_pairs],
        bins = bins,
        log = cfg.log,
        histtype = 'step',
        color = [cfg.sample_colours.get(hs[1]["sample"], "blue")
                 for hs in hist_pairs],
        label = [cfg.sample_names.get(hs[1]["sample"], hs[1]["sample"])+" (nNLO EW)"
                 for hs in hist_pairs],
    )
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
    handles, labels = axtop.get_legend_handles_labels()
    handles = handles[::-1]
    labels = labels[::-1]

    # Additional text added to the legend title
    if not hasattr(cfg, "text"):
        axtop.legend(handles, labels, labelspacing=0.1)
    else:
        axtop.legend(handles, labels, title=cfg.text, labelspacing=0.1)

    ratios = [{
        "name": hs[0]["name"],
        "sample": hs[0]["sample"],
        "bins": bins,
        "counts": hs[1]["counts"] + hs[0]["counts"],
        "yields": hs[1]["yields"] / hs[0]["yields"],
        "variance": (hs[1]["yields"]/hs[0]["yields"])**2*(hs[0]["variance"]/hs[0]["yields"]**2 + hs[1]["variance"]/hs[1]["yields"]**2),
    } for hs in hist_pairs]

    axbot.hist(
        [r["yields"] for r in ratios],
        bins = bins,
        histtype = 'step',
        color = [cfg.sample_colours.get(r["sample"], "blue") for r in ratios],
    )

    # x and y limits for the ratio plot
    axbot.set_xlim(bins[0], bins[-1])
    axbot.set_ylim(0.5, 1.5)

    # x and y title labels for the ratio (axtop shares x-axis)
    name = ratios[0]["name"]
    axbot.set_xlabel(cfg.axis_label[name] if name in cfg.axis_label else name,
                     fontsize='large')
    axbot.set_ylabel(r'$1+\kappa_{nNLO EW}$', fontsize='large')

    # Dashed line at 1. in the ratio plot
    axbot.plot([bins[0], bins[-1]], [1., 1.], color='black', linestyle=':')

    # Report
    print("Creating {}".format(filepath))

    # Actually save the figure
    plt.tight_layout()
    fig.savefig(filepath+".pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

    return "Success"
