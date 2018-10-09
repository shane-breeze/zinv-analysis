import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def dist_multicomp(df, filepath, cfg):
    # Define columns
    all_keys = list(df.index.get_level_values("key").unique())
    all_columns = list(df.index.names)
    columns_nokey = [c for c in all_columns if c != "key"]
    columns_nobins = [c for c in all_columns if "bin" not in c]

    # Remove under and overflow bins (add overflow into final bin)
    def truncate(indf):
        indf.iloc[-2] += indf.iloc[-1]
        indf = indf.iloc[1:-1]
        indf = indf.reset_index(columns_nobins, drop=True)
        return indf
    df = df.groupby(columns_nobins).apply(truncate)

    df_pivot_key = df.pivot_table(
        values = 'yield',
        index = columns_nokey,
        columns = 'key',
        aggfunc = np.sum,
    )
    df_pivot_ratio = df_pivot_key[[k for k in all_keys if k != ""]]
    try:
        df_pivot_ratio = df_pivot_ratio.div(df_pivot_key[""], axis=0)
    except KeyError:
        pass

    # Get the global bins
    bins_low = list(df_pivot_key.index.get_level_values("bin0_low"))
    bins_upp = list(df_pivot_key.index.get_level_values("bin0_upp"))
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
    axtop.set_xlim(bins[0], bins[-1])

    # Draw hists
    axtop.hist(
        [bin_centers]*len(all_keys),
        bins = bins,
        weights = [df_pivot_key[k] for k in all_keys],
        histtype = 'step',
        color = [cfg.sample_colours.get(k, "blue") for k in all_keys],
        label = [cfg.sample_names.get(k, k) for k in all_keys],
    )

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

    # Legend
    handles, labels = axtop.get_legend_handles_labels()

    new_handles, new_labels = [], []
    for hand, lab in zip(handles[::-1], labels[::-1]):
        if lab not in new_labels:
            new_handles.append(hand)
            new_labels.append(lab)
            axtop.legend(new_handles, new_labels)

    # Ratio in bottom panel
    all_var_keys = all_keys[1:]
    df_pivot_ratio = df_pivot_ratio[all_var_keys]

    next_bin_centers = [bin_centers]*len(all_var_keys)
    next_weights = [df_pivot_ratio[k] for k in all_var_keys]
    next_bin_centers, next_weights = zip(*[
        (b[~np.isnan(w)], w[~np.isnan(w)])
        for b, w in zip(next_bin_centers, next_weights)
    ])

    axbot.hist(
        next_bin_centers,
        bins = bins,
        weights = next_weights,
        histtype = 'step',
        color = [cfg.sample_colours.get(k, 'blue') for k in all_var_keys],
    )
    axbot.axhline(1, ls='--', color='gray')

    name = cfg.name
    ymin = min(df_pivot_ratio.min())
    ymax = max(df_pivot_ratio.max())
    padding = (ymax - ymin)*0.05
    if not (np.isinf(ymin) or np.isnan(ymin) or np.isinf(ymax) or np.isnan(ymax)):
        axbot.set_ylim((ymin-padding, ymax+padding))
    axbot.set_xlabel(cfg.axis_label.get(name, name), fontsize='large')
    axbot.set_ylabel(r'Relative variation', fontsize='large')

    # Report
    print("Creating {}.pdf".format(filepath))

    # Actually save the figure
    plt.tight_layout()
    fig.savefig(filepath+".pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

    return df
