import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import spline

def dist_ratio(df, filepath, cfg):
    # Define columns
    datasets = ["MET", "SingleMuon", "SingleElectron"]
    all_columns = list(df.index.names)
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

    # Sum (count, yield, variance) over all processes not data
    if not df.index.get_level_values('process').isin(["MCSum"]).any():
        df_mcsum = df[~df.index.get_level_values('process').isin(datasets+["MCSum"])]\
                .groupby(columns_noproc)\
                .sum()
        df_mcsum["process"] = "MCSum"
        df_mcsum = df_mcsum.set_index("process", append=True)\
                .reorder_levels(all_columns)
        df = pd.concat([df, df_mcsum])

    # Split dataframe into components
    df_data = df[df.index.get_level_values('process').isin(datasets)]
    df_data = df_data[df_data.index.get_level_values('process') \
                      == df_data.index.get_level_values('dataset')]
    df_mcsum = df[df.index.get_level_values('process').isin(["MCSum"])]
    df_mcall = df[~df.index.get_level_values('process').isin(datasets+["MCSum"])]

    # Ratio
    df_ratio = df_data["yield"].reset_index(level="process", drop=True)\
            / df_mcsum["yield"].reset_index(level="process", drop=True)
    df_ratio_data_var = df_data["variance"].reset_index(level="process", drop=True)\
            / (df_mcsum["yield"].reset_index(level="process", drop=True)**2)
    df_ratio_mc_var = df_mcsum["variance"].reset_index(level="process", drop=True)\
            / (df_mcsum["yield"].reset_index(level="process", drop=True)**2)

    # Combine df_mcall processes <1% together
    df_mcall_int_fraction = df_mcall\
            .groupby(columns_nobins)\
            .sum()\
            .groupby(columns_nobins_noproc)\
            .apply(lambda x: x/x.sum())
    df_mcall["to_merge"] = ~(
        (df_mcall_int_fraction["yield"]>=0.01) \
        | (df_mcall_int_fraction.index.get_level_values("process").isin(["QCD"]))
    )
    df_mcminor = df_mcall[df_mcall["to_merge"]]\
            .groupby(columns_noproc)\
            .sum()\
            .drop("to_merge", axis=1)
    df_mcminor["process"] = "Minor"
    df_mcminor = df_mcminor\
            .set_index("process", append=True)\
            .reorder_levels(all_columns)

    df_mcall = df_mcall[~df_mcall["to_merge"]]\
            .drop("to_merge", axis=1)
    if df_mcminor.sum()["yield"] > 0.:
        df_mcall = pd.concat([df_mcall, df_mcminor])

    # Split axis into top and bottom with ratio 3:1
    # Share the x axis, not the y axis
    # Figure size is 4.8 by 6.4 inches
    fig, (axtop, axbot) = plt.subplots(
        nrows=2, ncols=1, sharex='col', sharey=False,
        gridspec_kw={'height_ratios': [3, 1]},
        figsize = (4.8, 6.4),
    )

    # Get the global bins
    bins_low = list(df_mcsum.index.get_level_values("bin0_low"))
    bins_upp = list(df_mcsum.index.get_level_values("bin0_upp"))
    bins = np.array(bins_low[:]+[bins_upp[-1]])
    bin_centers = (bins[1:]+bins[:-1])/2
    bin_widths = (bins[1:]-bins[:-1])

    df_mcall_pivot_proc = df_mcall.pivot_table(values='yield',
                                               index=columns_noproc,
                                               columns='process',
                                               aggfunc=np.sum)

    sorted_processes = list(df_mcall_pivot_proc.sum().sort_values().index)
    if "Minor" in sorted_processes:
        sorted_processes.remove("Minor")
        sorted_processes = ["Minor"]+sorted_processes
    df_mcall_pivot_proc = df_mcall_pivot_proc[sorted_processes]

    axtop.hist(
        [df_mcall_pivot_proc[process].values for process in sorted_processes],
        bins = bins,
        log = cfg.log,
        stacked = True,
        color = [cfg.sample_colours.get(proc, "blue")
                 for proc in sorted_processes],
        label = sorted_processes,
    )

    axtop.hist(
        df_mcsum["yield"],
        bins = bins,
        log = cfg.log,
        histtype = 'step',
        color = "black",
        label = "",
    )

    if "formula" in df_mcsum.columns:
        try:
            xs = (bins[1:] + bins[:-1])/2
            ys = df_mcsum["yield"].sum()*df_mcsum["formula"]/df_mcsum["formula"].sum()
            nans = np.isnan(ys)
            xnew = np.linspace(xs[~nans].min(), xs[~nans].max(), xs[~nans].shape[0]*4)
            ynew = spline(xs[~nans], ys[~nans], xnew)
            axtop.plot(xnew, ynew, color='r', ls='--', label="MC fit")
        except ValueError:
            pass

    has_data = df_data["yield"].sum()>0.
    if has_data:
        axtop.errorbar(
            (bins[1:] + bins[:-1])/2,
            df_data["yield"],
            yerr = np.sqrt(df_data["variance"]),
            fmt = 'o',
            markersize = 3,
            linewidth = 1,
            capsize = 1.8,
            color = "black",
            label = set(df_data.index.get_level_values("process")).pop(),
        )

        if "formula" in df_data.columns:
            try:
                xs = (bins[1:] + bins[:-1])/2
                ys = df_data["yield"].sum()*df_data["formula"]/df_data["formula"].sum()
                nans = np.isnan(ys)
                xnew = np.linspace(xs[~nans].min(), xs[~nans].max(), xs[~nans].shape[0]*4)
                ynew = spline(xs[~nans], ys[~nans], xnew)
                axtop.plot(xnew, ynew, color='k', ls='--', label="Data fit")
            except ValueError:
                pass

    axtop.set_xlim(bins[0], bins[-1])

    # Set ymin limit to maximum matplotlib's chosen minimum and 0.5
    ymin = max(axtop.get_ylim()[0], 0.5)
    axtop.set_ylim(ymin, None)
    axtop.set_xlabel("")
    axtop.set_ylabel("")

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

    df_int_fraction = df_mcall\
            .groupby(columns_nobins)\
            .sum()\
            .groupby(columns_nobins_noproc)\
            .apply(lambda x: x/x.sum())\
            .reset_index()[["process", "yield"]]
    if has_data:
        df_int_fraction = pd.concat([
            df_int_fraction,
            pd.DataFrame({
                "process": [set(df_data.index.get_level_values("process")).pop()],
                "yield": [df_data["yield"].sum()/df_mcsum["yield"].sum()],
            }),
        ])

    fit_labels = [label for label in labels if "fit" in label]
    labels = ["{} {:.2f}".format(
        cfg.sample_names.get(label, label),
        df_int_fraction[df_int_fraction["process"].isin([label])]["yield"].iloc[0],
    ) for label in labels if "fit" not in label] + fit_labels

    # Additional text added to the legend title
    title = []
    if hasattr(cfg, "text"):
        title.extend(cfg.text)
    axtop.legend(handles, labels, title="\n".join(title), labelspacing=0.1)

    # Data/MC in the lower panel
    axbot.errorbar(
        (bins[1:] + bins[:-1])/2,
        df_ratio,
        yerr = np.sqrt(df_ratio_data_var),
        fmt = 'o',
        markersize = 3,
        linewidth = 1,
        capsize = 1.8,
        color = 'black',
        label = "",
    )

    # MC stat uncertainty in the lower panel
    axbot.fill_between(
        bins,
        list(1.-np.sqrt(df_ratio_mc_var))+[1.],
        list(1.+np.sqrt(df_ratio_mc_var))+[1.],
        step = 'post',
        color = "#aaaaaa",
        label = "MC stat. unc.",
    )

    # Ratio legend
    handles, labels = axbot.get_legend_handles_labels()
    axbot.legend(handles, labels)

    # x and y limits for the ratio plot
    axbot.set_xlim(bins[0], bins[-1])
    axbot.set_ylim(0.5, 1.5)

    # x and y title labels for the ratio (axtop shares x-axis)
    name = set(df.index.get_level_values("name")).pop()
    name = name[0] if isinstance(name, list) else name
    name = name.split("__")[0]
    axbot.set_xlabel(cfg.axis_label.get(name, name),
                     fontsize='large')
    axbot.set_ylabel("Data / SM Total",
                     fontsize='large')

    # Dashed line at 1. in the ratio plot
    axbot.plot(
        [bins[0], bins[-1]],
        [1., 1.],
        color = 'black',
        linestyle = ':',
    )

    # Report
    print("Creating {}.pdf".format(filepath))

    # Actually save the figure
    plt.tight_layout()
    fig.savefig(filepath+".pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

    return df
