import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(
    context = 'paper',
    style = 'ticks',
    font_scale = 1.8,
)

def dist_facet(df, filepath, cfg):
    variations = list(set(c.replace("Up","").replace("Down","") for c in df.columns))
    df_up = df.loc[:,[c for c in df.columns if "Up" in c]]
    df_down = df.loc[:,[c for c in df.columns if "Down" in c]]
    df_up.columns = [c.replace("Up","") for c in df_up.columns]
    df_up.columns.name = "key"
    df_down.columns = [c.replace("Down","") for c in df_down.columns]
    df_down.columns.name = "key"

    df_up = df_up.stack(level="key", dropna=False).reset_index()
    df_down = df_down.stack(level="key", dropna=False).reset_index()
    df_up["variation"] = "up"
    df_down["variation"] = "down"
    df = pd.concat([df_up, df_down])

    df = df.rename({
        "key": "Systematic",
        "process": "Process",
        "variation": "Variation",
        "bin0_low": cfg.xlabel,
        0: cfg.ylabel,
    }, axis='columns')

    g = sns.FacetGrid(
        df, row='Systematic', col='Process', hue='Variation',
        margin_titles=True, legend_out=True, col_order=cfg.process_order,
    )
    g.map(plt.step, cfg.xlabel, cfg.ylabel, where='post').add_legend()
    g.set(xlim=(0, 1000), ylim=(0.9, 1.1))

    g.fig.text(0.0, 1, r'$\mathbf{CMS}\ \mathit{Preliminary}$',
               ha='left', va='bottom', fontsize='large')
    g.fig.text(0.9, 1, r'$35.9\ \mathrm{fb}^{-1}(13\ \mathrm{TeV})$',
               ha='right', va='bottom', fontsize='large')

    # Report
    print("Creating {}.pdf".format(filepath))

    # Actually save the figure
    g.fig.savefig(filepath+".pdf", format="pdf", bbox_inches="tight")
    plt.close(g.fig)

    return df
