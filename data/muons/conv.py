import numpy as np
import pandas as pd
import uproot

def main():
    for path, outname in [
        ("EfficienciesAndSF_BCDEF.root:LooseISO_LooseID_pt_eta/abseta_pt_ratio", "muon_iso_loose_looseID_runBCDEF.txt"),
        ("EfficienciesAndSF_BCDEF.root:LooseISO_MediumID_pt_eta/abseta_pt_ratio", "muon_iso_loose_mediumID_runBCDEF.txt"),
        ("EfficienciesAndSF_BCDEF.root:LooseISO_TightID_pt_eta/abseta_pt_ratio", "muon_iso_loose_tightID_runBCDEF.txt"),
        ("EfficienciesAndSF_BCDEF.root:TightISO_MediumID_pt_eta/abseta_pt_ratio", "muon_iso_tight_mediumID_runBCDEF.txt"),
        ("EfficienciesAndSF_BCDEF.root:TightISO_TightID_pt_eta/abseta_pt_ratio", "muon_iso_tight_tightID_runBCDEF.txt"),
        ("EfficienciesAndSF_GH.root:LooseISO_LooseID_pt_eta/abseta_pt_ratio", "muon_iso_loose_looseID_runGH.txt"),
        ("EfficienciesAndSF_GH.root:LooseISO_MediumID_pt_eta/abseta_pt_ratio", "muon_iso_loose_mediumID_runGH.txt"),
        ("EfficienciesAndSF_GH.root:LooseISO_TightID_pt_eta/abseta_pt_ratio", "muon_iso_loose_tightID_runGH.txt"),
        ("EfficienciesAndSF_GH.root:TightISO_MediumID_pt_eta/abseta_pt_ratio", "muon_iso_tight_mediumID_runGH.txt"),
        ("EfficienciesAndSF_GH.root:TightISO_TightID_pt_eta/abseta_pt_ratio", "muon_iso_tight_tightID_runGH.txt"),
    ]:
        filepath, histpath = path.split(":")
        with uproot.open(filepath) as f:
            hist = f[histpath]

        xbins, ybins = hist.bins
        xlow, xhigh = xbins[:,0], xbins[:,1]
        ylow, yhigh = ybins[:,0], ybins[:,1]
        xlow, ylow = np.meshgrid(xlow, ylow)
        xhigh, yhigh = np.meshgrid(xhigh, yhigh)
        values = hist.values
        errors = np.sqrt(hist.variances)

        df = pd.DataFrame({
            "eta_low": xlow.ravel(),
            "eta_high": xhigh.ravel(),
            "pt_low": ylow.ravel(),
            "pt_high": yhigh.ravel(),
            "correction": values.ravel(),
            "unc_up": errors.ravel(),
            "unc_down": errors.ravel(),
        }, columns=["eta_low","eta_high","pt_low","pt_high","correction","unc_up","unc_down"])

        with open(outname, 'w') as f:
            f.write(df.to_string(
                formatters = {
                    "eta_low": lambda x: "{:.3f}".format(x),
                    "eta_high": lambda x: "{:.3f}".format(x),
                    "pt_low": lambda x: "{:.1f}".format(x),
                    "pt_high": lambda x: "{:.1f}".format(x),
                    "correction": lambda x: "{:.10e}".format(x),
                    "unc_up": lambda x: "{:.10e}".format(x),
                    "unc_down": lambda x: "{:.10e}".format(x),
                },
                index = False,
            ))

if __name__ == "__main__":
    main()
