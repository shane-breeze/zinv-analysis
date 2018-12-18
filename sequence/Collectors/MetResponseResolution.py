#import ROOT
#ROOT.gROOT.SetBatch(True)
#ROOT.gErrorIgnoreLevel = ROOT.kWarning

import os
import operator
import copy
import re
import numpy as np
import pandas as pd
import cPickle as pickle

from scipy.special import wofz
from uncertainties import ufloat

from drawing.dist_ratio import dist_ratio
from drawing.dist_scatter_pull import dist_scatter_pull

# Take the cfg module and drop unpicklables
from Histogrammer import HistReader, HistCollector

latex_eq_regex = re.compile("\$(.*)\$")

class MetResponseResolutionReader(HistReader):
    def __init__(self, **kwargs):
        super(MetResponseResolutionReader, self).__init__(**kwargs)
        self.cfg.log = False

class MetResponseResolutionCollector(HistCollector):
    def __init__(self, **kwargs):
        super(MetResponseResolutionCollector, self).__init__(**kwargs)
        #self.variations = [v for v in event.variations if v != ""]
        self.cfg.log = False

    def draw(self, histograms):
        args = []
#        datasets = ["MET", "SingleMuon", "SingleElectron"]
#
#        df = histograms.histograms
#        if df.empty:
#            return []
#
#        binning = histograms.binning
#        all_columns = list(df.index.names)
#        columns_noproc = [c for c in all_columns if c != "process"]
#        columns_nobin0 = [c for c in all_columns if "bin0" not in c]
#        columns_noproc_nobin0 = [c for c in columns_noproc if "bin0" not in c]
#        columns_noproc_nobin0_novar_nobin1 = [c for c in columns_noproc_nobin0 if "bin1" not in c]
#
#        # Add index for variation
#        all_columns.insert(all_columns.index("region")+1, "variation")
#        columns_noproc.insert(columns_noproc.index("region")+1, "variation")
#        columns_nobin0.insert(columns_nobin0.index("region")+1, "variation")
#        columns_noproc_nobin0.insert(columns_noproc_nobin0.index("region")+1, "variation")
#
#        df["region"] = df.index.get_level_values("region")
#        df["variation"] = df["region"].apply(lambda x: next((v for v in self.variations if v in x), "nominal"))
#        df = df.set_index("variation", append=True)
#
#        for variation in self.variations:
#            df["region"] = [c.replace(variation, "") for c in df.index.get_level_values("region")]
#            df["name"] = [c.replace(variation, "") for c in df.index.get_level_values("name")]
#            df["variable0"] = [c.replace(variation, "") for c in df.index.get_level_values("variable0")]
#            df["variable1"] = [c.replace(variation, "") for c in df.index.get_level_values("variable1")]
#            df = df.reset_index("region", drop=True)\
#                    .set_index("region", append=True)
#            df = df.reset_index("name", drop=True)\
#                    .set_index("name", append=True)
#            df = df.reset_index("variable0", drop=True)\
#                    .set_index("variable0", append=True)
#            df = df.reset_index("variable1", drop=True)\
#                    .set_index("variable1", append=True)
#        df = df.reorder_levels(all_columns)
#
#        # Create mc sum
#        df_data = df[df.index.get_level_values("process").isin(datasets)]
#        df_data = df[df.index.get_level_values("dataset")\
#                     == df.index.get_level_values("process")]
#        df_mcsum = df[~df.index.get_level_values("process").isin(datasets)]\
#                .groupby(columns_noproc)\
#                .sum()
#        df_mcsum["process"] = "MCSum"
#        df_mcsum = df_mcsum.set_index("process", append=True).reorder_levels(all_columns)
#        df = pd.concat([df, df_mcsum])
#        df_forfit = df[df.index.get_level_values("process").isin(datasets+["MCSum"])]
#
#        # Perform fitting
#        #df_fit = df_forfit.groupby(columns_nobin0).apply(fit)
#        #df = pd.merge(df.reset_index(["bin0_low", "bin0_upp"]), df_fit,
#        #              on=columns_nobin0, how='left', left_index=True)
#        #df["formula"] = voigt((df["bin0_low"]+df["bin0_upp"])/2,
#        #                      df["mean"], df["sigma"], df["gamma"])
#        df = df.set_index(["bin0_low", "bin0_upp"], append=True)
#
#        # Draw 1D plots
#        args = []
#        for cat, df_group in df.groupby(columns_noproc_nobin0):
#            # Create output directory structure
#            path = os.path.join(self.outdir, "plots", *cat[:2])
#            if not os.path.exists(path):
#                os.makedirs(path)
#
#            filename = "{}__{}To{}__{}".format(
#                cat[4],
#                int(cat[7]) if not np.isinf(cat[7]) else cat[7],
#                int(cat[8]) if not np.isinf(cat[8]) else cat[8],
#                cat[2],
#            )
#            filepath = os.path.abspath(os.path.join(path, filename))
#
#            # Add cfg text
#            cfg = copy.deepcopy(self.cfg)
#            cfg.text = []
#            params = df_group[(df_group.index.get_level_values("process").isin(datasets)\
#                               & (df_group.index.get_level_values("process")\
#                                  == df_group.index.get_level_values("dataset")))\
#                              | df_group.index.get_level_values("process").isin(["MCSum"])]\
#                    .groupby(columns_nobin0)\
#                    .apply(lambda x: x.iloc[1])[["mean", "mean_unc", "width", "width_unc"]]
#            params = params.reset_index("process")
#
#            try:
#                params_data = params[params.process.isin(datasets)].iloc[0]
#            except IndexError:
#                params_data = {"mean": 0., "mean_unc": 0., "width": 0., "width_unc": 0.}
#
#            try:
#                params_mc = params[params.process.isin(["MCSum"])].iloc[0]
#            except IndexError:
#                params_mc = {"mean": 0., "mean_unc": 0., "width": 0., "width_unc": 0.}
#
#            def get_sfs(val):
#                n = 0
#                test_val = abs(copy.deepcopy(val))
#                while 0. < test_val < 1.:
#                    test_val *= 10.
#                    n += 1
#                return n
#
#            if "up" not in cat[2].lower() and "down" not in cat[2].lower():
#                mean = params_data["mean"]
#                mean_unc = params_data["mean_unc"]
#                width = params_data["width"]
#                width_unc = params_data["width_unc"]
#
#                mean_fmt = "{:."+str(get_sfs(mean_unc))+"f}"
#                width_fmt = "{:."+str(get_sfs(width_unc))+"f}"
#                cfg.text.append("Data:")
#                cfg.text.append(r'$\mu = ' + mean_fmt.format(mean)\
#                                + r' \pm ' + mean_fmt.format(mean_unc)+'$')
#                cfg.text.append(r'$\sigma = ' + width_fmt.format(width)\
#                                + r' \pm ' + width_fmt.format(width_unc)+'$')
#
#            mean = params_mc["mean"]
#            mean_unc = params_mc["mean_unc"]
#            width = params_mc["width"]
#            width_unc = params_mc["width_unc"]
#
#            mean_fmt = "{:."+str(get_sfs(mean_unc))+"f}"
#            width_fmt = "{:."+str(get_sfs(width_unc))+"f}"
#            cfg.text.append("MC:")
#            cfg.text.append(r'$\mu = ' + mean_fmt.format(mean)\
#                            + r' \pm ' + mean_fmt.format(mean_unc) + r'$')
#            cfg.text.append(r'$\sigma = ' + width_fmt.format(width)\
#                            + r' \pm ' + width_fmt.format(width_unc) + r'$')
#
#            ylabel = self.cfg.axis_label.get(cat[4].split("__")[-1], cat[4].split("__")[-1])
#            match = latex_eq_regex.search(ylabel)
#            if match: ylabel = match.group(0).replace("$","")
#            cfg.text.append(r'${} < {} < {}$'.format(
#                int(cat[7]) if not np.isinf(cat[7]) else cat[7],
#                ylabel,
#                int(cat[8]) if not np.isinf(cat[8]) else cat[8],
#            ))
#
#            # Create args list for post-process drawing
#            bins = binning[cat[4]][0] # 2D -> 1D profile
#            with open(filepath+".pkl", 'w') as f:
#                pickle.dump((df_group, bins, filepath, cfg), f)
#            #args.append((dist_ratio, (df_group, bins, filepath, cfg)))

        #df_fit = df_fit[~(
        #    df_fit.index.get_level_values("process").isin(datasets)\
        #    & (df_fit.index.get_level_values("process")\
        #       != df_fit.index.get_level_values("dataset")))]

        #for cat, df_group in df_fit.groupby(columns_noproc_nobin0_novar_nobin1):
        #    df_grp = df_group[["mean", "mean_unc", "width", "width_unc"]]
        #    df_nominal = df_grp[df_grp.index.get_level_values("variation").isin(["nominal"])]\
        #            .reset_index("variation", drop=True)
        #    for variation in self.variations:
        #        df_var = df_grp[df_grp.index.get_level_values("variation").isin([variation])]\
        #                .reset_index("variation", drop=True)
        #        df_var = np.abs(df_var - df_nominal)
        #        for attr in ["mean", "width"]:
        #            df_nominal["{}_unc_{}".format(attr, variation)] = df_var[attr]

        #    path = os.path.join(self.outdir, "plots", *cat[:2])
        #    if not os.path.exists(path):
        #        os.makedirs(path)

        #    for label, attr in [("\mu", "mean"), ("\sigma", "width")]:
        #        filename = "{}__{}".format(cat[3], attr)
        #        filepath = os.path.abspath(os.path.join(path, filename))
        #        df_toplot = df_nominal[[c for c in df_nominal.columns if attr in c]]
        #        cfg = copy.deepcopy(self.cfg)

        #        xlabel = self.cfg.axis_label.get(cat[3].split("__")[1], cat[3].split("__")[1])
        #        ylabel = self.cfg.axis_label.get(cat[3].split("__")[0], cat[3].split("__")[1])
        #        match = latex_eq_regex.search(ylabel)
        #        if match: ylabel = r'${}({})$'.format(label, match.group(0).replace("$", ""))
        #        cfg.xlabel = xlabel
        #        cfg.ylabel = ylabel

        #        #dist_scatter_pull(df_toplot, self.variations, filepath, cfg)
        #        bins = binning[cat[3]][0] # 2D -> 1D profile
        #        with open(filepath+".pkl", 'w') as f:
        #            pickle.dump((df_toplot, bins, self.variations, filepath, cfg), f)
        #        args.append((dist_scatter_pull, (df_toplot, bins, self.variations, filepath, cfg)))

        return args

def fit(df):
    return None
#    bins = df.index.get_level_values("bin0_low").values[1:]
#    yields = df[["yield"]].values[1:-1].ravel()
#    vars = df[["variance"]].values[1:-1].ravel()
#
#    if yields.sum() == 0.:
#        return pd.DataFrame({
#            "mean": [0.],
#            "mean_unc": [0.],
#            "sigma": [0.],
#            "sigma_unc": [0.],
#            "gamma": [0.],
#            "gamma_unc": [0.],
#            "width": [0.],
#            "width_unc": [0.],
#            "chi2": [0.],
#            "ndof": [0.],
#        })
#
#    hdata = ROOT.TH1D("data", "", len(bins)-1, bins)
#
#    errs = np.array_equal(yields, vars)
#    if errs: hdata.Sumw2()
#
#    for ibin in range(1, hdata.GetNbinsX()+1):
#        hdata.SetBinContent(ibin, yields[ibin-1])
#        hdata.SetBinError(ibin, np.sqrt(vars[ibin-1]))
#
#    x = ROOT.RooRealVar("x", "x", bins[0], bins[-1])
#    #xframe = x.frame()
#    l = ROOT.RooArgList(x)
#    data = ROOT.RooDataHist("data", "data", l, hdata)
#
#    mean_guess = hdata.GetMean()
#    width_guess = hdata.GetStdDev()
#
#    mu_eq = "mu[{},{},{}]".format(mean_guess, bins[0], bins[-1])
#    gam_eq = "gam[{},{},{}]".format(width_guess, width_guess/20., width_guess*20.)
#    sig_eq = "sig[{},{},{}]".format(width_guess, width_guess/20., width_guess*20.)
#    voig_eq = "Voigtian:voig(x, {}, {}, {})".format(mu_eq, gam_eq, sig_eq)
#
#    ws = ROOT.RooWorkspace("ws")
#    getattr(ws, "import")(x)
#    ws.factory(voig_eq)
#    model = ws.pdf("voig")
#
#    args = [ROOT.RooFit.Minimizer("Minuit2", "Migrad"),
#            ROOT.RooFit.Offset(True),
#            ROOT.RooFit.PrintLevel(-1),
#            ROOT.RooFit.Save()]
#    if errs:
#        args.append(ROOT.RooFit.SumW2Error(True))
#
#    fit_result = model.fitTo(data, *args)
#    #data.plotOn(xframe)
#    #model.plotOn(xframe)
#    #chi2 = xframe.chiSquare(3)
#    chi2 = 0.
#
#    mu = ws.var("mu")
#    mean = ufloat(mu.getValV(), mu.getError())
#
#    sig = ws.var("sig")
#    sigma = ufloat(sig.getValV(), sig.getError())
#
#    gam = ws.var("gam")
#    gamma = ufloat(gam.getValV(), gam.getError())
#
#    # https://en.wikipedia.org/wiki/Voigt_profile
#    # FWHM / (2*sqrt(2*ln(2)))
#    width_breit = 2.*gamma
#    width_gauss = 2.*sigma*(2.*np.log(2.)**0.5)
#    if width_gauss != 0.:
#        phi = width_breit / width_gauss
#        c0 = 2.0056
#        c1 = 1.0593
#        width = sigma * (1. - c0*c1 + (phi**2 + 2.*c1*phi + c0**2*c1**2)**0.5)
#    else:
#        width = width_breit
#
#    return pd.DataFrame({
#        "mean": [mean.n],
#        "mean_unc": [mean.s],
#        "sigma": [sigma.n],
#        "sigma_unc": [sigma.s],
#        "gamma": [gamma.n],
#        "gamma_unc": [gamma.s],
#        "width": [width.n],
#        "width_unc": [width.s],
#        "chi2": [chi2],
#        "ndof": [len(bins)-3],
#    })

def voigt(x, mean, sigma, gamma):
    z = ((x-mean) + 1j*gamma*0.5) / (sigma*np.sqrt(2))
    return np.real(wofz(z)) / (sigma*np.sqrt(2*np.pi))
