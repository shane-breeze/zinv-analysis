import argparse
import pandas as pd
from array import array

import ROOT
ROOT.gROOT.SetBatch(True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("shape_file", help="Shape file to add QCD systematics")
    parser.add_argument("syst_file", help="Input file with QCD systematics")
    return parser.parse_args()

def get_df(path):
    return pd.read_table(path, sep='\s+')

def convert_to_hist(x, vals, name):
    x = array('f', x)
    hist = ROOT.TH1D(name, name, len(x)-1, x)
    hist.SetDirectory(0)

    for i, v in enumerate(vals):
        hist.SetBinContent(i+1, v if v>=1e-7 else 1e-7)
    return hist

def main():
    options = parse_args()

    df = get_df(options.syst_file)
    x = df["x"].values

    rfile = ROOT.TFile.Open(options.shape_file, "UPDATE")
    keys = [
        k.GetName()
        for k in rfile.GetListOfKeys()
        if "qcd" in [
            subk.GetName()
            for subk in rfile.Get(k.GetName()).GetListOfKeys()
        ]
    ]

    for key in keys:
        rfile.cd(key)
        qcd_original = ROOT.gDirectory.Get("qcd").Clone("qcd_original_{}".format(key))

        vals = 1.+df["qcd_syst"].values
        vals[vals<=0.] = 0.
        hist = convert_to_hist(x, vals, "qcd")
        hist.Multiply(qcd_original)
        if "qcdSystUp" not in [k.GetName() for k in ROOT.gDirectory.GetListOfKeys()]:
            ROOT.gDirectory.mkdir("qcdSystUp")
        ROOT.gDirectory.cd("qcdSystUp")
        hist.Write("", ROOT.TObject.kOverwrite)
        print("Writing {}/{}/{}".format(key, "qcdSystUp", "qcd"))
        hist.Delete()
        rfile.cd(key)

        vals = 1./(1.+df["qcd_syst"].values)
        vals[vals<=0.] = 0.
        hist = convert_to_hist(x, vals, "qcd")
        hist.Multiply(qcd_original)
        if "qcdSystDown" not in [k.GetName() for k in ROOT.gDirectory.GetListOfKeys()]:
            ROOT.gDirectory.mkdir("qcdSystDown")
        ROOT.gDirectory.cd("qcdSystDown")
        hist.Write("", ROOT.TObject.kOverwrite)
        print("Writing {}/{}/{}".format(key, "qcdSystDown", "qcd"))
        hist.Delete()
        rfile.cd(key)

        for ibin in range(df.columns.shape[0]-2):
            name = "qcd_syst_bin{}".format(ibin)
            vals = 1.+df[name].values
            vals[vals<=0.] = 0.
            hist = convert_to_hist(x, vals, "qcd")
            hist.Multiply(qcd_original)
            if "qcdSystBin{}Up".format(ibin) not in [k.GetName() for k in ROOT.gDirectory.GetListOfKeys()]:
                ROOT.gDirectory.mkdir("qcdSystBin{}Up".format(ibin))
            ROOT.gDirectory.cd("qcdSystBin{}Up".format(ibin))
            hist.Write("", ROOT.TObject.kOverwrite)
            print("Writing {}/{}/{}".format(key, "qcdSystBin{}Up".format(ibin), "qcd"))
            hist.Delete()
            rfile.cd(key)

            vals = 1./(1.+df[name].values)
            vals[vals<=0.] = 0.
            hist = convert_to_hist(x, vals, "qcd")
            hist.Multiply(qcd_original)
            if "qcdSystBin{}Down".format(ibin) not in [k.GetName() for k in ROOT.gDirectory.GetListOfKeys()]:
                ROOT.gDirectory.mkdir("qcdSystBin{}Down".format(ibin))
            ROOT.gDirectory.cd("qcdSystBin{}Down".format(ibin))
            hist.Write("", ROOT.TObject.kOverwrite)
            print("Writing {}/{}/{}".format(key, "qcdSystBin{}Down".format(ibin), "qcd"))
            hist.Delete()
            rfile.cd(key)

        qcd_original.Delete()
    rfile.Close()
    rfile.Delete()

if __name__ == "__main__":
    main()
