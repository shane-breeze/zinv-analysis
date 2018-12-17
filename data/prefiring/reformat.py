import ROOT
import numpy as np
import pandas as pd

path = "L1prefiring_photonpt_2016BtoH.root"

rfile = ROOT.TFile.Open(path)
hist = rfile.Get("L1prefiring_photonpt_2016BtoH")

data = []
for xidx in range(0, hist.GetNbinsX()+2):
    for yidx in range(0, hist.GetNbinsY()+2):
        xlow = hist.GetXaxis().GetBinLowEdge(xidx)
        xupp = hist.GetXaxis().GetBinUpEdge(xidx)
        ylow = hist.GetYaxis().GetBinLowEdge(yidx)
        yupp = hist.GetYaxis().GetBinUpEdge(yidx)
        content = hist.GetBinContent(xidx, yidx)
        error = hist.GetBinError(xidx, yidx)

        data.append({
            "xlow": xlow,
            "xupp": xupp,
            "ylow": ylow,
            "yupp": yupp,
            "content": content,
            "error": error,
        })
df = pd.DataFrame(data)
print(df.to_string())
