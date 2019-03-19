import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import Readers
import Collectors
from event_selection import event_selection
from alphatwirl.loop import NullCollector

import os
datapath = os.path.join(os.environ["TOPDIR"], "data")

collection_creator = Readers.CollectionCreator(
    name = "collection_creator",
    collections = ["GenPart", "GenDressedLepton", "LHEPart"],
    variations = [],
)

gen_boson_producer = Readers.GenBosonProducer(
    name = "gen_boson_producer",
    data = False,
)
lhe_part_assigner = Readers.LHEPartAssigner(
    name = "lhe_part_assigner",
    data = False,
)
gen_part_assigner = Readers.GenPartAssigner(
    name = "gen_part_assigner",
    data = False,
)

weight_creator = Readers.WeightCreator(
    name = "weight_creator",
)
weight_xsection_lumi = Readers.WeightXsLumi(
    name = "weight_xsection_lumi",
    data = False,
)

weight_qcd_ewk = Readers.WeightQcdEwk(
    name = "weight_qcd_ewk",
    input_paths = {
        "ZJetsToNuNu":   (datapath+"/qcd_ewk/vvj.dat", "vvj_pTV_{}"),
        "WJetsToLNu":    (datapath+"/qcd_ewk/evj.dat", "evj_pTV_{}"),
        "DYJetsToLL":    (datapath+"/qcd_ewk/eej.dat", "eej_pTV_{}"),
        "ZJetsToLL":     (datapath+"/qcd_ewk/eej.dat", "eej_pTV_{}"),
        "GStarJetsToLL": (datapath+"/qcd_ewk/eej.dat", "eej_pTV_{}"),
    },
    underflow = True,
    overflow = True,
    formula = "((K_NNLO + d1k_qcd*d1K_NNLO + d2k_qcd*d2K_NNLO + d3k_qcd*d3K_NNLO)"\
              " /(K_NLO + d1k_qcd*d1K_NLO + d2k_qcd*d2K_NLO + d3k_qcd*d3K_NLO))"\
              "*(1 + kappa_EW + d1k_ew*d1kappa_EW + isz*(d2k_ew_z*d2kappa_EW + d3k_ew_z*d3kappa_EW)"\
                                                 "+ isw*(d2k_ew_w*d2kappa_EW + d3k_ew_w*d3kappa_EW))"\
              "+ dk_mix*dK_NLO_mix",
    params = ["K_NLO", "d1K_NLO", "d2K_NLO", "d3K_NLO", "K_NNLO", "d1K_NNLO",
              "d2K_NNLO", "d3K_NNLO", "kappa_EW", "d1kappa_EW", "d2kappa_EW",
              "d3kappa_EW", "dK_NLO_mix"],
    nuisances = ["d1k_qcd", "d2k_qcd", "d3k_qcd", "d1k_ew", "d2k_ew_z",
                 "d2k_ew_w", "d3k_ew_z", "d3k_ew_w", "dk_mix"],
)

selection_producer = Readers.SelectionProducer(
    name = "selection_producer",
    event_selection = event_selection,
)

gstar_correction_reader = Collectors.GStarCorrectionReader(
    name = "gstar_correction_reader",
    cfg = Collectors.GStarCorrection_cfg,
)
gstar_correction_collector = Collectors.GStarCorrectionCollector(
    name = "gstar_correction_collector",
    plot = True,
    cfg = Collectors.GStarCorrection_cfg,
)

sequence = [
    # Readers
    (collection_creator, NullCollector()),
    (gen_boson_producer, NullCollector()),
    #(lhe_part_assigner, NullCollector()),
    #(gen_part_assigner, NullCollector()),
    (weight_creator, NullCollector()),
    (weight_xsection_lumi, NullCollector()),
    (selection_producer, NullCollector()),
    # Collectors
    (gstar_correction_reader, gstar_correction_collector),
]
