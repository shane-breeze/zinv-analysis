from alphatwirl.loop import NullCollector
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
toppath = os.path.abspath(os.path.join(os.environ["TOPDIR"], "zinv"))
datapath = os.path.join(toppath, "data")
collpath = os.path.join(toppath, "sequence", "Collectors")

import zinv.sequence.Readers as Readers
import zinv.sequence.Collectors as Collectors

event_tools = Readers.EventTools(
    name = "event_tools",
    maxsize = int(12*1024**3), # 6 GB
)

# Initialise readers and collectors
collection_creator = Readers.CollectionCreator(
    name = "collection_creator",
    collections = ["CaloMET", "MET", "Jet", "Electron", "Muon", "Photon", "Tau",
                   "GenMET", "GenPart", "GenJet", "GenDressedLepton", "LHEPart"],
)

# Gen/Lhe level producers
gen_boson_producer = Readers.GenBosonProducer(
    name = "gen_boson_producer",
    data = False,
)
lhe_part_assigner = Readers.LHEPartAssigner(
    name = "lhe_part_assigner",
    old_parents = ["WJetsToLNu", "DYJetsToLL", "ZJetsToLL", "GStarJetsToLL"],
    data = False,
)
gen_part_assigner = Readers.GenPartAssigner(
    name = "gen_part_assigner",
    old_parents = ["WJetsToLNu", "DYJetsToLL", "ZJetsToLL", "GStarJetsToLL"],
    data = False,
)

weight_xsection_lumi = Readers.WeightXsLumi(
    name = "weight_xsection_lumi",
    data = False,
)
weight_pdf_scale = Readers.WeightPdfScale(
    name = "weight_pdf_scale",
    #parents_to_skip = ["SingleTop", "QCD", "Diboson"],
    data = False,
)
weight_qcd_ewk = Readers.WeightQcdEwk(
    name = "weight_qcd_ewk",
    input_paths = {
        "ZJetsToNuNu": (datapath+"/qcd_ewk/vvj.dat", "vvj_pTV_{}"),
        "WJetsToLNu":  (datapath+"/qcd_ewk/evj.dat", "evj_pTV_{}"),
        "DYJetsToLL":  (datapath+"/qcd_ewk/eej.dat", "eej_pTV_{}"),
    },
    underflow = True,
    overflow = True,
    formula = (
        "((K_NNLO + d1k_qcd*d1K_NNLO + d2k_qcd*d2K_NNLO + d3k_qcd*d3K_NNLO)"
        "*(1 + kappa_EW + d1k_ew*d1kappa_EW + isz*(d2k_ew_z*d2kappa_EW + d3k_ew_z*d3kappa_EW)"
                                           "+ isw*(d2k_ew_w*d2kappa_EW + d3k_ew_w*d3kappa_EW))"
        "+ dk_mix*dK_NNLO_mix)"
        "/(K_NLO + d1k_qcd*d1K_NLO + d2k_qcd*d2K_NLO + d3k_qcd*d3K_NLO)"
    ),
    params = [
        "K_NLO", "d1K_NLO", "d2K_NLO", "d3K_NLO", "K_NNLO", "d1K_NNLO",
        "d2K_NNLO", "d3K_NNLO", "kappa_EW", "d1kappa_EW", "d2kappa_EW",
        "d3kappa_EW", "dK_NNLO_mix",
    ],
    variation_names = [
        "d1k_qcd", "d2k_qcd", "d3k_qcd", "d1k_ew", "d2k_ew_z", "d2k_ew_w",
        "d3k_ew_z", "d3k_ew_w", "dk_mix",
    ],
    data = False,
)

hdf5_reader = Collectors.HDF5Reader(
    name = "hdf5_reader",
)

sequence = [
    # Setup caching, nsig and source
    (event_tools, NullCollector()),
    # Creates object collections accessible through the event variable. e.g.
    # event.Jet.pt rather than event.Jet_pt.
    (collection_creator, NullCollector()),
    # Try to keep GenPart branch stuff before everything else. It's quite big
    # and is deleted after use. Don't want to add the memory consumption of
    # this with all other branches
    (gen_boson_producer, NullCollector()),
    (lhe_part_assigner, NullCollector()),
    (gen_part_assigner, NullCollector()),
    # # Weighters. The generally just apply to MC and that logic is dealt with by
    # # the ScribblerWrapper.
    (weight_xsection_lumi, NullCollector()),
    (weight_pdf_scale, NullCollector()),
    (weight_qcd_ewk, NullCollector()),
    # Add collectors (with accompanying readers) at the end so that all
    # event attributes are available to them
    (hdf5_reader, NullCollector()),
]
