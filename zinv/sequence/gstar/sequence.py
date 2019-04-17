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
    maxsize = int(2*1024**3), # 6 GB
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

sqlite_reader = Collectors.SqliteReader(
    name = "sqlite_reader",
    cfg = os.path.join(collpath, "Sqlite_cfg.yaml"),
)
sqlite_collector = Collectors.SqliteCollector(
    name = "sqlite_collector",
    cfg = os.path.join(collpath, "Sqlite_cfg.yaml"),
)

sequence = [
    # Setup caching, nsig and source
    (event_tools, NullCollector()),
    # Creates object collections accessible through the event variable. e.g.
    # event.Jet.pt rather than event.Jet_pt.
    (collection_creator, NullCollector()),
    # selection and weight producers. They only create functions and hence can
    # be placed near the start
    (weight_producer, NullCollector()),
    (selection_producer, NullCollector()),
    # # Try to keep GenPart branch stuff before everything else. It's quite big
    # # and is deleted after use. Don't want to add the memory consumption of
    # # this with all other branches
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
    (sqlite_reader, sqlite_collector),
]
