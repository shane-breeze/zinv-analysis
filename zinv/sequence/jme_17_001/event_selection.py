from utils.classes import EmptyClass
event_selection = EmptyClass()

lumi_selection = "ev: ev.IsCertified"
trigger_selection = "ev: ev.IsTriggered"
filter_selection = "ev: (ev.Flag_goodVertices>0.5) & "\
                       "(ev.Flag_globalTightHalo2016Filter>0.5) & "\
                       "(ev.Flag_HBHENoiseFilter>0.5) & "\
                       "(ev.Flag_HBHENoiseIsoFilter>0.5) & "\
                       "(ev.Flag_EcalDeadCellTriggerPrimitiveFilter>0.5) & "\
                       "(ev.Flag_eeBadScFilter>0.5) & "\
                       "(ev.Flag_BadChargedCandidateFilter>0.5) & "\
                       "(ev.Flag_BadGlobalMuon>0.5) & "\
                       "(ev.Flag_BadPFMuonFilter>0.5) & "\
                       "(ev.Flag_CloneGlobalMuon>0.5)"
dphi_jet_met_selection = "ev: ev.MinDPhiJ1234METnoX > 0.5"
dphi_jet_met_inv_selection = "ev: ev.MinDPhiJ1234METnoX <= 0.5"
muon_selection = "ev: (ev.MuonSelection.size == ev.MuonVeto.size) & (ev.MuonVeto.size == {})"
ele_selection = "ev: (ev.ElectronSelection.size == ev.ElectronVeto.size) & (ev.ElectronVeto.size == {})"
mtw_selection = "ev: (ev.MTW >= 30.) & (ev.MTW < 125.)"
mll_selection = "ev: (ev.MLL >= 80.) & (ev.MLL < 100.)"

ngen_boson_selection = "ev: True if ev.config.dataset.parent not in 'EWKV2Jets' else (ev.nGenBosons==1)"

# Selections
event_selection.data_selection = [
    ("lumi_selection", lumi_selection),
    ("trigger_selection", trigger_selection),
]

event_selection.mc_selection = [
    ("ngen_boson_selection", ngen_boson_selection),
]

event_selection.baseline_selection = [
    ("filter_selection", filter_selection),
]

event_selection.monojet_selection = [
    ("dphi_jet_met_selection", dphi_jet_met_selection),
    ("muon_selection_fmt_0", muon_selection.format(0)),
    ("ele_selection_fmt_0", ele_selection.format(0)),
]
event_selection.monojetsb_selection = [
    ("dphi_jet_met_selection", dphi_jet_met_selection),
    ("muon_selection_fmt_0", muon_selection.format(0)),
    ("ele_selection_fmt_0", ele_selection.format(0)),
]
event_selection.monojetsr_selection = [
    ("dphi_jet_met_selection", dphi_jet_met_selection),
    ("muon_selection_fmt_0", muon_selection.format(0)),
    ("ele_selection_fmt_0", ele_selection.format(0)),
]

event_selection.singlemuon_selection = [
    ("muon_selection_fmt_1", muon_selection.format(1)),
    ("ele_selection_fmt_0", ele_selection.format(0)),
    ("mtw_selection", mtw_selection),
]
event_selection.singlemuonsb_selection = [
    ("muon_selection_fmt_1", muon_selection.format(1)),
    ("ele_selection_fmt_0", ele_selection.format(0)),
    ("mtw_selection", mtw_selection),
]
event_selection.singlemuonsr_selection = [
    ("muon_selection_fmt_1", muon_selection.format(1)),
    ("ele_selection_fmt_0", ele_selection.format(0)),
    ("mtw_selection", mtw_selection),
]

event_selection.doublemuon_selection = [
    ("muon_selection_fmt_2", muon_selection.format(2)),
    ("ele_selection_fmt_0", ele_selection.format(0)),
    ("mll_selection", mll_selection),
]
event_selection.doublemuonsb_selection = [
    ("muon_selection_fmt_2", muon_selection.format(2)),
    ("ele_selection_fmt_0", ele_selection.format(0)),
    ("mll_selection", mll_selection),
]
event_selection.doublemuonsr_selection = [
    ("muon_selection_fmt_2", muon_selection.format(2)),
    ("ele_selection_fmt_0", ele_selection.format(0)),
    ("mll_selection", mll_selection),
]

event_selection.singleelectron_selection = [
    ("muon_selection_fmt_0", muon_selection.format(0)),
    ("ele_selection_fmt_1", ele_selection.format(1)),
    ("mtw_selection", mtw_selection),
]
event_selection.singleelectronsb_selection = [
    ("muon_selection_fmt_0", muon_selection.format(0)),
    ("ele_selection_fmt_1", ele_selection.format(1)),
    ("mtw_selection", mtw_selection),
]
event_selection.singleelectronsr_selection = [
    ("muon_selection_fmt_0", muon_selection.format(0)),
    ("ele_selection_fmt_1", ele_selection.format(1)),
    ("mtw_selection", mtw_selection),
]

event_selection.doubleelectron_selection = [
    ("muon_selection_fmt_0", muon_selection.format(0)),
    ("ele_selection_fmt_2", ele_selection.format(2)),
    ("mll_selection", mll_selection),
]
event_selection.doubleelectronsb_selection = [
    ("muon_selection_fmt_0", muon_selection.format(0)),
    ("ele_selection_fmt_2", ele_selection.format(2)),
    ("mll_selection", mll_selection),
]
event_selection.doubleelectronsr_selection = [
    ("muon_selection_fmt_0", muon_selection.format(0)),
    ("ele_selection_fmt_2", ele_selection.format(2)),
    ("mll_selection", mll_selection),
]

event_selection.monojetqcd_selection = [
    ("dphi_jet_met_inv_selection", dphi_jet_met_inv_selection),
    ("muon_selection_fmt_0", muon_selection.format(0)),
    ("ele_selection_fmt_0", ele_selection.format(0)),
]
event_selection.monojetqcdsb_selection = [
    ("dphi_jet_met_inv_selection", dphi_jet_met_inv_selection),
    ("muon_selection_fmt_0", muon_selection.format(0)),
    ("ele_selection_fmt_0", ele_selection.format(0)),
]
event_selection.monojetqcdsr_selection = [
    ("dphi_jet_met_inv_selection", dphi_jet_met_inv_selection),
    ("muon_selection_fmt_0", muon_selection.format(0)),
    ("ele_selectio_fmt_0", ele_selection.format(0)),
]
