from utils.classes import EmptyClass
event_selection = EmptyClass()

lumi_selection = "ev: ev.IsCertified"
trigger_selection = "ev: ev.Is{}Triggered"
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
met_selection = "ev: ev.METnoX.pt > 200."
dphi_jet_met_selection = "ev: ev.MinDPhiJ1234METnoX > 0.5"
dphi_jet_met_inv_selection = "ev: ev.MinDPhiJ1234METnoX <= 0.5"
dcalo_pfmet_selection = "ev: ev.MET.dCaloMET < 0.6"
jet_selection = "ev: (ev.JetSelection.size > 0) & "\
                    "(ev.JetSelection.size == ev.JetVeto.size)"
lead_jet_selection = "ev: (get_nth_object(ev.JetSelection.pt, 0, ev.size)>200.) & "\
                         "(get_nth_object(ev.JetSelection.chHEF, 0, ev.size)>0.1) & "\
                         "(get_nth_object(ev.JetSelection.chHEF, 0, ev.size)<0.95)"
muon_selection = "ev: (ev.MuonSelection.size == ev.MuonVeto.size) & (ev.MuonVeto.size == {})"
muon_total_charge = "ev: ev.MuonTotalCharge == {}"
ele_total_charge = "ev: ev.ElectronTotalCharge == {}"
ele_selection = "ev: (ev.ElectronSelection.size == ev.ElectronVeto.size) & (ev.ElectronVeto.size == {})"
tau_selection = "ev: (ev.TauSelection.size == ev.TauVeto.size) & (ev.TauVeto.size == {})"
pho_veto = "ev: (ev.PhotonSelection.size == ev.PhotonVeto.size) & (ev.PhotonVeto.size == 0)"
nbjet_veto = "ev: (ev.nBJetSelectionMedium == 0) if ev.config.dataset.isdata else np.ones(ev.size, dtype=bool)"
mtw_selection = "ev: (ev.MTW >= 30.) & (ev.MTW < 125.)"
mll_selection = "ev: (ev.MLL >= 71.) & (ev.MLL < 111.)"
met_pf_selection = "ev: ev.MET_pt > 100."

ngen_boson_selection = "ev: np.ones(ev.size, dtype=bool) if ev.config.dataset.parent not in 'EWKV2Jets' else (ev.nGenBosons==1)"

blind_mask = "ev: ev.BlindMask"

# Selections
event_selection.data_selection = [
    ("lumi_selection", lumi_selection),
    ("trigger_selection", trigger_selection.format("")),
]

event_selection.mc_selection = [
    ("ngen_boson_selection", ngen_boson_selection),
]

event_selection.baseline_selection = [
    ("filter_selection", filter_selection),
    ("met_selection", met_selection),
    ("dcalo_pfmet_selection", dcalo_pfmet_selection),
    ("jet_selection", jet_selection),
    ("lead_jet_selection", lead_jet_selection),
    ("pho_veto", pho_veto),
    ("nbjet_veto", nbjet_veto),
]

event_selection.monojet_selection = [
    ("blind_mask", blind_mask),
    ("dphi_jet_met_selection", dphi_jet_met_selection),
    ("muon_selection_fmt_0", muon_selection.format(0)),
    ("ele_selection_fmt_0", ele_selection.format(0)),
    ("tau_selection_fmt_0", tau_selection.format(0)),
]

event_selection.singlemuon_selection = [
    ("dphi_jet_met_selection", dphi_jet_met_selection),
    ("muon_selection_fmt_1", muon_selection.format(1)),
    ("ele_selection_fmt_0", ele_selection.format(0)),
    ("tau_selection_fmt_0", tau_selection.format(0)),
    ("mtw_selection", mtw_selection),
]
event_selection.singlemuonqcd_selection = [
    ("dphi_jet_met_inv_selection", dphi_jet_met_inv_selection),
    ("muon_selection_fmt_1", muon_selection.format(1)),
    ("ele_selection_fmt_0", ele_selection.format(0)),
    ("tau_selection_fmt_0", tau_selection.format(0)),
    ("mtw_selection", mtw_selection),
]
event_selection.singlemuonplus_selection = [
    ("muon_total_charge_fmt_pve1", muon_total_charge.format(1)),
]
event_selection.singlemuonminus_selection = [
    ("muon_total_charge_fmt_nve1", muon_total_charge.format(-1)),
]

event_selection.doublemuon_selection = [
    ("blind_mask", blind_mask),
    ("dphi_jet_met_selection", dphi_jet_met_selection),
    ("muon_selection_fmt_2", muon_selection.format(2)),
    ("ele_selection_fmt_0", ele_selection.format(0)),
    ("tau_selection_fmt_0", tau_selection.format(0)),
    ("mll_selection", mll_selection),
]

event_selection.triplemuon_selection = [
    ("dphi_jet_met_selection", dphi_jet_met_selection),
    ("muon_selection_fmt_3", muon_selection.format(3)),
    ("ele_selection_fmt_0", ele_selection.format(0)),
    ("tau_selection_fmt_0", tau_selection.format(0)),
]

event_selection.quadmuon_selection = [
    ("dphi_jet_met_selection", dphi_jet_met_selection),
    ("muon_selection_fmt_4", muon_selection.format(4)),
    ("ele_selection_fmt_0", ele_selection.format(0)),
    ("tau_selection_fmt_0", tau_selection.format(0)),
]

event_selection.singleelectron_selection = [
    #("trigger_ele_selection", trigger_selection.format("SingleElectron")),
    ("met_pf_selection", met_pf_selection),
    ("dphi_jet_met_selection", dphi_jet_met_selection),
    ("muon_selection_fmt_0", muon_selection.format(0)),
    ("ele_selection_fmt_1", ele_selection.format(1)),
    ("tau_selection_fmt_0", tau_selection.format(0)),
    ("mtw_selection", mtw_selection),
]
event_selection.singleelectronqcd_selection = [
    #("trigger_ele_selection", trigger_selection.format("SingleElectron")),
    ("met_pf_selection", met_pf_selection),
    ("dphi_jet_met_inv_selection", dphi_jet_met_inv_selection),
    ("muon_selection_fmt_0", muon_selection.format(0)),
    ("ele_selection_fmt_1", ele_selection.format(1)),
    ("tau_selection_fmt_0", tau_selection.format(0)),
    ("mtw_selection", mtw_selection),
]
event_selection.singleelectronplus_selection = [
    ("ele_total_charge_fmt_pve1", ele_total_charge.format(1)),
]
event_selection.singleelectronminus_selection = [
    ("ele_total_charge_fmt_nve1", ele_total_charge.format(-1)),
]

event_selection.doubleelectron_selection = [
    ("blind_mask", blind_mask),
    #("trigger_ele_selection", trigger_selection.format("SingleElectron")),
    ("dphi_jet_met_selection", dphi_jet_met_selection),
    ("muon_selection_fmt_0", muon_selection.format(0)),
    ("ele_selection_fmt_2", ele_selection.format(2)),
    ("tau_selection_fmt_0", tau_selection.format(0)),
    ("mll_selection", mll_selection),
]

event_selection.tripleelectron_selection = [
    #("trigger_ele_selection", trigger_selection.format("SingleElectron")),
    ("dphi_jet_met_selection", dphi_jet_met_selection),
    ("muon_selection_fmt_0", muon_selection.format(0)),
    ("ele_selection_fmt_3", ele_selection.format(3)),
    ("tau_selection_fmt_0", tau_selection.format(0)),
]
event_selection.quadelectron_selection = [
    #("trigger_ele_selection", trigger_selection.format("SingleElectron")),
    ("dphi_jet_met_selection", dphi_jet_met_selection),
    ("muon_selection_fmt_0", muon_selection.format(0)),
    ("ele_selection_fmt_4", ele_selection.format(4)),
    ("tau_selection_fmt_0", tau_selection.format(0)),
]

event_selection.monojetqcd_selection = [
    ("dphi_jet_met_inv_selection", dphi_jet_met_inv_selection),
    ("muon_selection_fmt_0", muon_selection.format(0)),
    ("ele_selection_fmt_0", ele_selection.format(0)),
    ("tau_selection_fmt_0", tau_selection.format(0)),
]

event_selection.singletau_selection = [
    ("dphi_jet_met_selection", dphi_jet_met_selection),
    ("muon_selection_fmt_0", muon_selection.format(0)),
    ("ele_selection_fmt_0", ele_selection.format(0)),
    ("tau_selection_fmt_1", tau_selection.format(1)),
]

event_selection.singletauqcd_selection = [
    ("dphi_jet_met_inv_selection", dphi_jet_met_inv_selection),
    ("muon_selection_fmt_0", muon_selection.format(0)),
    ("ele_selection_fmt_0", ele_selection.format(0)),
    ("tau_selection_fmt_1", tau_selection.format(1)),
]

event_selection.doubletau_selection = [
    ("dphi_jet_met_selection", dphi_jet_met_selection),
    ("muon_selection_fmt_0", muon_selection.format(0)),
    ("ele_selection_fmt_0", ele_selection.format(0)),
    ("tau_selection_fmt_2", tau_selection.format(2)),
]

event_selection.met_trigger_selection = [
    ("met_trigger_selection", trigger_selection.format("MET")),
]

event_selection.doubleelectron_alt_selection = [
    ("filter_selection", filter_selection),
    ("met_low_selection", "ev: ev.METnoX.pt <= 200."),
    ("jet_selection", jet_selection),
    ("ele_selection_fmt_2", ele_selection.format(2)),
]
