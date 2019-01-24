from utils.classes import EmptyClass
event_selection = EmptyClass()

mll_selection = "ev: (ev.GenPartBoson_mass >= 71.) & (ev.GenPartBoson_mass < 111.)"

# Selections
event_selection.data_selection = []
event_selection.mc_selection = []
event_selection.baseline_selection = []
event_selection.met_trigger_selection = []

event_selection.monojet_selection = []
event_selection.monojetqcd_selection = []

event_selection.singlemuon_selection = []
event_selection.singlemuonqcd_selection = []
event_selection.singlemuonplus_selection = []
event_selection.singlemuonminus_selection = []
event_selection.doublemuon_selection = [("mll_selection", mll_selection)]
event_selection.triplemuon_selection = []
event_selection.quadmuon_selection = []

event_selection.singleelectron_selection = []
event_selection.singleelectronqcd_selection = []
event_selection.singleelectronplus_selection = []
event_selection.singleelectronminus_selection = []
event_selection.doubleelectron_selection = [("mll_selection", mll_selection)]
event_selection.doubleelectron_alt_selection = []
event_selection.tripleelectron_selection = []

event_selection.singletau_selection = []
event_selection.singletauqcd_selection = []
event_selection.doubletau_selection = []
