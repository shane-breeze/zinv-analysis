# Z invisible analysis modules

The modules defined here are designed as follows

## CertifiedLumiChecked

Add a cached functions which returns a numpy array of booleans if the event is within the provided JSON file provided by the `lumi_json_path` argument.

## CollectionCreator

In nanoAOD physics objects are labelled with the object name and attribute split by an underscore, e.g. `Muon_pt`, `Jet_eta`, `Photon_sieie`. This module adds the object name as an attribute to the event with attributes as class variables. For example the jet pt can be accessed by both of the following

```
event.Jet_pt
event.Jet.pt
```

with the second only usable as a result of this module. The collections to create are given as an argument.

## EventFunctions

Defines cached functions to calculate event-based variables, such as, `METnoX`, `MinDPhiJ1234METnoX`, `MTW`, `MLL`, `LeptonCharge`, and others.

## EventTools

Handles caching of branches and functions.

## GenBosonProducer

Define the cached functions `nGenBosons` to count the number of generator-level bosons, `GenLepCandidates` to create a collection outgoing generator-particles corresponding to the decay of a W or Z boson, and `GenPartBoson` to create a collection of bosons reconstructed from the `GenLepCandidates`.

## JecVariations

Use the POG provided inputs to apply the JER corrections and determine the JES and JER uncertainties on each jet.

## LHEPartAssigner

Add a cached function flagging the event as containing LHE particles which are electrons, muons or tau-leptons. This allows the separation of W and Z decays into electron, muon or tau.

## GenPartAssigner

Add cached function counting the number of generator-level leptonic tau decays. This allows the separration of W and Z tau decays into leptonic or hadronic.

## ObjectCrossCleaning

Add a cached function which returns a boolean array for each object. True if the object doesn't overlap with another object in eta-phi space and False if it does. The use of this array is handled by the `ObjectFunctions` module.

## ObjectFunctions

Add cached functions to shift object pts by a systematic uncertainty. The systematic uncertainty is set by its name in `event.source` and its value in terms of the number of sigma in `event.nsig`. Also determines the phi shift for MET-like quantities and dphi between objects and MET.

Also creates a skimmed version of already defined physics objects to create analysis-level physics objects. Multiple derived objects may arise from a single physics object with a different selection. The selection used comes from a boolean array created in the `SkimCollections` module.

## ScribblerWrapper

Wrapper to all modules added to the sequence to allow modules to be skipped for data or mc.

## SelectionProducer

Adds cached functions which return a boolean array corresponding to a desired selection. Typically this is not used to keep the selection flexible.

## SkimCollections

Adds cached functions to create a boolean array to define the objects corresponding to a particular selection. Instead of copying and creating a new collection the boolean array is created and updated and applied to the original collection whenever the user asks for it. The selection is defined in a yaml config file.

## TriggerChecker

Adds a cached function to check if an event has passed a certain set of triggers. Typically this is not used to keep the trigger selection flexible.

## WeightBTagging

Adds a cached function for the POG measured b-tagging scale-factors to be applied to MC. Includes the uncertainties.

## WeightMetTrigger

Adds a cached function for the MET trigger weights measured by this analysis.

## WeightObjects

Adds a cached function for the POG object scale factors applied to the event weight (i.e. not JECs)

## WeightPdfScale

Adds a cached function for the PDF and QCD scale variations

## WeightPileup

Adds a cached function for the pileup reweighting and systematic uncertainties

## WeightPreFiring

Adds a cached function for the prefiring weight and systematic uncertainties

## WeightQcdEwk

Adds a cached function for the higher order QCD and EWK corrections for V+jets and all the associated systematic uncertainties.

## WeightXsLumi

Adds a cached function for the `genWeight*xsection*lumi/sum(genWeight)` for each event.
