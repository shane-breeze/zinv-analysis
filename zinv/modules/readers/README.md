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

Add a cached function which returns a boolean array for each object. True if the object doesn't overlap with another object in eta-phi space and False if it does. The use of this array is handled by the `SelectionProducer` module.

## ObjectFunctions

Add cached functions to shift object pts by a systematic uncertainty. The systematic uncertainty is set by its name in `event.source` and its value in terms of the number of sigma in `event.nsig`. Also determines the phi shift for MET-like quantities and dphi between objects and MET.
