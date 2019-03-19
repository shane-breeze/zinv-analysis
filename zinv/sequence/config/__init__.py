import imp
import os
from zinv.sequence.Readers import ScribblerWrapper

def build_sequence(
    sequence_cfg_path, outdir, es_path, pos_path, ts_path, ws_path, nuisances,
):
    seq = imp.load_source('sequence.sequence', sequence_cfg_path)
    for r, c in seq.sequence:
        r.event_selection_path = os.path.abspath(es_path)
        r.physics_object_selection_path = os.path.abspath(pos_path)
        r.trigger_selection_path = os.path.abspath(ts_path)
        r.weight_sequence_path = os.path.abspath(ws_path)
        r.nuisances = nuisances
        c.outdir = os.path.abspath(outdir)
    return [
        (ScribblerWrapper(reader), collector)
        for (reader, collector) in seq.sequence
    ]
