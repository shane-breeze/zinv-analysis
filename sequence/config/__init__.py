import imp
from sequence.Readers import ScribblerWrapper

def build_sequence(sequence_cfg_path, outdir, es_path, pos_path, ts_path, ws_path):
    seq = imp.load_source('sequence.sequence', sequence_cfg_path)
    for r, c in seq.sequence:
        r.event_selection_path = es_path
        r.physics_object_selection_path = pos_path
        r.trigger_selection_path = ts_path
        r.weight_sequence_path = ws_path
        c.outdir = outdir
    return [
        (ScribblerWrapper(reader), collector)
        for (reader, collector) in seq.sequence
    ]
