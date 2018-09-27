import imp
from sequence.Readers import ScribblerWrapper

def build_sequence(sequence_cfg_path, outdir):
    seq = imp.load_source('sequence.sequence', sequence_cfg_path)
    for _, c in seq.sequence:
        c.outdir = outdir
    return [(ScribblerWrapper(reader), collector)
            for (reader, collector) in seq.sequence]
