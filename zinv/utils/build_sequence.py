import os
import importlib
import oyaml as yaml
from alphatwirl.loop import NullCollector
from zinv.modules.readers import ScribblerWrapper

def open_yaml(path):
    with open(path, 'r') as f:
        return yaml.full_load(f)

def import_function(module):
    path = ".".join(module.split(".")[:-1])
    attr = module.split(".")[-1]
    return getattr(importlib.import_module(path), attr)

def build_sequence(
    sequence_cfg_path, outdir, es_path, pos_path, ts_path, hdf_path,
):
    sequence = []
    sequence_cfg = open_yaml(sequence_cfg_path)
    for module in sequence_cfg["sequence"]:
        reader_cls = import_function(module["module"])
        reader = reader_cls(name=module["name"], **module.get("args", {}))

        reader.event_selection_path = os.path.abspath(es_path)
        reader.physics_object_selection_path = os.path.abspath(pos_path)
        reader.trigger_selection_path = os.path.abspath(ts_path)
        reader.hdf5_config_path = os.path.abspath(hdf_path)
        reader.outdir = os.path.abspath(outdir)

        sequence.append((ScribblerWrapper(reader), NullCollector()))
    return sequence
