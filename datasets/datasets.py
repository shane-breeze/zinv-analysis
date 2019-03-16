from __future__ import print_function
from collections import namedtuple
import oyaml as yaml
import os
import six
import logging

class Dataset(object):
    args = ["name", "parent", "isdata", "xsection", "lumi", "energy",
            "sumweights", "files", "associates", "tree"]
    def __init__(self, **kwargs):
        kwargs.setdefault("associates", [])
        kwargs.setdefault("tree", "Events")
        for arg in self.args:
            setattr(self, arg, kwargs[arg])

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__,
            ", ".join(["{} = {!r}".format(k, getattr(self, k))
                       for k in self.args[:-2]]),
            "associates = {}".format(", ".join([associate.name
                                                for associate in self.associates])),
        )

def get_datasets(path):
    with open(path, 'r') as f:
        datasets_dict = yaml.load(f)

    datasets = []
    #dataset_info_path = datasets_dict["path"]
    default = datasets_dict["default"]

    temp_datasets = []
    for d in datasets_dict["datasets"].keys():
        if d not in temp_datasets:
            temp_datasets.append(d)

    for dataset in temp_datasets:
        dataset_kwargs = datasets_dict["datasets"][dataset]
        dataset_kwargs.update(default)
        datasets.append(Dataset(**dataset_kwargs))

    # Associate samples
    not_extensions = [dataset
                      for dataset in datasets
                      if "_ext" not in dataset.name]
    for not_extension in not_extensions:
        associated_datasets = [dataset
                               for dataset in datasets
                               if not_extension.name in dataset.name]
        for dataset in associated_datasets:
            dataset.associates = associated_datasets

    return datasets

def _from_string(dataset, path, default):
    cfg = default.copy()
    cfg["name"] = dataset
    return _extend_info(cfg, dataset, path)


def _from_dict(dataset, path, default):
    cfg = default.copy()
    cfg.update(dataset)
    if "name" not in cfg:
        raise RuntimeError("Dataset provided as dict, without key-value pair for 'name'")
    return _extend_info(cfg, dataset["name"], path)


def _extend_info(cfg, name, path):
    infopath = path.format(name)
    try:
        with open(infopath, 'r') as f:
            info = yaml.load(f)
            if info["name"] != cfg["name"]:
                raise ValueError("Mismatch between expected and read names: "
                                 "{} and {}".format(cfg["name"], info["name"]))
            cfg.update(info)
    except IOError:
        logger = logging.getLogger(__name__)
        logger.warning("IOError: {}".format(infopath))

    return cfg


if __name__ == "__main__":
    datas = get_datasets("datasets/cms_public_test.yaml")
    for d in datas:
        print(d)
