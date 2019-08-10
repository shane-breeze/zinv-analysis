import oyaml as yaml

class Dataset(object):
    args = ["name", "parent", "isdata", "xsection", "lumi", "energy", "idx",
            "sumweights", "files", "associates", "tree"] #, "file_nevents"]
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
    default = datasets_dict["default"]

    temp_datasets = []
    for d in datasets_dict["datasets"].keys():
        if d not in temp_datasets:
            temp_datasets.append(d)

    for idx, dataset in enumerate(temp_datasets):
        dataset_kwargs = datasets_dict["datasets"][dataset]
        dataset_kwargs.update(default)
        dataset_kwargs["idx"] = idx
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
