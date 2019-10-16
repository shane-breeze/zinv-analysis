[![CircleCI](https://circleci.com/gh/shane-breeze/zinv-analysis.svg?style=shield)](https://circleci.com/gh/shane-breeze/zinv-analysis)

[![codecov](https://codecov.io/gh/shane-breeze/zinv-analysis/branch/master/graph/badge.svg)](https://codecov.io/gh/shane-breeze/zinv-analysis)

# Z invisible analysis

This code processes CMS event-based data and simulation stored in a flat `ROOT.TTree` format (i.e. branches correspond to simple data types such as `bool`, `int`, `float`, ... or an `std::vector` of these data types). Typically, this is done on [nanoAOD](https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookNanoAOD). The output is a dataframe(s) of similar data types (with the exclusion of vectors) either directly taken from the nanoAOD files or derived from these variables to create an analysis-level dataframe.

This is achieved by reading in nanoAOD files with [uproot](https://github.com/scikit-hep/uproot) applying a set of modules to generate derived variables and storing these in a dataframe saved to disk. Yaml config files are passed to define the input data, modules and output.

## Usage

Install with pip:

```bash
pip install zinv-analysis
```

or in editable mode to alter the code:

```bash
git clone git@github.com:shane-breeze/zinv-analysis.git
cd zinv-analysis
pip install -e .
```

Either run with the CLI

```bash
zinv_analysis.py --help
```

or the python API

```python
import zinv
help(zinv.modules.analyse)
```

## Layout

### Interfaces

Interfaces to the underlying code is located in [analyse.py](https://github.com/shane-breeze/zinv-analysis/blob/master/zinv/modules/analyse.py) and [resume.py](https://github.com/shane-breeze/zinv-analysis/blob/master/zinv/modules/resume.py).

Scripts using these functions are found in [zinv/scripts/](https://github.com/shane-breeze/zinv-analysis/tree/master/zinv/scripts).

### Modules

A set of modules which create derived variables are found in [zinv/modules/readers](https://github.com/shane-breeze/zinv-analysis/tree/master/zinv/modules/readers). These modules are applied to the data with the (alphatwirl)[https://github.com/alphatwirl/alphatwirl] package and contain a class (possibly) with the `begin`, `event` and `end` methods.

The `begin` method is run at the start of processing the data to initialise some required parameters. The [EventTools](https://github.com/shane-breeze/zinv-analysis/blob/master/zinv/modules/readers/EventTools.py) module adds a `register_function` method to the `event` to allows functions to be cached for lazy-evaluation (e.g. the JEC variations function is not run if the JEC variations are not saved in the output).

The `event` method is applied to each iteration over the input data. This corresponds to a chunk of events which are loaded into numpy arrays with [uproot](https://github.com/scikit-hep/uproot). Here the derived variables are evaluated. However, because of thee lazy-evaluation this is typically blank for most modules.

The `end` method ia applied at the end of processing to clear up anything that needs to be cleared. If this is run in multiprocessing or batch processing mode then modules are serialised. Lambda functions are not serialisable and hence must be created with the `begin` method and cleared in the `end` method.

### Output

A special module defines the output. Currently this is [HDF5.py](https://github.com/shane-breeze/zinv-analysis/blob/master/zinv/modules/collectors/HDF5.py). Instead of creating derived variables, this module will evaluate the previously defined functions and store them into a `.h5` file using pandas. The actual output is defined by yaml config.

### Config

The yaml config is defined externally by the user and controls where the datasets are found, which modules are applied and the output into the dataframes. However, with this flexibility extra care must be taken so modules which depend on each other are defined and in the correct order. For example, if the JEC variations are saved by the `HDF5` module, then the `JECVariation` module must be included in the sequence before the output module.
