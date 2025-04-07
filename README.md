# HYPSTAT

The Hydrogen Production, Storage, and Transmission Analysis Tool (HYPSTAT) is a modeling framework developed by the National Renewable Energy Laboratory (NREL) to support the analysis of hydrogen systems. HYPSTAT focuses on key components of hydrogen infrastructure, particularly electrolytic hydrogen production, hydrogen storage, and hydrogen transmission, as part of the transition to decarbonized energy systems

HYPSTAT operates as a supply-to-demand model, taking a fixed exogenous demand as input and optimizing the design and operation of the hydrogen system to meet that demand. It determines cost-optimal configurations based on specified technology options and system constraints.

For detailed model documentation, see [documentation].

## Dependencies

HYPSTAT is programmed and run in Python. In addition to built-in packages, HYPSTAT uses the following external packages.

-[NumPy](https://numpy.org/)
-[Pandas](https://pandas.pydata.org/)
-[PyYaml](https://pypi.org/project/PyYAML/)
-[Pyomo](https://www.pyomo.org/)

Ensure that these are installed for HYPSTAT to run properly.

## Set Up

Clone the HYPSTAT repository to access the model code. 

To set up a model run (or runs), make and collect all the necessary input files. HYPSTAT uses a comprehensive set of csv files representing the hydrogen demand, technology options, electricity generation supply curves, costs, and network.

A run is ultimately defined by the scenario YAML file, which specifies model controls for that run and contains paths to other input files.

We suggest using the input files for the Case Study included here as a template for developing your own input files. See the full documentation for more detail on input files.

## Running the model

First, initialize a model scenario as an instance of the HYPSTAT class, using the YAML scenario file:

```python
from HYPSTAT import HYPSTAT
model = HYPSTAT(yaml_file_path='Path/To/YAML_scenario.yaml')
```

It is suggested to use HYPSTAT with a two-step solve process, in which it first solves for pipeline locations at a coarser time resolution, and then fully otpimizes the system at a finer time resolution. This can be directly done with the HYPSTAT model class:
```python
model.two_step_solve(solver='glpk')
model.write_outputs('Path/To/Output_Directory')
```

In this case, we use `write_outputs` to write the outputs of the model to the noted directory. Note that `write_outputs` will create the directory if it does not exist, and it will require user input on whether or not to overwrite the contents of the directory if it already exists.

You can manually run a single solve of the model by successively loading the inputs, model, and constraints, followed by solving the model. 
```python
model.load_inputs(optimize_pipelines=True)
model.load_model(optimize_pipelines=True)
model.load_constraints(optimize_pipelines=True)
model.solve_model(optimize_pipelines=True,solver=solver)
```

Each function can take a boolean argument, `optimize_pipelines`. If `True`, HYPSTAT will formulate a mixed-integer optimization problem which optimizes pipeline locations. The mixed-integer variables cause the model to take significantly longer to run. `optimize_pipelines` defaults to `False`, in which case HYPSTAT formulates a fully linear problem using pre-specified pipeline locations which typically solves much faster. (Pipeline sizes are optimized in either case.) `optimize_pipelines` must be the same value for all functions. 

If `optimize_pipelines` is false, you can pass `load_inputs` an optional parameter, `Pipeline_Exists`, to tell the model where to build pipelines. If no value is passed, the model will assume it is optimal to build a pipeline at all links and may build impractically small pipelines in some locations.

## Solver

The `solve_model` and `two_step_solve` functions take an argument, `solver`, which a string that specifies which solver to use. Right now, they can take either `'glpk'` or `'gurobi'` as the solver. You can manually program the use of other solvers with Pyomo within the `solve_model` function (a `solver` passed to `two_step_solve` is simply passed on to `solve_model`).

## Recommend Citation
Brauch, Joe, Yijin Li, Steven Percy, and Jesse Cruce. 2025. *HYPSTAT*. National Renewable Energy Laboratory and Swinburne University of Technology. https://github.com/NREL/HYPSTAT.

## Legal information
Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Swinburne University of Technology. All rights reserved.

NREL Software Record : SWR-23-04 "HYPSTAT (Hydrogen Production, Storage, and Transmission Analysis Tool)"
