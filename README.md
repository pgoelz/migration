Read me
=======
This repository contains the simulation code for the paper

> REDACTED AUTHORS: Migration as Submodular Optimization. 2018.

Requirements
------------
- Python 3.6 (higher versions might work, but so far are not supported by
  Gurobi)
- Gurobi with gurobipy python bindings (we used version 8.0.1)
- Igraph (0.7.1)
- Seaborn (0.9.0)
- Pandas (0.23.4)
- Jupyter (4.4.0)

For academic use, Gurobi provides free licenses at
<http://www.gurobi.com/academia/for-universities>.

At the time of this writing, the
[installation manual](http://igraph.org/python/#pyinstallosx) for igraph on OSX
is out of date. We had to directly install `brew install igraph`, then
`pip install python-igraph`. Should you see the warning

> DeprecationWarning: To avoid name collision with the igraph project, this
> visualization library has been renamed to  'jgraph'. Please upgrade when
> convenient.

when importing igraph, then the following StackOverflow answer might help:
<https://stackoverflow.com/a/36203972>.

Replication of experiments in the paper
---------------------------------------
The individual experiments are provided as Jupyter/IPython notebooks. Running
`jupyter notebook num_localities.ipynb`, for instance, opens a browser window,
in which you can see our simulation results and easily rerun them. While we fix
the random seed to `0` in all our experiments, your simulations might produce
slightly different results due to different library versions being used,
non-determinism in these libraries or Python implementation details such as the
iteration order over dictionaries. For what it is worth, we ran our simulations
on a MacBook Pro (2017) with a 3.1 GHz Intel Core Duo i5 processor, with 16 GB
of RAM, and running MacOS 10.12.6.

Questions
---------
For questions on the simulations, please contact REDACTED.
