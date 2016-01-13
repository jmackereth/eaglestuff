# eaglestuff
Tools for Analysing EAGLE simulations


Current version uses the functions in newetools.py, which is in development and not yet complete. The older and more clunky eagletools.py contains functions for saving haloes, and making various plots for single halos. None of this works without the readEagle python module installed.

import the module by using
```
import newetools as et
```
in the directory containing the .py file.

the default simulation which is loaded is the 100Mpc Reference run.
the function `et.loadparticles(run=default_run, sim=default_sim, tag=default_tag)` loads the PARTDATA
