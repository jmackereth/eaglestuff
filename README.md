# eaglestuff
Tools for Analysing EAGLE simulations


Current version uses the functions in newetools.py, which is in development and not yet complete. The older and more clunky eagletools.py contains functions for saving haloes, and making various plots for single halos. None of this works without the readEagle python module installed.

import the module by using
```
import newetools as et
```
in the directory containing the .py file.

the default simulation which is loaded is the 50Mpc Reference run.
the function `et.loadparticles(run=default_run,sim=default_sim,tag=default_tag)` loads the PARTDATA for every particle and returns an array, which is confusingly long (i will update this to explain it another time) but for now the format of this does not matter because the next function you need to use is `et.stackparticles(partdat)` where you input the partdat array and you get a much neater array of particles in the format:
> PartType | GroupNumber | SubGroupNumber | x | y | z | v_x | v_y | v_z | mass 

for each particle (this array is long).

you must also use `et.loadfofdat(run=default_run,sim=default_sim,tag=default_tag)` and get the total data for each central halo.

then `et.halo(partstack,fofdat,groupnum, plot = True)` aligns and plots a halo selected by groupnum (centrals only!). This function is still heavily under development and is no where near complete (use with caution).


