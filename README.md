# NET_forecast_python

Git repository for this python NET calculator.
Repository was developped on Harvard odyssey cluster and is currently online at
http://bicep.rc.fas.harvard.edu/dbarkats/postings/NET_calculator/
A full detailed documentation of this calculator is linked at the top of that site.

This repository contains:
   - a basic constant library to store physical constants (contants.py)
   - the NET calculator libraries (NETlib_v0X.py).
   - a symlink that links to the latest version of the NET libraries (NETlib.py@)
   - a python NET wrapper to run the calculator from the command line (NET.py)

Additional folders/files needed are:
   - am_spectra: contains 2 subfolders: sites and zones. The zones (rarely used) contains the annual profiles for the 5 zones defined in Scott Paine's am cookbook (https://www.cfa.harvard.edu/~spaine/am/cookbook/unix/zonal/), expanded to multiple altitudes and multiple elevations.  The sites contains the annual profiles for all the sites defined in Scott Paine's am cookbook (https://www.cfa.harvard.edu/~spaine/am/cookbook/unix/sites/). Some sites (ALMA, ACT, Spole ) have been expanded to work at any elevation from 14 to 90 deg.