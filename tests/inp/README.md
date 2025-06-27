# INPUT FILES FOR THE TESTS

This document explains the use of the files in this directory, that serve as an input for the tests in the parent 
directory. They are all meant to be both light and representative of the ones used for scientific results.

### `- apec_fakeit_for_test.pha`
This file was created with `fakeit` using the `bapec` model (*kT* = 7 keV, *Abund* = 0.2 Z<sub>Sun</sub>, *z* = 0.15,
*norm* = 0.1). Thermal broadening included, Anders & Grevesse (1989) abundance table. The response
files are `resolve_h5ev_2019a.rmf` and `resolve_pnt_heasim_noGV_20190701.arf`, the exposure was set to 1500 s. 
Statistical fluctuations turned *ON*.

### `- apec_fakeit_nostat_for_test.pha`
Same as `apec_fakeit_for_test.pha`, but with statistical fluctuations were turned *OFF* so that in every energy the 
spectrum should correspond to the model precisely.

### `- bapec_fakeit_for_test.pha`
Same as `apec_fakeit_for_test.pha` but using the `bapec` model (*kT* = 5 keV, *Abund* = 0.3 Z<sub>Sun</sub>, *z* = 0.2,
*b* = 300 km/s, *norm* = 0.1). Statistical fluctuations turned *ON*.

### `- bapec_fakeit_nostat_for_test.pha`
Same as `bapec_fakeit_for_test.pha`, but with statistical fluctuations were turned *OFF* so that in every energy the
spectrum should correspond to the model precisely.

### `- resolve_pnt_heasim_noGV_20190701.arf, resolve_h5ev_2019a.rmf`  
*XRISM-Resolve* response files. They were chosen between the ones available in Xspec as they are the largest ones below 
10MB (the Github file size limit).  

### `- snap_Gadget_sample, snap_sample.hdf5`
A *Gadget* snapshot containing a galaxy cluster simulated at low resolution in Gadget-2 and HDF5 format, 
respectively.  

### `- test_emission_table.fits`
An emission table with `APEC` spectra at *Abund* = 0.3 Z<sub>Sun</sub>. The table contains spectra at 6 different 
redshift and 40 different temperatures.
