# XRAYSIM

## Requirements

This Python package requires the installation of Xspec and Sixte to work properly. It also needs some basic 
configuration steps before being installed and used. Here is the list of things you need to do.

## Before installing Xraysim (if not done already...)
1) Install Xspec, available at [this website](https://heasarc.gsfc.nasa.gov/xanadu/xspec/), including its Python library PyXspec
2) Install Sixte, an X-ray telescope/instrument simulator available at 
[this website](https://www.sternwarte.uni-erlangen.de/sixte/) that includes several telescopes. Ensure you download the 
pacakge(s) relative to the instrument(s) you want to use.

## Environment configuration
Before running the installation script, you should set up your environment. The following instructions should be part 
of your default environment configuration, so you should put them in your ~/.cshrc or ~/.bashrc or ~/.profile setup 
files.
1) Setup the `XRAYSIM` environment variable, pointing to this folder (i.e. the folder where this file is located)
2) Setup the `SIXTE` environment variable, pointing to the folder where Sixte is installed: this should have already 
been done during the installation of Sixte
3) Include the PyXspec library in your Python path: this should be located in the `lib/python` subfolder of your 
`HEASoft` packages folder (necessary for the installation of Xspec), pointed by the `HEADAS` enviroment variable 
during the installation of Xspec
4) For easier use, you can also include this folder in your Python path, to allow `import xraysim` directly from your 
Python scripts

If you use Bourne shell variants (bash/sh/zsh) these are the commands:
```
export XRAYSIM=/path/to/this/folder
export SIXTE=/path/to/Sixte/folder
export PYTHONPATH=$HEADAS/lib/python:$PYTHONPATH
export PYTHONPATH=$XRAYSIM:$PYTHONPATH  # (optional)
```

In C-shell variants (tcsh/csh):
```
setenv XRAYSIM /path/to/this/folder
setenv SIXTE /path/to/Sixte/folder
setenv PYTHONPATH $HEADAS/lib/python:$PYTHONPATH
setenv PYTHONPATH $XRAYSIM:$PYTHONPATH  # (optional)
```

## Installation
Run `./makeit.sh`. This should install/update the necessary Python packages (listed in `requirements.txt`) and 
compile the Cython source files. This script will also create a subfolder called `xrism-resolve-test` files in your 
`$SIXTE/share/sixte/instruments` folder containing the configuration files of a fake instrument that is useful to test 
the code. Then run `pytest` to ensure the package is installed and configured correctly: some warnings may be issued, 
and some test may be skipped (marked 's') but this is ok. If an error occurs, please, [submit an issue]
(https://github.com/mroncarelli/xraysim/issues) including the error message and some detail.

## Instruments configuration
In order to use the Sixte instruments directly with Xraysim you need to set them up in the `sixte_instruments.json` 
file. Each instrument should be a JSON object like this one
```
  {
    "name": "instrument_name",
    "subdir": "path/to/xmlfile",
    "xml": "sixte_instrument_file.xml"
  },
```
In this object, `instrument_name` is a name of your choice to be used in your Python script to refer to the instrument. 
The other two properties should be the subfolder of the where the .xml file of the instrument is located and the name 
of the file itself. Keep in mind that Sixte requires the instruments to be placed in the 
`$SIXTE/share/sixte/instruments` folder, so this is the place where you have to look for these info. More instruments 
should be listed one after the other, as a JSON array, keeping the `[]` at the beginning/end of the file. Some 
instruments are already set up in the `sixte_instruments.json` file, including the `xrism-resolve-test` fake instrument 
mentioned above, and they should work if you have downloaded the corresponding files. 

For older Sixte versions (before version 3) also an additional advanced xml file may be required in the same subfolder: 
you can add the `adv_xml` attribute to configure it (see `sixte_instruments_v2.json` for some example). Configuring 
*eRosita* requires the additional attribute `"special": "erosita"`, and the 7 .xml files corresponding to the different 
telescope modules listed in the `xml` attribute, comma separated. For the survey mode the `attitude` attribute must 
also be added containing the attitude file (see the instruments `erosita` and `erass1` in the `sixte_instruments.json` 
file).

In order to verify if your instruments are set up correctly, open a Python console and run
```
from xraysim import sixte
sixte.instruments.verify()
```
You should get the list of your instruments marked `OK` if configured correctly, or with an error message otherwise.

## Using Xraysim
Open a Python script or console and type:
```
import xraysim
```
and you are ready to go.