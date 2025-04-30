# XRAYSIM

## Requirements

This Python package requires the installation of Xspec and Sixte to work properly. It also needs some basic 
configuration before being installed and used. Here is the list of things you need to do.

## Before installing Xraysim
1) Install Xspec, if not done already, including its Python library PyXspec, available at
[this website](https://heasarc.gsfc.nasa.gov/xanadu/xspec/)
2) Install Sixte, if not done already. Sixte is an X-ray telescope/instrument simulator available at 
[this website](https://www.sternwarte.uni-erlangen.de/sixte/) that includes several telescopes. Ensure you download the 
pacakge(s) relative to the instrument(s) you want to use.

## Environment configuration
Before running the installation script, you should setup your environment. The following instructions should be part of 
your default environment configuration, so you should put them in your ~/.cshrc or ~/.bashrc or ~/.profile setup files.
1) Setup the `XRAYSIM` environment variable, pointing to this folder (i.e. the folder where this file is located)
2) Setup the `SIXTE` environment variable, pointing to the folder where Sixte is installed: this should have already 
been done during the installation of Sixte
3) Include the PyXspec library in your Python path: this should be located in the `lib/python` subfolder of your 
`HEASoft` packages folder (necessary for the installation of Xspec), pointed by the `HEADAS` enviroment variable 
during the installation of Xspec
4) For easier use, you can also include this folder in your Python path, to allow `import xraysim` from your Python 
scripts

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
compile the Cython source files. Then run `pytest` to ensure the package is installed and configured correctly: some 
warnings may be issued, but this is ok. If an error occurs, please, send me an email at mauro.roncarelli@inaf.it 
including the error message.

## Instrument configuration
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
of the file itself. If an instrument requires more .xml files, list them in the `xml` attribute, comma separated. Keep 
in mind that Sixte requires the instruments to be placed in the `$SIXTE/share/sixte/instruments` folder, so this is the 
place where you have to look for these info. More instruments should be listed one after the other, as a JSON array, 
keeping the `[]` at the beginning/end. Some instruments are already set up in the `sixte_instruments.json` file 
(including a fake instrument for testing) and they should work if you have downloaded the corresponding files. 

For older Sixte versions (before version 3) also an additional advanced xml file was required in the same subfolder: 
you can add the `adv_xml` attribute to configure it (see `sixte_instruments_v2.json` for some example). Configuring 
eRosita requires the additional attribute `"special": "erosita"`.

In order to verify if your instruments are set up correctly, open a Python console and run
'''
from xraysim import sixte
sixte.instruments.verify()
'''
You should get the list of your instruments marked `OK` if configured correctly, or with an error message otherwise.

## Using Xraysim
Open a Python script or console and type:
```
import xraysim
```
and you are ready to go.