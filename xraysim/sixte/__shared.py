import os
import warnings

# SIXTE instruments configuration file
instrumentsConfigFile = os.path.join(os.path.dirname(__file__), '../../sixte_instruments.json')

# SIXTE instruments parent directory
instrumentsDir = os.environ.get('SIXTE') + '/share/sixte/instruments'

# Default SIXTE command
defaultSixteCommand = 'sixtesim'

# List of special instruments implemented in xraysim
specialInstrumentsList = ['erosita']

# List of keywords to transfer from one file to the following
keywordList = ['INFO', 'SIM_TYPE', 'SIM_FILE', 'SP_FILE', 'SIMPUT_F', 'PARENT_F', 'PROJ', 'X_MIN', 'X_MAX', 'Y_MIN',
               'Y_MAX', 'Z_MIN', 'Z_MAX', 'Z_COS', 'D_C', 'NPIX', 'NENE', 'ANG_PIX', 'ANG_MAP', 'ISO_T', 'SMOOTH',
               'VPEC', 'NSAMPLE', 'NH', 'RA_C', 'DEC_C', 'FLUXSC', 'T_CUT']

def get_version():
    """
    Gets SIXTE version
    :return: (str) String containing the SIXTE version, None if undetermined (warning)
    """
    svout = os.popen('sixteversion').read().split('\n')
    svout_line0 = svout[0].split(' ')
    warnmsg = "Unable to verify SIXTE version"
    if len(svout_line0) == 3:
        if svout_line0[0].lower() == 'sixte' and svout_line0[1].lower() == 'version':
            return svout_line0[2]
        else:
            warnings.warn(warnmsg)
            return None
    else:
        warnings.warn(warnmsg)
        return None


# SIXTE version (string)
version = get_version()

# SIXTE version (tuple of int)
versionTuple = tuple([int(x) for x in version.split('.')]) if version is not None else (0, 0, 0)
