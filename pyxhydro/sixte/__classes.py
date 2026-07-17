from distutils.spawn import find_executable
import json
import os
import warnings
from xml.etree import ElementTree

from .__shared import instrumentsDir, defaultSixteCommand, specialInstrumentsList


class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# CAUTION. This method is a monkeypatch: do not modify the arguments, otherwise it might stop working
def showwarning_new(message, category, filename, lineno, file=None, line=None):
    """
    Monkeypatch of warnings.showwarning
    :param message:
    :param category:
    :param filename:
    :param lineno:
    :param file:
    :param line:
    :return: None
    """
    msg_list = '\n'.join(warnings.formatwarning(message, category, filename, lineno, line).split('\n')[0:-2]).split(':')
    print("WARNING:" + ':'.join(msg_list[3:]))


warnings.showwarning = showwarning_new


class Instrument:
    """
    This class define a Sixte instrument, with the attributes necessary to
    """

    def __init__(self, name: str, subdir: str, xml: str, command=None, special=None, attitude=None):
        self.name = name
        self.subdir = subdir
        self.command = command if command else defaultSixteCommand
        self.special = special
        self.attitude = attitude
        self.path = instrumentsDir + "/" + self.subdir
        xml_ = xml.strip(' ,')
        dummy = []
        for xml in xml_.split(','):
            dummy.append(xml.strip())
        self.xml = tuple(dummy)

        # Getting data from XML file
        focal_length, diameter, arf, psf, vignetting, rmf, nxb = [], [], [], [], [], [], []

        def get_attrib(root, path, attrib, cast=None):
            element = root
            for tag in path:
                element = element.find(tag)
                if element is None:
                    return None

            value = element.attrib.get(attrib)
            if value is None:
                return None

            return cast(value) if cast else value

        for xml in self.xml:
            input_file = self.path + '/' + xml
            if os.path.isfile(input_file):
                root = ElementTree.parse(input_file).getroot()

                focal_length.append(get_attrib(root, ['telescope', 'focallength'], 'value', float))
                diameter.append(get_attrib(root, ['telescope', 'fov'], 'diameter', float))
                arf.append(get_attrib(root, ['telescope', 'arf'], 'filename'))
                psf.append(get_attrib(root, ['telescope', 'psf'], 'filename'))
                vignetting.append(get_attrib(root, ['telescope', 'vignetting'], 'filename'))
                rmf.append(get_attrib(root, ['detector', 'rmf'], 'filename'))
                nxb.append(get_attrib(root, ['detector', 'phabackground'], 'filename'))

            else:
                warnings.warn("File '" + input_file + "' not found.")

        self.focal_length = tuple(focal_length)
        self.diameter = tuple(diameter)
        self.arf = tuple(arf)
        self.psf =tuple(psf)
        self.vignetting = tuple(vignetting)
        self.rmf = tuple(rmf)
        self.nxb = tuple(nxb)

    @property
    def n_telescopes(self):
        return len(self.xml)

    def show(self):
        print("Name: " + self.name)
        print("Subdir: " + self.subdir)
        print("Xml: " + ', '.join(self.xml))
        print("Command: " + self.command)
        if self.special:
            print("Special: " + self.special)
        if self.attitude:
            print("Attitude: " + self.attitude)

    def verify(self, warn=False, verbose=1):
        """
        Verifies if the instrument has been correctly initialized: checks if the subfolder exists and contains the xml
        files, checks if the command exists and if the special attribute corresponds to the implemented ones. Checks
        that the files in the 'telescope' element of the xml file exist. Prints out a message containing "Instrument
        'name' OK" or the list of wrong things.
        :param warn: (bool) If set the messages are issued as warnings.
        :param verbose (int) If > 0 messages are printed out, otherwise not. Determines also the output type.
        :return: None if verbose > 0 (default) or (bool) True/False if the instrument is set up correctly or not
        """
        msg = []
        if not os.path.isdir(self.path):
            msg.append("Instrument " + self.name + " subdirectory does not exist: " + self.path)
        else:
            xml_ok = True
            for xml in self.xml:
                full_xml = self.path + '/' + xml
                if not os.path.isfile(full_xml):
                    xml_ok = False
                    msg.append("Instrument " + self.name + " xml file does not exist: " + full_xml)
            
            if xml_ok:
                for index in range(self.n_telescopes):
                    tag = " (telescope n. " + str(index + 1) + ")" if self.n_telescopes > 1 else ""
                    full_arf = self.path + '/' + self.arf[index]
                    if not os.path.isfile(full_arf):
                        msg.append("Instrument " + self.name + " arf file does not exist" + tag + ": " + full_arf)
                    full_psf = self.path + '/' + self.psf[index]
                    if not os.path.isfile(full_psf):
                        msg.append("Instrument " + self.name + " psf file does not exist" + tag + ": " + full_psf)
                    full_vig = self.path + '/' + self.vignetting[index]
                    if not os.path.isfile(full_vig):
                        msg.append("Instrument " + self.name + " vignetting file does not exist" + tag + ": " + full_vig)


        if find_executable(self.command) is None:
            msg.append("Instrument " + self.name + " command does not exist: " + self.command)

        if self.special:
            if self.special.strip().lower() not in [''] + specialInstrumentsList:
                msg.append("Instrument " + self.name + " special not recognized: " + self.special)

        ok = not msg
        if ok:
            msg = ["Instrument " + self.name + " OK"]

        if warn:
            if not ok:
                for msg in msg:
                    warnings.warn(msg)
        else:
            if verbose > 0:
                for msg in msg:
                    print(BColors.OKGREEN + msg + BColors.ENDC) if ok else print(BColors.WARNING + msg + BColors.ENDC)

        return None if verbose > 0 else ok

    def vignetted_arf(self, theta: float, outfile: str, index=0, overwrite=True) -> None:
        """
        Creates a vignetted arf based on the original one and on the vignetting file used by Sixte. Considers only the
        off-axis distance. TODO test
        :param theta: (float) Off-axis distance [deg].
        :param outfile: (str) Output file name.
        :param index: (int) Index of the telescope. Default 0.
        :param overwrite: (bool) If True (default) overwrites existing file.
        :return: None
        """
        from pyxhydro.sixte import vignetting # as __vignettingMethod
        vignetting(self.path + '/' + self.arf[index], self.path + '/' + self.vignetting[index], theta, outfile,
                   overwrite=overwrite)


def load_instrument(inp: dict) -> Instrument:
    """
    Initializes an instrument from a dictionary record.
    :param inp: (dict) Dictionary record usually derived from a JSON file
    """
    return Instrument(inp['name'], inp['subdir'], inp['xml'], inp.get('command'), inp.get('special'),
                      inp.get('attitude'))


class SixteInstruments:
    def __init__(self):
        self.__data = {}

    def keys(self):
        return self.__data.keys()

    def get(self, name) -> Instrument:
        return self.__data.get(name)

    def add(self, instrument: Instrument):
        self.__data[instrument.name] = instrument

    def load(self, inp):
        type_ = type(inp)
        if type_ == list:
            for d in inp:
                self.add(load_instrument(d))
        elif type_ == str:
            with open(inp) as file:
                json_data = json.load(file)
            self.load(json_data)
        else:
            raise ValueError("Invalid input type")

    def show(self, opt=None):
        if self.__data == {}:
            print("No instrument loaded")
        else:
            instr = self.get(opt)
            if instr is not None:
                instr.show()
            else:
                print("List of available instruments:")
                for name in self.__data:
                    if type(opt) == str and opt.strip().lower() == 'full':
                        instr = self.get(name)
                        print(" - " + name)
                        print("     Subdir: " + instr.subdir)
                        print("     Xml: " + ', '.join(instr.xml))
                        print("     Command: " + instr.command)
                        if instr.special:
                            print("     Special: " + instr.special)
                        if instr.attitude:
                            print("     Attitude: " + instr.attitude)
                    else:
                        print("   - " + name)

    def verify(self, warn=False):
        if self.__data == {}:
            print("No instrument loaded")
        else:
            for name in self.__data:
                self.get(name).verify(warn=warn, verbose=1)
