import numpy as np
import xspec as xsp


def str2bool(v):
    """

    :param v:
    :return:
    """
    if v == 'True':
        return True
    elif v == 'False':
        return False


class XspecModel:
    def __init__(self, model_name: str, energy: np.array) -> None:
        """
        This Xspec Model constructor sets the PyXspec model energies, initializes the APEC version, and configures the
        emission model based on the string variable model_name. It can be either APEC or VVAPEC.
        :param model_name: str - type of X-ray Emission Model apec/vvapec
        :param energy: list of float - Represents the range of energy values in KeV for spectrum calculation in pyxspec.
        """
        self.xspec_model_name = model_name
        # TODO: This line is creating a problem to other procedures: we should restore the AllModels default after
        # the methods are called
        xsp.AllModels.setEnergies(f"{energy.min()} {energy.max()} {len(energy) - 1} lin")
        xsp.Xset.addModelString("APECROOT", "3.0.9")
        # TODO: check if we need to add xsp.Xset.addModelString("APECTHERMAL", "yes")

        # This is to turn off the logs
        xsp.Xset.chatter = 0
        self.xspec_model = xsp.Model(self.xspec_model_name)

    # doesn't change the object itself, that's why we have this warning
    def set_xspec_commands(self, commands: dict) -> None:
        """
        This class method set up all the commands for the XSPEC model
        :param commands: dict : dictionary of commands specific to XSPEC which are set up iteratively inside a loop
        :return: None
        """
        xspec_settings = {
            'abund': lambda cmd: setattr(xsp.Xset, cmd['method'], cmd['arg']),
            'addModelString': lambda cmd: xsp.Xset.addModelString(cmd['arg'][0], cmd['arg'][1]),
        }

        for command in commands:
            xspec_settings.get(command['method'], lambda cmd: None)(command)

    def calculate_spectrum(self, z, temperature, metallicity, element_index,
                           norm) -> np.array:
        """
        This class method computes the X-ray emission spectra for a gas particle using Pyxspec.
        :param z: float - redshift for the gas particle
        :param temperature: float - Temperature in keV in for the gas particle
        :param metallicity: list of float - metallicity array normalized to Anders and Grevesse solar abundance values
        :param element_index: list of abundance to set
        :param norm: float - xspec normalization value, units - 10^-14 cm^-5
        :return: the emission spectra for the gas particle in the units -
                norm * units from xspec module--->(10^-14 cm^-5) * (photons s^-1 cm^3)---->10^-14 photons s^-1 cm-2
        """
        params = {1: temperature, 32: z, 33: norm} if self.xspec_model_name == 'vvapec' \
            else {1: temperature, 3: z, 4: norm} if self.xspec_model_name == 'apec' \
            else None
        # print(element_index,metallicity)
        if (params is not None) and (len(element_index)>1):
            params.update({i + 2: metallicity[i] for i in element_index.tolist()})
        else:
            params.update({2: metallicity[0]})

        params = {key: np.float64(value) for key, value in params.items()}

        #for key, value in params.items():
        #    print(f"Data type of {key}: {(value)}")

        self.xspec_model.setPars(params)

        # self.xspec_model.show()
        result = self.xspec_model.values(0)

        return np.array(result)
