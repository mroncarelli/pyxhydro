import os
import sys

sys.path.append(os.environ.get("HEADAS") + "/lib/python")
# TODO: the two lines above are necessary only to make the code work in IntelliJ (useful for debugging), os is used

from astropy.io import fits
import copy as cp
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
import numpy as np

SP = np.float32

import xspec as xsp
from ..sixte import keywordList

xsp.Xset.allowNewAttributes = True


def __notice_list_split(notice) -> list:
    """
    Splits a list with notice channels into intervals to be used with the notice command.
    :param notice: (list of int) Notice channels, in increasing order
    :return: (list of tuples) List containing tuples with the starting and endpoint of the intervals
    """
    result = []
    if notice is not None:
        index = 0
        start = notice[index]
        if len(notice) == 1:
            result.append((notice[0], notice[0]))
        else:
            while index < len(notice) - 1:
                index += 1
                while notice[index] == notice[index - 1] + 1 and index < len(notice) - 1:
                    index += 1
                if index != len(notice) - 1:
                    result.append((start, notice[index - 1]))
                    start = notice[index]
                else:
                    if notice[index] == notice[index - 1] + 1:
                        result.append((start, notice[index]))
                    else:
                        result.append((start, notice[index - 1]))
                        result.append((notice[index], notice[index]))

    return result


def pyxspec_reset() -> None:
    """
    Restores the initial settings of PyXspec. Caution: deletes all Spectra and Model instances including attributes of
    SpecFit instances. Deletes also all Chain objects. WARNING: seed is not reinitialized.
    :return: None
    """
    print("Resetting PyXspec initial settings")
    xsp.Xset.chatter = 0
    xsp.Xset.abund = 'angr'
    xsp.Xset.cosmo = '70, 0, 0.7'
    xsp.Xset.closeLog()
    xsp.Xset.logChatter = 10
    xsp.Xset.modelStrings = {}
    xsp.Xset.xsect = 'vern'
    xsp.AllData.clear()  # Removes all Spectrum objects.
    xsp.AllModels.clear()  # Removes all Model objects.
    xsp.AllModels.setEnergies("reset")  # Resets to the PyXspec default
    xsp.AllChains.clear()  # Removes all Chain objects.
    xsp.Xset.chatter = 10
    return None


def __save_xspec_state(self) -> None:
    """
    Saves the state of some global Xspec variables before fitting, ready to be restored after the fit has been
    performed. It is defined as a method of the xspec.Xset singleton, to which it attaches specific attributes to save
    the data.
    :param self: (xspec.XspecSettings) xspec.Xset
    :return: None
    """

    # xspec.AllData
    self.noticeState = []
    for index in range(xsp.AllData.nSpectra):
        self.noticeState.append(xsp.AllData(index + 1).noticed)

    # xspec.AllModels
    self.activeModel = xsp.AllModels.sources[1]

    # xspec.Fit
    self.savedFitMethod = xsp.Fit.method
    self.savedFitNIterations = xsp.Fit.nIterations
    self.savedFitCriticalDelta = xsp.Fit.criticalDelta

    # xspec.Xset (self)
    self.savedAbund = self.abund[0:4]
    self.savedChatter = self.chatter
    self.savedModelStrings = self.modelStrings


xsp.XspecSettings.saveXspecState = __save_xspec_state


def __restore_xspec_state(self) -> None:
    """
    Restores the state of the global Xspec variables previously saved with `xspec.Xse.saveXspecState`, and deletes the
    attributes of `xspec.Xset` created by it.
    :param (xspec.XspecSettings) xspec.Xset
    :return: None
    """

    # xspec.AllData
    for index in range(xsp.AllData.nSpectra):
        intervals_list = __notice_list_split(self.noticeState[index])
        if len(intervals_list) >= 1:
            command_string = str(intervals_list[0][0]) + '-' + str(intervals_list[0][1])
            for i in range(1, len(intervals_list)):
                command_string += ',' + str(intervals_list[i][0]) + '-' + str(intervals_list[i][1])
            xsp.AllData(index + 1).notice(command_string)
    del self.noticeState

    # xspec.AllModels
    xsp.AllModels.setActive(xsp.Xset.activeModel)
    del xsp.Xset.activeModel

    # xspec.Fit
    xsp.Fit.method = self.savedFitMethod
    del self.savedFitMethod
    xsp.Fit.nIterations = self.savedFitNIterations
    del self.savedFitNIterations
    xsp.Fit.criticalDelta = self.savedFitCriticalDelta
    del self.savedFitCriticalDelta

    # xspec.Xset
    self.abund = self.savedAbund
    del self.savedAbund
    self.modelStrings = self.savedModelStrings
    del self.savedModelStrings
    self.chatter = self.savedChatter
    del self.savedChatter


xsp.XspecSettings.restoreXspecState = __restore_xspec_state


def __highlight_spectrum(self, index=1) -> None:
    """
    Highlights a single spectrum to prepare it for the fit by ignoring all channel of all the other spectra
    :param self: (xspec.DataManager) xspec.AllData
    :param index: Index of the spectrum, default 1
    :return: None
    """
    for i in range(1, self.nSpectra + 1):
        if i != index and self(i).noticed != []:
            self(i).ignore("**")


xsp.DataManager.highlightSpectrum = __highlight_spectrum

xsp.ModelManager.nSpecFit = 0


class SpecFit:
    def __init__(self, specFile, model, backFile='USE_DEFAULT', respFile='USE_DEFAULT', arfFile='USE_DEFAULT',
                 setPars=None, header=None, verbose=0):
        __savedChatter = xsp.Xset.chatter
        xsp.Xset.chatter = verbose
        if specFile is not None and os.path.isfile(specFile):
            # Disabling prompting to account for the cases when the response and background files saved in the header
            # are not present
            allow_prompting_ = xsp.Xset.allowPrompting
            xsp.Xset.allowPrompting = False
            self.spectrum = xsp.Spectrum(specFile, backFile=backFile, respFile=respFile, arfFile=arfFile)
            xsp.Xset.allowPrompting = allow_prompting_
            self.keywords = fits.open(specFile)[0].header
            if header:
                for key in header:
                    # The header of the spectrum file prevails in case of duplicates
                    if key not in self.keywords:
                        self.keywords.append(key, header.get(key), header.comments[key])
        else:
            if specFile is None:
                print('Spectrum not loaded (specFile is None)')
            else:
                print('Spectrum not loaded: file ' + str(specFile) + ' not found, spectrum not loaded')
            self.spectrum = None
            self.keywords = cp.deepcopy(header)

        self.model = xsp.Model(model, modName='SpecFit' + str(xsp.AllModels.nSpecFit + 1), setPars=setPars)
        xsp.AllModels.nSpecFit += 1

        # Removing keywords not relevant to the simulation
        if self.keywords is not None:
            keysToDelete = set()
            for key in self.keywords.keys():
                if key not in keywordList + ['SPECFILE', 'ANCRFILE', 'RESPFILE', 'BACKFILE', 'EXPOSURE']:
                    keysToDelete.add(key)
            for key in keysToDelete:
                del self.keywords[key]

        self._isRestored = False
        xsp.Xset.chatter = __savedChatter
        del __savedChatter

    @property
    def restored(self) -> bool:
        return self._isRestored

    @property
    def nParameters(self) -> int:
        """
        Number of model parameters.
        :return: (int) The number of model parameters.
        """
        return self.model.nParameters

    @property
    def parNames(self) -> tuple:
        """
        Model parameter names.
        :return: (tuple) Tuple of strings containing the parameter names.
        """
        return tuple(self.model(self.model.startParIndex + index).name for index in range(self.model.nParameters))

    @property
    def fitDone(self) -> bool:
        """
        Determines if the fit of the model with the spectrum has been run.
        :return: (bool) True if it has been run, False if not
        """
        if self.model is None:
            return hasattr(self, 'fitResult')
        else:
            ind = np.where(self.__get_parfree())[0]
            if len(ind) == 0:
                return False
            else:
                return self.model(self.model.startParIndex + int(ind[0])).sigma > 0

    def __get_parfixed(self) -> tuple:
        """
        Fixed (frozen) status of the model parameters, taken from the model attribute.
        :return: (tuple) Tuple of bool containing True if the parameter is fixed (frozen), False if it is free.
        """
        return tuple(self.model(self.model.startParIndex + index).frozen for index in range(self.model.nParameters))

    def __get_parfree(self) -> tuple:
        """
        Free status of the model parameters, taken from the model attribute.
        :return: (tuple) Tuple of bool containing True if the parameter is free, False if it is fixed (frozen).
        """
        return tuple([not f for f in self.__get_parfixed()])

    def __get_parvals(self) -> tuple:
        """
        Returns a tuple with the model parameter values, taken from the model attribute.
        :return: (tuple) Tuple containing the parameter values.
        """
        return tuple(self.model(self.model.startParIndex + index).values[0] for index in range(self.model.nParameters))

    def __get_parunits(self) -> tuple:
        """
        Returns a tuple with the model parameter units, taken from the model attribute.
        :return: (tuple) Tuple containing the parameter units.
        """
        return tuple(self.model(self.model.startParIndex + index).unit for index in range(self.model.nParameters))

    def __get_sigma(self) -> tuple:
        """
        Returns a tuple with the model parameter errors, taken from the model attribute, overwritten to 0 if the
        parameter is fixed.
        :return: (tuple) Tuple containing the parameter errors.
        """
        return tuple(
            self.model(self.model.startParIndex + index).sigma if self.__get_parfree()[index] else 0
            for index in range(self.model.nParameters)
        )

    @property
    def nFixed(self):
        """
        Number of fixed (frozen) parameters of the model.
        :return: (int) The number of fixed (frozen) parameters, None if the fit has not been run yet.
        """
        if self.fitDone or self.restored:
            result = 0
            for index in range(self.nParameters):
                if self.model(self.model.startParIndex + index).frozen:
                    result += 1
            return result
        else:
            return None

    @property
    def nFree(self):
        """
        Number of free parameters of the model.
        :return: (int) The number of free parameters, None if the fit has not been run yet.
        """
        if self.fitDone or self.restored:
            result = 0
            for index in range(self.nParameters):
                if not self.model(self.model.startParIndex + index).frozen:
                    result += 1
            return result
        else:
            return None

    @property
    def nValid(self):
        """
        Number of valid parameters of the model, i.e. parameters that are free and for which sigma has been evaluated
        without errors.
        :return: (int) The number of valid parameters, None if the fit has not been run yet.
        """
        if self.fitDone or self.restored:
            result = 0
            for index in range(len(self.fitResult.get("parnames"))):
                if self.fitResult["free"][index] and self.fitResult["error_flags"][index][0] == 'F':
                    result += 1
            return result
        else:
            return None

    @property
    def parFixed(self):
        """
        Fixed (frozen) status of the model parameters.
        :return: (tuple) Tuple of bool containing True if the parameter is fixed (frozen), False if it is free. If the
                 fit has not been run, returns None.
        """
        return self.__get_parfixed() if self.fitDone or self.restored else None

    @property
    def parFree(self):
        """
        Free status of the model parameters.
        :return: (tuple) Tuple of bool containing True if the parameter is free, False if it is fixed (frozen). If the
                 fit has not been run, returns None.
        """
        return self.__get_parfree() if self.fitDone or self.restored else None

    @property
    def parValid(self):
        """
        Valid status of the model parameters.
        :return: (tuple) Tuple of bool containing True if the parameter is free, False if it is fixed (frozen). If the
                 fit has not been run, returns None.
        """
        if self.fitDone or self.restored:
            return tuple([self.fitResult["free"][index] and self.fitResult["error_flags"][index][0] == 'F'
                          for index in range(len(self.fitResult.get("parnames")))])
        else:
            return None

    @property
    def fixedParNames(self):
        """
        Fixed (frozen) parameters' names.
        :return: (tuple) Tuple of strings containing the fixed (frozen) parameters' names, None if the fit has not
        been run.
        """
        if self.fitDone or self.restored:
            return tuple([name for i, name in enumerate(self.parNames) if self.parFixed[i]])
        else:
            return None

    @property
    def freeParNames(self):
        """
        Free parameters' names.
        :return: (tuple) Tuple of strings containing the free parameters' names, None if the fit has not
        been run.
        """
        if self.fitDone or self.restored:
            return tuple([name for i, name in enumerate(self.parNames) if self.parFree[i]])
        else:
            return None

    @property
    def validParNames(self):
        """
        Valid parameters' names.
        :return: (tuple) Tuple of strings containing the valid parameters' names, None if the fit has not
        been run.
        """
        if self.fitDone or self.restored:
            return tuple([name for i, name in enumerate(self.fitResult["parnames"]) if self.parValid[i]])
        else:
            return None

    def __get_counts(self):
        return np.asarray(self.spectrum.values, dtype=SP) * self.spectrum.exposure  # [---]

    def __get_energy(self):
        return 0.5 * (np.asarray(self.spectrum.energies, dtype=SP)[:, 0] +
                      np.asarray(self.spectrum.energies, dtype=SP)[:, 1])  # [keV]

    def __get_denergy(self):
        return (np.asarray(self.spectrum.energies, dtype=SP)[:, 1] -
                np.asarray(self.spectrum.energies, dtype=SP)[:, 0])  # [keV]

    def __get_spectrum(self):
        return np.asarray(self.spectrum.values, dtype=SP) / self.__get_denergy()  # cts/s/keV [keV]

    def __set_energy_range(self, erange=(None, None)) -> None:
        """
        Sets the energy range of a spectrum for the fit.
        :param erange: (float 2, or int 2, or str 2) Energy range.
        :return: None
        """

        def __ignore_string(x) -> str:
            """
            Turns a variable containing a number into a string that contains a dot ('.') for floating points.
            This is mandatory for the ignore command of Xspec as otherwise it would consider channel instead of
            energy.
            :param x: Energy value (either float, int or str).
            :return: (str) The sting suited for the ignore command.
            """

            result = str(x)
            if "." not in result:
                result += "."
            return result

        if len(erange) >= 2:
            if erange[0] is not None and erange[1] is not None:
                if erange[1] <= erange[0]:
                    raise ValueError("Invalid input:", erange, "Second argument must be larger than the first")

        self.spectrum.notice("**")

        # Setting lower energy limit
        if erange[0] is not None:
            self.spectrum.ignore("**-" + __ignore_string(erange[0]))
        # Setting lower energy limit
        if erange[1] is not None:
            self.spectrum.ignore(__ignore_string(erange[1]) + "-**")

        return None

    def __perform(self, error=False) -> None:
        """
        Equivalent of the `xspec.Fit.perform` method adapted to the `SpecFit` class. It allows to run the fit of the
        `xspec.Spectrum` loaded in the `spectrum` attribute with the `xspec.Model` of the instance and saves the fit
        results and the data points in the fitResult and fitPoints attributes.
        :param error: (bool) If set to `True` after performing the fit it will run `error` on the free parameters, and
            save the error flags in the `fitResult` attribute. Default `False`.
        :return: None
        """

        xsp.Fit.perform()

        if error:
            for index in range(self.model.nParameters):
                if self.__get_parfree()[index]:
                    xsp.Fit.error(str(self.model.startParIndex + index))

        def get_errflags() -> tuple:
            """
            Return an error flag 'F' or 'T' for each free parameter if its error computed by the fit is > 0 or not,
            respectively. It adds nine '-' to allow room for error flags that may be computed later with the error
            procedure.
            Returns the 10 error flags of the parameters. The first one, labeled 0 checks that the error computed by the
            fit is > 0. The other nine, labeled 1-9 are the ones defined by the Xspec error command (see
            https://heasarc.gsfc.nasa.gov/docs/xanadu/xspec/manual/node79.html) and are present only if the `error`
            argument is set to True, otherwise they are all set to '-', i.e. not calculated.
            :return: (tuple) Tuple containing the error flags.
            """
            flagList = []
            for index in range(self.model.nParameters):
                if self.__get_parfree()[index]:
                    flagList.append(('F' if self.__get_sigma()[index] > 0 else 'T') + '---------')
                else:
                    flagList.append('')

            return tuple(flagList)

        # Saving fit results
        self.fitResult = {
            "parnames": self.parNames,
            "units": self.__get_parunits(),
            "values": self.__get_parvals(),
            "free": self.__get_parfree(),
            "sigma": self.__get_sigma(),
            "error_flags": get_errflags(),
            "statistic": xsp.Fit.statistic,
            "dof": xsp.Fit.dof,
            "rstat": xsp.Fit.statistic / (xsp.Fit.dof - 1),
            "covariance": xsp.Fit.covariance,
            "method": xsp.Fit.statMethod,
            "nIterations": xsp.Fit.nIterations,
            "criticalDelta": xsp.Fit.criticalDelta,
            "abund": xsp.Xset.abund[0:4],
            "apecroot": [item[1] for item in xsp.Xset.modelStrings if item[0] == 'APECROOT'][0]
        }

        # Saving the data of the fit points
        self.fitPoints = {
            "energy": self.__get_energy(),  # [keV]
            "spectrum": self.__get_spectrum(),  # cts/s/keV [keV]
            "sigma": np.divide(self.__get_spectrum(), np.sqrt(self.__get_counts()),
                               out=np.zeros_like(self.__get_spectrum(), dtype=SP), where=self.__get_counts() > 0),
            "model": (np.asarray(self.model.folded(self.spectrum.index), dtype=SP) / self.__get_denergy()),
            "counts": self.__get_counts(),  # [---]
            "dEne": self.__get_denergy(),  # [keV]
            "noticed": np.asarray(self.spectrum.noticed, dtype=np.int32)
        }

        return None

    def clear(self, attr=None) -> None:
        """
        Deletes the spectrum and model attributes and the corresponding objects from the xspec global variables.
        :param attr: (str) The name of the attribute to delete, can be either 'spectrum' or 'model'. Default None, i.e.
            deletes both.
        :return: None
        """
        del_spectrum, del_model = True, True
        if type(attr) is str:
            attr_ = attr.lower().strip()
            if 'spectrum'.startswith(attr_):
                del_spectrum, del_model = True, False
            elif 'model'.startswith(attr_):
                del_spectrum, del_model = False, True
            else:
                raise ValueError("Invalid attribute name " + attr + ". Must be either 'spectrum' or 'model'.")

        if del_spectrum and self.spectrum is not None:
            xsp.AllData -= self.spectrum.index
            self.spectrum = None
        if del_model and self.model is not None:
            xsp.AllModels -= self.model.name
            self.model = None

    def covariance_matrix(self):
        """
        Returns the covariance matrix of the fit.
        :return: (2D array) The covariance matrix
        """
        if not self.fitDone and not self.restored:
            print("No data available, the fit has not been run yet.")
            return None
        else:
            # Creating covariance matrix
            result = np.ndarray([self.nValid, self.nValid])

            # Filling diagonal and lower part
            index = 0
            for i in range(self.nValid):
                for j in range(i + 1):
                    result[i, j] = self.fitResult["covariance"][index]
                    index += 1
            # Filling upper part
            for i in range(self.nValid):
                for j in range(i + 1, self.nValid):
                    result[i, j] = result[j, i]

            return result

    def correlation_matrix(self):
        """
        Returns the correlation matrix of the fit.
        :return: (2D array) The correlation matrix
        """
        result = self.covariance_matrix()
        if result is not None:
            valid_inds = np.where(self.parValid)[0]
            for i, j in np.ndindex(result.shape):
                result[i, j] /= self.fitResult["sigma"][valid_inds[i]] * self.fitResult["sigma"][valid_inds[j]]

        return result

    def run(self, erange=(None, None), start=None, fixed=None, method="chi", niterations=100, criticaldelta=1.e-3,
            abund='angr', verbose=0, apecroot=None, apecthermal=True, error=False) -> None:
        """
        Standard procedure to fit spectra.
        :param erange: (float, float) Energy range [keV]. If the first (second) elements is None the lower (higher)
            energy limit is not set. Default (None, None), i.e. all energy channels are considered.
        :param start: (float n) Starting parameters for the fit. The size depends on the model.
        :param fixed: (bool n) Indicates whether a parameter is fixed (True) or free (False). Default all False.
        :param method: (str) Fitting method, can be 'chi' or 'cstat'. Default 'chi'.
        :param niterations: (int) Number of iterations. Default 100.
        :param criticaldelta: (float) The absolute change in the fit statistic between iterations, less than which the
            fit is deemed to have converged.
        :param abund: (str) Abundance table, see `abund` command in Xspec. Default 'angr' i.e. Anders & Grevesse (1989).
        :param verbose: (int) Verbosity level. Default 0.
        :param apecroot: (str or tuple) Root table for Apec version. Default None, i.e. latest version of Apec tables.
        :param apecthermal: (bool) If set to True thermal broadening for Apec turned on. Default True.
        :param error: (bool) If set to True the error procedure is run over all the parameters. The fitResult attribute
         will also containg the error flags defined by the Xspec error command (see
         https://heasarc.gsfc.nasa.gov/docs/xanadu/xspec/manual/node79.html). Default False.
        """

        if self.spectrum is None:
            print('Spectrum not loaded: no data to fit.')
            return None

        else:
            if apecroot is not None and apecroot != "latest":
                apecroot_ = str(apecroot).strip('(').strip(')').replace(', ', '.')  # works also with tuple
            else:
                # Finding latest version of apec tables
                folder = os.environ.get("HEADAS") + "/../spectral/modelData/"
                file_beg, file_end = "apec_v", "_coco.fits"
                version_list = [file.strip(file_beg).strip(file_end) for file in os.listdir(folder)
                                if file.startswith(file_beg) and file.endswith(file_end)]
                version_list.sort()
                apecroot_ = version_list[-1]

            xsp.Xset.saveXspecState()
            xsp.Xset.chatter = verbose
            xsp.Xset.abund = abund
            xsp.Xset.addModelString("APECROOT", apecroot_)
            xsp.Xset.addModelString("APECTHERMAL", "yes" if apecthermal else "no")
            xsp.Xset.addModelString("APEC_TRACE_ABUND", "Fe")
            xsp.AllData.highlightSpectrum(self.spectrum.index)
            xsp.AllModels.setActive(self.model.name)

            # Energy range
            self.__set_energy_range(erange)

            # Initial conditions
            if start is not None:
                for index, par in enumerate(start):
                    self.model(self.model.startParIndex + index).values = par

            # Fixed/free parameter
            if fixed is not None:
                for index, frozen in enumerate(fixed):
                    self.model(self.model.startParIndex + index).frozen = frozen
            else:
                for index in range(self.model.nParameters):
                    self.model(self.model.startParIndex + index).frozen = False

            # Statistic method
            if method is not None:
                xsp.Fit.statMethod = method

            # Number of iterations
            if niterations is not None:
                xsp.Fit.nIterations = niterations

            # Critical delta
            if criticaldelta is not None:
                xsp.Fit.criticalDelta = criticaldelta

            # Fitting
            self.__perform(error=error)

            # Running error on free parameters TODO: check why it does not work
            if error:
                for index in range(self.model.nParameters):
                    if self.__get_parfree()[index]:
                        xsp.Fit.error(str(self.model.startParIndex + index))
                        self.fitResult["error_flags"][index][1:] = self.model(self.model.startParIndex + index).error[2]

            # Restoring the Xspec state
            xsp.Xset.restoreXspecState()

            return None

    def plot(self, rebin=None, xscale='lin', yscale='lin', xlim=None, ylim=None, nsample=1) -> None:
        """
        Plots the spectrum data with errorbars, along with the best fit model and the residuals.
        :param rebin: (2 x float) Combining adjacent bins for higher significance: 1st value - sigma significance, 2nd
               value (may not be present) - maximum number of bins to reach the significance. Default None, i.e. no
               rebinning. Equivalent of `setplot rebin` command in Xspec.
        :param xscale: (str) Scaling of the x-axis, can be either 'lin'/'linear' or 'log'/'logarithmic'. Default 'lin'.
        :param yscale: (str) Same as xscale but for the y-axis.
        :param xlim: (2 x float) Lower and upper limits of the x-axis.
        :param ylim: (2 x float) Same as xscale but for the y-axis.
        :param nsample: (int) If set it defines a sampling for the data points, for better visualization. Not
               considered if rebin is present. Default 1, i.e. all points are shown.
        """

        if not self.fitDone and not self.restored:
            print("No data available, the fit has not been run yet.")
        else:
            fig, (axd, axr) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [5, 2], 'hspace': 0})

            xscale_ = xscale.lower().strip()
            if xscale_ in ['lin', 'linear']:
                axd.set_xscale('linear')
                axr.set_xscale('linear')
            elif xscale_ in ['log', 'logarithmic']:
                axd.set_xscale('log')
                axr.set_xscale('log')
            else:
                raise ValueError(
                    "Invalid input type for xscale, must be one of 'lin' ('linear') or 'log' ('logarithmic')")

            if xlim is not None:
                axd.set_xlim(xlim)
                axr.set_xlim(xlim)

            if ylim is not None:
                axd.set_ylim(ylim)

            yscale_ = yscale.lower().strip()
            if yscale_ in ['lin', 'linear']:
                axd.set_yscale('linear')
            elif yscale_ in ['log', 'logarithmic']:
                axd.set_yscale('log')
            else:
                raise ValueError(
                    "Invalid input type for yscale, must be one of 'lin' ('linear') or 'log' ('logarithmic')")
            axd.set_ylabel("counts s$^{-1}$ keV$^{-1}$")
            axr.set_xlabel("Energy (keV)")
            axr.set_ylabel("Diff.")

            # Creating values to plot (x, y, y_error, residuals)
            if rebin is None:
                x = self.fitPoints["energy"][::nsample]  # [keV]
                y = self.fitPoints["spectrum"][::nsample]  # [cts/s/keV]
                y_error = self.fitPoints["sigma"][::nsample]  # [cts/s/keV]
                residuals = self.fitPoints["spectrum"][::nsample] - self.fitPoints["model"][::nsample]  # [cts/s/keV]
            else:
                exposure_ = self.spectrum.exposure if hasattr(self.spectrum, 'exposure') else self.keywords.get(
                    'EXPOSURE')
                if exposure_ is None:
                    print('Exposure not present: no rebinning possible')
                    return None

                larr = len(self.fitPoints["energy"])
                try:
                    larg = len(rebin)
                    cts_to_reach = rebin[0] ** 2
                    if larg == 1:
                        nmax = larr
                    else:
                        nmax = rebin[1]
                except:
                    cts_to_reach = rebin ** 2
                    nmax = larr

                x, y, y_error, residuals = [], [], [], []
                istart = 0
                while istart < larr:
                    n = 1
                    c = self.fitPoints["counts"][istart]
                    while c < cts_to_reach and n < nmax and (istart + n) < larr:
                        n += 1
                        c += self.fitPoints["counts"][istart + n - 1]
                    # Width of tbe bin [keV]
                    delta_ene = self.fitPoints["energy"][istart + n - 1] + 0.5 * self.fitPoints["dEne"][
                        istart + n - 1] - self.fitPoints["energy"][istart] + 0.5 * self.fitPoints["dEne"][istart]
                    # Center of the bin [keV]
                    x.append(self.fitPoints["energy"][istart] - 0.5 * self.fitPoints["dEne"][istart] + 0.5 * delta_ene)
                    y_ = c / (exposure_ * delta_ene)  # [cts/s/keV]
                    y.append(y_)  # [cts/s/keV]
                    y_error.append(np.sqrt(c) / (exposure_ * delta_ene))  # [cts/s/keV]
                    residuals.append(
                        y_ - np.sum((self.fitPoints["model"] * self.fitPoints["dEne"])[istart:istart + n]) / delta_ene)
                    istart = istart + n

                x = np.asarray(x, dtype=np.float32)  # [keV]
                y = np.asarray(y, dtype=np.float32)  # [cts/s/keV]
                y_error = np.asarray(y_error, dtype=np.float32)  # [cts/s/keV]
                residuals = np.asarray(residuals, dtype=np.float32)  # [cts/s/keV]

            axd.errorbar(x, y, yerr=y_error, color='black', linestyle='', fmt='.', zorder=0)
            axd.plot(self.fitPoints["energy"], self.fitPoints["model"], color="limegreen", zorder=1)
            axr.errorbar(x, residuals, yerr=y_error, color='black', linestyle='', fmt='.', zorder=0)
            axr.plot((self.fitPoints["energy"][0], self.fitPoints["energy"][-1]), (0, 0), color="limegreen", zorder=1)

        return None

    def show_correlation(self) -> None:
        """
        Plots a heatmap of the correlation matrix
        :return: None
        """
        corr_matrix = self.correlation_matrix()

        # Plotting
        if corr_matrix is not None:
            fig, ax = plt.subplots()
            plt.subplots_adjust(bottom=0.15)
            ax.set_xticks(range(self.nValid), labels=self.validParNames, rotation=45, ha="center", va="top",
                          rotation_mode="default")
            ax.set_yticks(range(self.nValid), labels=self.validParNames)
            ax.tick_params(axis='both', which='major', length=0, pad=10)
            ax.tick_params(axis='both', which='minor', length=5)
            ax.imshow(corr_matrix, cmap=cm["bwr"], aspect='equal', vmin=-1, vmax=1)
            # Drawing grid lines requires some tweaks due to a bug
            # (see https://github.com/matplotlib/matplotlib/issues/12934)
            minor_ticks = np.arange(self.nValid + 1) - 0.51
            minor_ticks[-1] += 0.01
            ax.set_xticks(minor_ticks, minor=True, labels=np.full(self.nValid + 1, None))
            ax.set_yticks(minor_ticks, minor=True, labels=np.full(self.nValid + 1, None))
            ax.grid(which='minor', color='black', linewidth=1)

            # Text annotations
            for i, j in np.ndindex(corr_matrix.shape):
                ax.text(j, i, "{:.3f}".format(corr_matrix[i, j]), ha="center", va="center", color="black")

    def show(self) -> None:
        """
        Prints the fit results in readable way
        :return: None
        """

        def __val2str(val: float, just='l') -> str:
            if val < 0.1 or val > 100:
                return "{:.3E}".format(val)
            else:
                if just == 'l':
                    return "{:6.3F}".format(val).strip().ljust(9)
                else:
                    return "{:6.3F}".format(val).rjust(9)

        def __par_string(ipar: int) -> str:
            par_string = self.fitResult["parnames"][ipar].rjust(8) + "  "
            par_string += str(self.fitResult["units"][ipar]).ljust(6) + " "
            par_string += __val2str(self.fitResult["values"][ipar], 'r') + " "
            if self.fitResult["free"][ipar]:
                par_string += "± " + __val2str(self.fitResult["sigma"][ipar])
            else:
                par_string += "(fixed)    "

            # Error flags are shown only if at least one error is present
            errfl = str(self.fitResult["error_flags"][ipar])
            if 'T' in errfl:
                par_string += "  "
                for i, ich in enumerate(errfl):
                    if ich == 'T':
                        par_string += "\033[91m" + str(i) + "\033[0m"  # red number -> error
                    elif ich == 'F':
                        par_string += "\033[92m\u2588\033[0m"  # green block -> no error
                    else:
                        par_string += "\033[97m\u2588\033[0m"  # grey block -> error flag not present)
            return par_string

        if not self.fitDone and not self.restored:
            print("Nothing to show, the fit has not been run yet.")
            return None
        else:
            for ipar in range(len(self.fitResult.get("parnames"))):
                print(__par_string(ipar))
        print("")
        print(str(self.fitResult["method"]) + " = " + __val2str(self.fitResult["statistic"]))
        print("D.o.f. = " + str(self.fitResult["dof"]))
        print("Red. " + self.fitResult["method"] + " = " + __val2str(self.fitResult["rstat"]))

    def save(self, fileName: str, overwrite=True):
        if not self.fitDone and not self.restored:
            print("Cannot save, the fit has not been run yet.")
            return None
        elif self.spectrum is None:
            print("Cannot save, the spectrum attribute has been cleared.")
            return None
        elif self.model is None:
            print("Cannot save, the model attribute has been cleared.")
            return None
        else:
            # Creating the FITS file
            hdulist = fits.HDUList()

            # Primary (empty)
            hdulist.append(fits.PrimaryHDU([0]))

            # Files related to the spectrum to save in the header
            sp_file = self.spectrum.fileName if hasattr(self.spectrum, 'fileName') else None
            arf_file, rmf_file, bkg_file = None, None, None
            if hasattr(self.spectrum, 'response'):
                try:
                    arf_file = self.spectrum.response.arf
                except:
                    pass
                try:
                    rmf_file = self.spectrum.response.rmf
                except:
                    pass
                try:
                    bkg_file = self.spectrum.response.background
                except:
                    pass

            # Exposure to save in the header
            exposure = self.spectrum.exposure if hasattr(self.spectrum, 'exposure') else None  # [s]

            def header_set_special(k, v, c=None):
                """
                Sets a value in the header if not None, or else tries to find it in self.keywords
                """
                if v is not None:
                    hdulist[0].header.set(k, v, c)
                else:
                    if k in self.keywords:
                        hdulist[0].header.set(k, self.keywords.get(k), self.keywords.comments[k])

            header_set_special('SPECFILE', sp_file)
            header_set_special('ANCRFILE', arf_file)
            header_set_special('RESPFILE', rmf_file)
            header_set_special('BACKFILE', bkg_file)
            header_set_special('EXPOSURE', exposure, 'Exposure time (s)')

            # Model to save in the header (many components may be present)
            hdulist[0].header.set('MODEL', '*'.join(self.model.componentNames))

            # Fit results to save in the header
            hdulist[0].header.set('STAT', self.fitResult["statistic"])
            hdulist[0].header.set('DOF', self.fitResult["dof"])
            hdulist[0].header.set('RSTAT', self.fitResult["rstat"])
            hdulist[0].header.set('METHOD', self.fitResult["method"])
            hdulist[0].header.set('N_ITER', self.fitResult["nIterations"])
            hdulist[0].header.set('CR_DELTA', self.fitResult["criticalDelta"])
            hdulist[0].header.set('ABUND', self.fitResult["abund"])
            hdulist[0].header.set('APECROOT', self.fitResult["apecroot"])

            # Inheriting keywords related to the simulation
            for key in keywordList:
                if key in self.keywords:
                    hdulist[0].header.set(key, self.keywords.get(key), self.keywords.comments[key])

            # Adding table with fit results
            fit_results_columns = [
                fits.Column(name='PARNAME', format='8A', array=self.fitResult["parnames"]),
                fits.Column(name='UNITS', format='6A', array=self.fitResult["units"]),
                fits.Column(name='VALUES', format='E', array=self.fitResult["values"]),
                fits.Column(name='FREE', format='L', array=self.fitResult["free"]),
                fits.Column(name='SIGMA', format='E', array=self.fitResult["sigma"]),
                fits.Column(name='ERRFLAGS', format='10A', array=self.fitResult["error_flags"])
            ]
            hdulist.append(fits.BinTableHDU.from_columns(fits.ColDefs(fit_results_columns), name="Results"))

            # Adding covariance matrix data
            hdulist.append(fits.ImageHDU(self.fitResult["covariance"], name='Covariance'))

            # Adding table with fit points
            fit_points_columns = [
                fits.Column(name='ENERGY', format='E', array=self.fitPoints["energy"], unit='keV'),
                fits.Column(name='D_ENERGY', format='E', array=self.fitPoints["dEne"], unit='keV'),
                fits.Column(name='SPECTRUM', format='E', array=self.fitPoints["spectrum"], unit='cts/s/keV'),
                fits.Column(name='SIGMA', format='E', array=self.fitPoints["sigma"], unit='cts/s/keV'),
                fits.Column(name='MODEL', format='E', array=self.fitPoints["model"], unit='cts/s/keV'),
                fits.Column(name='COUNTS', format='E', array=self.fitPoints["counts"], unit='---'),
                fits.Column(name='NOTICED', format='J', array=self.fitPoints["noticed"], unit='---')
            ]
            hdulist.append(fits.BinTableHDU.from_columns(fits.ColDefs(fit_points_columns), name='Points'))

            # Writing FITS file
            return hdulist.writeto(fileName, overwrite=overwrite)


def restore(file: str, path=None, quick=False, verbose=0) -> SpecFit:
    """
    Restores a SpecFit object previously saved in a file with the save method.
    :param file: (str) The saved file.
    :param path: (str or list/tuple of str) Path where to look for files to allow a complete restore of the SpcFit
    object: if set it will look for spectrum, arf, rmf and background files present in the file header also in the
    folders indicated, in order.
    :param quick: (bool) If set to True the spectrum is not loaded, useful for checking the results. Default False.
    :return: (SpecFit) The restored SpecFit object.
    """

    __savedChatter = xsp.Xset.chatter
    xsp.Xset.chatter = verbose

    def __path_search(file: str, path):
        baseName = os.path.basename(file)
        if path is not None:
            path_ = [path] if isinstance(path, str) else path
            i = 0
            found = os.path.isfile(path[i] + '/' + baseName)
            while not found and i < len(path_) - 1:
                i += 1
                found = os.path.isfile(path[i] + '/' + baseName)
            if found:
                return path[i] + '/' + baseName
            else:
                return None
        else:
            return None

    hdulist = fits.open(file)
    h0 = hdulist[0].header
    specFile = h0.get('SPECFILE')
    if not os.path.isfile(specFile):
        specFile = __path_search(h0.get('SPECFILE'), path)
    ancrFile = h0.get('ANCRFILE')
    if ancrFile is not None and not os.path.isfile(ancrFile):
        ancrFile = __path_search(ancrFile, path)
    respFile = h0.get('RESPFILE')
    if respFile is not None and not os.path.isfile(respFile):
        respFile = __path_search(respFile, path)
    backFile = h0.get('BACKFILE')
    if backFile is not None and not os.path.isfile(backFile):
        backFile = __path_search(backFile, path)
    model = h0.get('MODEL')
    values = hdulist[1].data['VALUES']

    # Initialization
    result = SpecFit(None if quick else specFile, model, backFile=backFile, respFile=respFile, arfFile=ancrFile,
                     setPars=tuple(np.float64(values)), header=h0)

    # Flagging the output as restored
    result._isRestored = True

    # Setting fitResult
    d1 = hdulist[1].data
    free_pars = tuple(d1['FREE'])
    result.fitResult = {
        "parnames": tuple(d1['PARNAME']),
        "units": tuple(d1['UNITS']),
        "values": tuple(d1['VALUES']),
        "free": free_pars,
        "sigma": tuple(d1['SIGMA']),
        "error_flags": tuple(d1['ERRFLAGS']),
        "statistic": h0.get('STAT'),
        "dof": h0.get('DOF'),
        "rstat": h0.get('RSTAT'),
        "covariance": tuple(hdulist[2].data),
        "method": h0.get('METHOD'),
        "nIterations": h0.get('N_ITER'),
        "criticalDelta": h0.get('CR_DELTA'),
        "abund": h0.get('ABUND'),
        "apecroot": h0.get('APECROOT')
    }

    # Setting free/fixed parameters in model attributes (will make fitDone property work correctly)
    for index in range(result.model.nParameters):
        result.model(result.model.startParIndex + index).frozen = not free_pars[index]

    # Setting fitPoints
    d3 = hdulist[3].data
    result.fitPoints = {
        "energy": d3['ENERGY'],  # [keV]
        "spectrum": d3['SPECTRUM'],  # cts/s/keV [keV]
        "sigma": d3['SIGMA'],
        "model": d3['MODEL'],
        "counts": d3['COUNTS'],  # [---]
        "dEne": d3['D_ENERGY'],  # [keV]
        "noticed": d3['NOTICED']
    }

    xsp.Xset.chatter = __savedChatter
    del __savedChatter

    return result
