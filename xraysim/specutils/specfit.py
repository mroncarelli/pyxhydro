import os
import sys

sys.path.append(os.environ.get("HEADAS") + "/lib/python")
# TODO: the three lines above are necessary only to make the code work in IntelliJ (useful for debugging)

from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
import numpy as np

sp = np.float32

import xspec as xsp
from ..sixte import keywordList

xsp.Xset.allowNewAttributes = True
xsp.Xset.chatter = 0
xsp.Xset.addModelString("APECROOT", "3.0.9")
xsp.Xset.addModelString("APECTHERMAL", "yes")


def __notice_list_split(notice) -> list:
    """
    Splits a list with notice channels into intervals to be used with the notice command.
    :param notice: (list of int) Notice channels, in increasing order
    :return: (list of tuples) List containg tuples with the starting and endpoint of the intervals
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


def __save_xspec_state(self) -> None:
    """
    Saves the state of some global Xspec variables before fitting, ready to be restored after the fit has been
    performed. In detail, it saves the notice arrays in the xspec.AllData object by creating the `noticeState`
    attribute, the name of the active Model in the xspec.AllModels object by creating the `activeModel` attribute, and
    the abundance table by creating the `abundTable` attribute.
    :param self: (xspec.XspecSettings) xspec.Xset
    :return: None
    """
    self.noticeState = []
    for index in range(xsp.AllData.nSpectra):
        self.noticeState.append(xsp.AllData(index + 1).noticed)

    self.activeModel = xsp.AllModels.sources[1]
    self.abundTable = xsp.Xset.abund[0:4]


xsp.XspecSettings.saveXspecState = __save_xspec_state


def __restore_xspec_state(self) -> None:
    """
    Restores the state of the notice arrays in `xspec.AllData` object and of the active Model in `xspec.AllModels`.
    Deletes the `noticeState`, `activeModel` and `abundTable` attributes from `xspec.Xset` after.
    :param (xspec.XspecSettings) xspec.Xset
    :return: None
    """
    for index in range(xsp.AllData.nSpectra):
        intervals_list = __notice_list_split(self.noticeState[index])
        if len(intervals_list) >= 1:
            command_string = str(intervals_list[0][0]) + '-' + str(intervals_list[0][1])
            for i in range(1, len(intervals_list)):
                command_string += ',' + str(intervals_list[i][0]) + '-' + str(intervals_list[i][1])
            xsp.AllData(index + 1).notice(command_string)

    del self.noticeState

    xsp.AllModels.setActive(xsp.Xset.activeModel)
    del xsp.Xset.activeModel
    xsp.Xset.abund = xsp.Xset.abundTable
    del xsp.Xset.abundTable


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


class SpecFit:
    def __init__(self, spectrum, model, bkg='USE_DEFAULT', rmf='USE_DEFAULT', arf='USE_DEFAULT', setPars=None):
        self.spectrum = xsp.Spectrum(spectrum, backFile=bkg, respFile=rmf, arfFile=arf)
        self.keywords = fits.open(spectrum)[0].header
        # Removing keywords not relevant to the simulation
        keysToDelete = set()
        for key in self.keywords.keys():
            if key not in keywordList + ['SP_FILE', 'ARF_FILE', 'RMF_FILE', 'BKG_FILE', 'MODEL']:
                keysToDelete.add(key)
        for key in keysToDelete:
            del self.keywords[key]
        self.model = xsp.Model(model, modName='SpecFit' + str(self.spectrum.index), setPars=setPars)

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
        ind = np.where(self.__get_parfree())[0]
        if len(ind) == 0:
            return False
        else:
            return self.model(self.model.startParIndex + int(ind[0])).sigma > 0

    def __get_parfixed(self):
        """
        Fixed (frozen) status of the model parameters, taken from the model attribute.
        :return: (tuple) Tuple of bool containing True if the parameter is fixed (frozen), False if it is free.
        """
        return tuple(self.model(self.model.startParIndex + index).frozen for index in range(self.model.nParameters))

    def __get_parfree(self):
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

    def __get_errors(self) -> tuple:
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
        if self.fitDone:
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
        if self.fitDone:
            result = 0
            for index in range(self.nParameters):
                if not self.model(self.model.startParIndex + index).frozen:
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
        return self.__get_parfixed() if self.fitDone else None

    @property
    def parFree(self):
        """
        Free status of the model parameters.
        :return: (tuple) Tuple of bool containing True if the parameter is free, False if it is fixed (frozen). If the
                 fit has not been run, returns None.
        """
        return self.__get_parfree() if self.fitDone else None

    @property
    def fixedParNames(self):
        """
        Fixed (frozen) parameters' names.
        :return: (tuple) Tuple of strings containing the fixed (frozen) parameters' names, None if the fit has not
        been run.
        """
        # return tuple([self.parNames[i] for i in range(self.nParameters) if self.__get_parfixed()[i]])
        return tuple([name for i, name in enumerate(self.parNames) if self.parFixed[i]]) if self.fitDone else None

    @property
    def freeParNames(self):
        """
        Free parameters' names.
        :return: (tuple) Tuple of strings containing the free parameters' names, None if the fit has not
        been run.
        """
        return tuple([name for i, name in enumerate(self.parNames) if self.parFree[i]]) if self.fitDone else None

    def __get_counts(self):
        return np.asarray(self.spectrum.values, dtype=sp) * self.spectrum.exposure  # [---]

    def get_energy(self):
        return 0.5 * (np.asarray(self.spectrum.energies, dtype=sp)[:, 0] +
                      np.asarray(self.spectrum.energies, dtype=sp)[:, 1])  # [keV]

    def __get_denergy(self):
        return (np.asarray(self.spectrum.energies, dtype=sp)[:, 1] -
                np.asarray(self.spectrum.energies, dtype=sp)[:, 0])  # [keV]

    def __get_spectrum(self):
        return np.asarray(self.spectrum.values, dtype=sp) / self.__get_denergy()  # cts/s/keV [keV]

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
                    print("ERROR in ignore: invalid input:", erange, "Second argument must be larger than the first")
                    raise ValueError

        self.spectrum.notice("**")

        # Setting lower energy limit
        if erange[0] is not None:
            self.spectrum.ignore("**-" + __ignore_string(erange[0]))
        # Setting lower energy limit
        if erange[1] is not None:
            self.spectrum.ignore(__ignore_string(erange[1]) + "-**")

        return None

    def __perform(self, abund='angr') -> None:
        """
        Equivalent of the `xspec.Fit.perform` method adapted to the `SpecFit` class. It allows to run the fit of the
        `xspec.Spectrum` loaded in the `spectrum` attribute with the `xspec.Model` of the instance while preserving the
        state of the `xspec` global objects (i.e. `xspec.AllData` and `xspec.AllModels`). It also saves the fit results
        and the data points in the fitResult and fitPoints attributes.
        :param abund: (str) Abundance table, see `abund` command in Xspec. Default 'angr' i.e. Anders & Grevesse (1989).
        :return: None
        """
        xsp.Xset.saveXspecState()
        xsp.Xset.abund = abund
        xsp.AllData.highlightSpectrum(self.spectrum.index)
        xsp.AllModels.setActive(self.model.name)
        xsp.Fit.perform()

        # Saving fit results
        self.fitResult = {
            "parnames": self.parNames,
            "values": self.__get_parvals(),
            "sigma": self.__get_errors(),
            "statistic": xsp.Fit.statistic,
            "dof": xsp.Fit.dof,
            "rstat": xsp.Fit.statistic / (xsp.Fit.dof - 1),
            "covariance": xsp.Fit.covariance,
            "method": xsp.Fit.statMethod,
            "nIterations": xsp.Fit.nIterations,
            "criticalDelta": xsp.Fit.criticalDelta,
            "abund": abund
        }

        # Saving the data of the fit points

        self.fitPoints = {
            "energy": 0.5 * (np.asarray(self.spectrum.energies)[:, 0] + np.asarray(self.spectrum.energies)[:, 1]),
            # [keV]
            "spectrum": self.__get_spectrum(),  # cts/s/keV [keV]
            "error": np.divide(self.__get_spectrum(), np.sqrt(self.__get_counts()),
                               out=np.zeros_like(self.__get_spectrum(), dtype=sp), where=self.__get_counts() > 0),
            "model": (np.asarray(self.model.folded(self.spectrum.index), dtype=sp) / self.__get_denergy()),
            "counts": self.__get_counts(),  # [---]
            "dEne": self.__get_denergy(),  # [keV]
            "noticed": np.asarray(self.spectrum.noticed, dtype=np.int32)
        }

        xsp.Xset.restoreXspecState()

    def covariance_matrix(self):
        """
        Returns the covariance matrix of the fit.
        :return: (2D array) The covariance matrix
        """

        if not self.fitDone:
            print("No data available, the fit has not been run yet.")
            return None
        else:
            # Creating covariance matrix
            result = np.ndarray([self.nFree, self.nFree])

            # Filling diagonal and lower part
            index = 0
            for i in range(self.nFree):
                for j in range(i + 1):
                    result[i, j] = self.fitResult["covariance"][index]
                    index += 1
            # Filling upper part
            for i in range(self.nFree):
                for j in range(i + 1, self.nFree):
                    result[i, j] = result[j, i]

            return result

    def correlation_matrix(self):
        """
        Returns the correlation matrix of the fit.
        :return: (2D array) The correlation matrix
        """
        result = self.covariance_matrix()
        if result is not None:
            free_inds = np.where(self.parFree)[0]
            for i, j in np.ndindex(result.shape):
                result[i, j] /= self.fitResult["sigma"][free_inds[i]] * self.fitResult["sigma"][free_inds[j]]

        return result

    def run(self, erange=(None, None), start=None, fixed=None, method="chi", niterations=100, criticaldelta=1.e-3,
            abund='angr'):
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
        """

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
        self.__perform(abund=abund)

    def plot(self, rebin=None, xscale='lin', yscale='lin', nsample=1) -> None:
        """
        Plots the spectrum data with errorbars, along with the best fit model and the residuals.
        :param rebin: (2 x float) Combining adjacent bins for higher significance: 1st value - sigma significance, 2nd
               value (may not be present) - maximum number of bins to reach the significance. Default None, i.e. no
               rebinning. Equivalent of `setplot rebin` command in Xspec.
        :param xscale: (str) Scaling of the x-axis, can be either 'lin'/'linear' or 'log'/'logarithmic'. Default 'lin'.
        :param yscale: (str) Same as xscale but for the y-axis.
        :param nsample: (int) If set it defines a sampling for the data points, for better visualization. Not
               considered if rebin is present. Default 1, i.e. all points are shown.
        """

        xscale_ = xscale.lower().strip()
        yscale_ = yscale.lower().strip()
        if not self.fitDone:
            print("No data available, the fit has not been run yet.")
        else:
            fig, (axd, axr) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [5, 2], 'hspace': 0})

            if xscale_ in ['lin', 'linear']:
                axd.set_xscale('linear')
                axr.set_xscale('linear')
            elif xscale_ in ['log', 'logarithmic']:
                axd.set_xscale('log')
                axr.set_xscale('log')
            else:
                raise ValueError(
                    "Invalid input type for xscale, must be one of 'lin' ('linear') or 'log' ('logarithmic')")

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
                y_error = self.fitPoints["error"][::nsample]  # [cts/s/keV]
                residuals = self.fitPoints["spectrum"][::nsample] - self.fitPoints["model"][::nsample]  # [cts/s/keV]
            else:
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
                    y_ = c / (self.spectrum.exposure * delta_ene)  # [cts/s/keV]
                    y.append(y_)  # [cts/s/keV]
                    y_error.append(np.sqrt(c) / (self.spectrum.exposure * delta_ene))  # [cts/s/keV]
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
            ax.set_xticks(range(self.nFree), labels=self.freeParNames, rotation=45, ha="right",
                          rotation_mode="anchor")
            ax.set_yticks(range(self.nFree), labels=self.freeParNames)
            ax.imshow(corr_matrix, cmap=cm["bwr"], aspect='equal', vmin=-1, vmax=1)

            # Text annotations
            for i, j in np.ndindex(corr_matrix.shape):
                ax.text(j, i, "{:.3f}".format(corr_matrix[i, j]), ha="center", va="center", color="black")

    def save(self, fileName: str, overwrite=True):
        if not self.fitDone:
            print("Cannot save before the fit has been run")
            return None
        else:
            # Creating the FITS file
            hdulist = fits.HDUList()

            # Primary (empty)
            hdulist.append(fits.PrimaryHDU([0]))

            # Files related to the spectrum to save in the header
            sp_file, arf_file, rmf_file, bkg_file = None, None, None, None
            if hasattr(self, 'spectrum'):
                if hasattr(self.spectrum, 'fileName'):
                    sp_file = self.spectrum.fileName
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

            if sp_file is not None:
                hdulist[0].header.set('SP_FILE', sp_file)
            else:
                if 'SP_FILE' in self.keywords:
                    hdulist[0].header.set('SP_FILE', self.keywords.get('SP_FILE'))

            if arf_file is not None:
                hdulist[0].header.set('ARF_FILE', arf_file)
            else:
                if 'ARF_FILE' in self.keywords:
                    hdulist[0].header.set('ARF_FILE', self.keywords.get('ARF_FILE'))

            if rmf_file is not None:
                hdulist[0].header.set('RMF_FILE', rmf_file)
            else:
                if 'RMF_FILE' in self.keywords:
                    hdulist[0].header.set('RMF_FILE', self.keywords.get('RMF_FILE'))

            if bkg_file is not None:
                hdulist[0].header.set('BKG_FILE', bkg_file)
            else:
                if 'BKG_FILE' in self.keywords:
                    hdulist[0].header.set('BKG_FILE', self.keywords.get(''))

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

            # Adding table with fit results
            fit_results_columns = [
                fits.Column(name='PARNAME', format='10A', array=self.fitResult["parnames"]),
                fits.Column(name='VALUES', format='E', array=self.fitResult["values"]),
                fits.Column(name='SIGMA', format='E', array=self.fitResult["sigma"]),
                fits.Column(name='FREE', format='L', array=self.parFree)
            ]
            hdulist.append(fits.BinTableHDU.from_columns(fits.ColDefs(fit_results_columns), name="Results"))

            # Adding covariance matrix data
            hdulist.append(fits.ImageHDU(self.fitResult["covariance"], name='Covariance'))

            # Adding table with fit points
            fit_points_columns = [
                fits.Column(name='ENERGY', format='E', array=self.fitPoints["energy"], unit='keV'),
                fits.Column(name='D_ENERGY', format='E', array=self.fitPoints["dEne"], unit='keV'),
                fits.Column(name='SPECTRUM', format='E', array=self.fitPoints["spectrum"], unit='cts/s/keV'),
                fits.Column(name='ERROR', format='E', array=self.fitPoints["error"], unit='cts/s/keV'),
                fits.Column(name='MODEL', format='E', array=self.fitPoints["model"], unit='cts/s/keV'),
                fits.Column(name='COUNTS', format='E', array=self.fitPoints["counts"], unit='---'),
                fits.Column(name='NOTICED', format='J', array=self.fitPoints["noticed"], unit='---')
            ]
            hdulist.append(fits.BinTableHDU.from_columns(fits.ColDefs(fit_points_columns), name='Points'))

            # Writing FITS file
            return hdulist.writeto(fileName, overwrite=overwrite)
