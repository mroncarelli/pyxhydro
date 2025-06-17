import os
import sys

sys.path.append(os.environ.get("HEADAS") + "/lib/python")
# TODO: the three lines above are necessary only to make the code work in IntelliJ (useful for debugging)

import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
import numpy as np
import xspec as xsp

xsp.Xset.allowNewAttributes = True
xsp.Xset.chatter = 0
xsp.Xset.addModelString("APECROOT", "3.0.9")
xsp.Xset.addModelString("APECTHERMAL", "yes")
xsp.Xset.abund = "angr"  # TODO: is a default needed?


def notice_list_split(notice) -> list:
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


def save_xspec_state(self) -> None:
    """
    Saves the state of some global Xspec variables before fitting, ready to be restored after the fit has been
    performed. In detail, it saves the notice arrays in the xspec.AllData object by creating the `noticeState`
    attribute, and the name of the active Model in the xspec.AllModels object by creating the `activeModel` attribute.
    :param self: (xspec.XspecSettings) xspec.Xset
    :return: None
    """
    self.noticeState = []
    for index in range(xsp.AllData.nSpectra):
        self.noticeState.append(xsp.AllData(index + 1).noticed)

    self.activeModel = xsp.AllModels.sources[1]
    self.plotState = {"xAxis": xsp.Plot.xAxis}


xsp.XspecSettings.saveXspecState = save_xspec_state


def restore_xspec_state(self) -> None:
    """
    Restores the state of the notice arrays in `xspec.AllData` object and of the active Model in `xspec.AllModels`.
    Deletes the `noticeState` and `activeModel` attributes from `xspec.Xset` after.
    :param (xspec.XspecSettings) xspec.Xset
    :return: None
    """
    for index in range(xsp.AllData.nSpectra):
        intervals_list = notice_list_split(self.noticeState[index])
        if len(intervals_list) >= 1:
            command_string = str(intervals_list[0][0]) + '-' + str(intervals_list[0][1])
            for i in range(1, len(intervals_list)):
                command_string += ',' + str(intervals_list[i][0]) + '-' + str(intervals_list[i][1])
            xsp.AllData(index + 1).notice(command_string)

    del self.noticeState

    xsp.AllModels.setActive(xsp.Xset.activeModel)
    del xsp.Xset.activeModel

    xsp.Plot.xAxis = self.plotState["xAxis"]
    del xsp.Xset.plotState


xsp.XspecSettings.restoreXspecState = restore_xspec_state


def highlight_spectrum(self, index=1) -> None:
    """
    Highlights a single spectrum to prepare it for the fit by ignoring all channel of all the other spectra
    :param self: (xspec.DataManager) xspec.AllData
    :param index: Index of the spectrum, default 1
    :return: None
    """
    for i in range(1, self.nSpectra + 1):
        if i != index and self(i).noticed != []:
            channel_min = min(self(i).noticed)
            channel_max = max(self(i).noticed)
            self(i).ignore(str(channel_min) + "-" + str(channel_max))


xsp.DataManager.highlightSpectrum = highlight_spectrum


def ignore_string(x) -> str:
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


def ignore(spectrum: xsp.Spectrum, erange=(None, None)) -> None:
    """
    Sets the energy range to ignore on a spectrum.
    :param spectrum: (xspec.Spectrum) Spectrum.
    :param erange: (float 2, or int 2, or str 2) Energy range.
    :return: None
    """
    if len(erange) >= 2:
        if erange[0] is not None and erange[1] is not None:
            if erange[1] <= erange[0]:
                print("ERROR in ignore: invalid input:", erange, "Second argument must be larger than the first")
                raise ValueError

    # Setting lower energy limit
    if erange[0] is not None:
        spectrum.ignore("**-" + ignore_string(erange[0]))
    # Setting lower energy limit
    if erange[1] is not None:
        spectrum.ignore(ignore_string(erange[1]) + "-**")

    return None


class SpecFit(xsp.Model):
    def __init__(self, spectrum, model, bkg='USE_DEFAULT', rmf='USE_DEFAULT', arf='USE_DEFAULT', setPars=None):
        self.spectrum = xsp.Spectrum(spectrum, backFile=bkg, respFile=rmf, arfFile=arf)
        xsp.Model.__init__(self, model, modName='SpecFit' + str(self.spectrum.index), setPars=setPars)
        self.fitData = None
        self.fitResult = None

    def get_parnames(self) -> tuple:
        """
        Returns a tuple with the model parameter names.
        :param self: (ModelFit)
        :return: (tuple) Tuple of strings containing the parameter names.
        """
        return tuple(self(self.startParIndex + index).name for index in range(self.nParameters))

    def get_parvals(self) -> tuple:
        """
        Returns a tuple with the model parameter values.
        :param self: (ModelFit)
        :return: (tuple) Tuple containing the parameter values.
        """
        return tuple(self(self.startParIndex + index).values[0] for index in range(self.nParameters))

    def get_errors(self) -> tuple:
        """
        Returns a tuple with the model parameter errors.
        :param self: (ModelFit)
        :return: (tuple) Tuple containing the parameter errors.
        """
        return tuple(self(self.startParIndex + index).sigma for index in range(self.nParameters))

    def perform(self) -> None:
        """
        Equivalent of the `xspec.Fit.perform` method adapted to the `SpecFit` class. It allows to run the fit of the
        `xspec.Spectrum` loaded in the `spectrum` attribute with the `xspec.Model` of the instance while preserving the
        state of the `xspec` global objects (i.e. `xspec.AllData` and `xspec.AllModels`).
        :return: None
        """
        xsp.Xset.saveXspecState()
        xsp.AllData.highlightSpectrum(self.spectrum.index)
        xsp.AllModels.setActive(self.name)
        xsp.Fit.perform()

        # Saving the data of the fit points
        xsp.Plot.xAxis = "keV"
        xsp.Plot("data")
        self.fitData = {
            "x": np.asarray(xsp.Plot.x()),  # energy [keV]
            "y": np.asarray(xsp.Plot.y()),
            "yErr": np.asarray(xsp.Plot.yErr()),
            "model": np.asarray(xsp.Plot.model())
        }

        # Saving fit results
        self.fitResult = {
            "parnames": self.get_parnames(),
            "values": self.get_parvals(),
            "sigma": self.get_errors(),
            "statistic": xsp.Fit.statistic,
            "dof": xsp.Fit.dof,
            "rstat": xsp.Fit.statistic / (xsp.Fit.dof - 1),
            "covariance": xsp.Fit.covariance,
            "method": xsp.Fit.statMethod,
            "nIterations": xsp.Fit.nIterations,
            "criticalDelta": xsp.Fit.criticalDelta
        }

        xsp.Xset.restoreXspecState()

    def covariance_matrix(self):
        """
        Returns the covariance matrix of the fit.
        :return: (2D array) The covariance matrix
        """

        if self.fitData is None:
            print("No data available, the fit has not been run yet.")
            return None
        else:
            # Creating covariance matrix
            npar = self.nParameters
            result = np.ndarray([npar, npar])

            # Filling diagonal and lower part
            index = 0
            for i in range(npar):
                for j in range(i + 1):
                    result[i, j] = self.fitResult["covariance"][index]
                    index += 1
            # Filling upper part
            for i in range(npar):
                for j in range(i + 1, npar):
                    result[i, j] = result[j, i]

            return result

    def correlation_matrix(self):
        """
        Returns the correlation matrix of the fit.
        :return: (2D array) The correlation matrix
        """
        result = self.covariance_matrix()
        if result is not None:
            for i, j in np.ndindex(result.shape):
                result[i, j] /= self.fitResult["sigma"][i] * self.fitResult["sigma"][j]

        return result

    def run(self, erange=(None, None), start=None, fixed=None, method="chi", niterations=100, criticaldelta=1.e-3):
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
        """

        # Energy range
        ignore(self.spectrum, erange)

        # Initial conditions
        if start is not None:
            for index, par in enumerate(start):
                self(self.startParIndex + index).values = par

        # Fixed/free parameter
        if fixed is not None:
            for index, frozen in enumerate(fixed):
                self(self.startParIndex + index).frozen = frozen
        else:
            for index in range(self.nParameters):
                self(self.startParIndex + index).frozen = False

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
        self.perform()

    def plot(self, nsample=1, xscale='lin', yscale='lin') -> None:
        """
        Plots the spectrum data with errorbars, along with the best fit model and the residuals.
        :param nsample: (int) If set it defines a sampling for the data points, for better visualization
        :param xscale: (str) Scaling of the x-axis, can be either 'lin'/'linear' or 'log'/'logarithmic'. Default 'lin'.
        :param yscale: (str) Same as xscale but for the y-axis.
        """

        xscale_ = xscale.lower().strip()
        yscale_ = yscale.lower().strip()
        if self.fitData is None:
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
            axd.errorbar(self.fitData["x"][::nsample], self.fitData["y"][::nsample],
                         yerr=self.fitData["yErr"][::nsample], color='black', linestyle='', fmt='.', zorder=0)
            axd.plot(self.fitData["x"], self.fitData["model"], color="limegreen", zorder=1)

            axr.set_xlabel("Energy (keV)")
            axr.set_ylabel("Diff.")
            axr.errorbar(self.fitData["x"][::nsample], self.fitData["y"][::nsample] - self.fitData["model"][::nsample],
                         yerr=self.fitData["yErr"][::nsample], color='black', linestyle='', fmt='.', zorder=0)
            axr.plot((self.fitData["x"][0], self.fitData["x"][-1]), (0, 0), color="limegreen", zorder=1)

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
            ax.set_xticks(range(self.nParameters), labels=self.fitResult["parnames"], rotation=45, ha="right",
                          rotation_mode="anchor")
            ax.set_yticks(range(self.nParameters), labels=self.fitResult["parnames"])
            ax.imshow(corr_matrix, cmap=cm["bwr"], aspect='equal', vmin=-1, vmax=1)

            # Text annotations
            for i, j in np.ndindex(corr_matrix.shape):
                ax.text(j, i, "{:.3f}".format(corr_matrix[i, j]), ha="center", va="center", color="black")
