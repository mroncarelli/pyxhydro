"""
Set of methods connected to observational quantities.
"""

import numpy as np
from astropy.io import fits
from gadgetutils.phys_const import keV2erg
from pyxhydro import sixte
from pyxhydro.sphprojection.mapping import read_specmap

SP = np.float32

def countrate(inp, arf, telescope=1, xrange=None, yrange=None, erange=None) -> float:
    """
    Calculates the expected countrate of a spectral map for a given response.
    :param inp: (fits.HDUList or str) Spectral map. The input can be either a specmap (mapping module), or a Simput
        file or a string with the name of the file that contains them.
    :param arf: (fits.HDUList or sixte.Instrument or str) The response containing the effective area as a function of
        energy. The input can be either a response HDUList (or string with the file) or and Instrument of the sixte
        module (or a string with the instrument name).
    :param telescope: (int) The telescope number to use, considered only it the arf is provided via a sixte.Instrument.
        Default 1.
    :param xrange: (2 x float) Range in the x-axis [arcmin]. For spectral map assumes 0 in the center. Default None.
    :param yrange: (2 x float) Range in the y-axis [arcmin]. For spectral map assumes 0 in the center. Default None.
    :param erange: (2 x float) Energy range [keV]. Default None.
    :return: (float) The expected countrate [cts s^-1].
    """

    def e_sp_from_spmap(spmap: dict, xrange=None, yrange=None, erange=None) -> tuple:
        """
        Extracts the energy bins spectra from a spectral map.
        :param spmap: (dict) Spectral map.
        :param xrange: (2 x float) Range in the x-axis [deg]. Assumes 0 in the center. Default None.
        :param yrange: (2 x float) Range in the y-axis [deg]. Assumes 0 in the center. Default None.
        :param erange: (2 x float) Energy range [keV]. Default None.
        :return: (2 x float array) Central energy of the bins [keV] and total spectrum [photons s^-1 cm^-2].
        """

        energy = spmap["energy"]  # [keV]
        data = spmap["data"]
        d_ene = spmap["energy_interval"]  # [keV]

        if xrange is not None or yrange is not None:
            npix = data.shape[0]
            size = spmap["size"]  # [deg]
            step = size / npix  # [arcmin]
            pvec = np.linspace(0.5 * (-size + step), 0.5 * (size - step), num=npix, endpoint=True)  # [deg]

            if xrange is not None:
                data = data[np.where((pvec >= xrange[0]) & (pvec < xrange[1]))[0], :, :]

            if yrange is not None:
                data = data[:, np.where((pvec >= yrange[0]) & (pvec < yrange[1]))[0], :]

        if erange is not None:
            index_ecut =  np.where((energy >= erange[0]) & (energy < erange[1]))[0]
            energy = energy[index_ecut]
            d_ene = d_ene[index_ecut]
            data = data[:, :, index_ecut]
            del index_ecut

        spectrum = np.zeros(shape=(len(energy)), dtype=SP)  # [photons (or keV) keV^-1 s^-1 cm^-2 arcmin^-2]
        nx, ny, nene = data.shape
        for ipix in range(nx):
            for jpix in range(ny):
                spectrum += data[ipix, jpix, :]  # [photons (or keV) keV^-1 s^-1 cm^-2 arcmin^-2]

        if spmap["flag_ene"]:
            spectrum /= energy  # [photons keV^-1 s^-1 cm^-2 arcmin^-2]

        spectrum *= d_ene * spmap["pixel_size"] ** 2  # [photons s^-1 cm^-2]

        return energy, spectrum  # [keV], [photons s^-1 cm^-2]


    def e_sp_from_simput(simput: fits.hdu.hdulist.HDUList, xrange=None, yrange=None, erange=None) -> tuple:
        """
        Extracts the energy bins spectra from a imput file HDUList. Assumes that the energy coordinate is the same for
        all spectra and that it is uniform.
        :param simput: (HDUList) Simput file HDUList.
        :param xrange: (2 x float) Range in the x-axis (RA) [deg].
        :param yrange: (2 x float) Range in the y-axis (DEC) [deg].
        :param erange: (2 x float) Energy range [keV].
        :return: (2 x float array) Central energy of the bins [keV] and total spectrum [photons s^-1 cm^-2]
        """

        energy = simput[2].data['ENERGY'][0]  # Energy coordinates, assumed to be the same for all spectra [keV]
        d_ene = (energy[-1] - energy[0]) / (len(energy) - 1)  # [keV]
        data = simput[2].data['FLUXDENSITY']  # [photons s^-1 cm^-2 keV^-1]
        for isp in range(len(data)):
            data[isp] *= d_ene  # [photons s^-1 cm^-2]

        # Renormalization. This is actually not necessary if the Simput file has been created with the sixte module.
        flux = simput[1].data['FLUX']  # [erg s^-1 cm^-2]
        for isp in range(len(data)):
            flux0 = np.sum(data[isp] * energy) * keV2erg  # [erg s^-1 cm^-2]
            data[isp] *= flux[isp] / flux0  # [photons s^-1 cm^-2]
        del flux

        if xrange is not None and yrange is None:
            ra = simput[1].data['RA']  # [deg]
            data = data[np.where((ra >= xrange[0]) & (ra < xrange[1]))[0], :]
            del ra
        elif xrange is None and yrange is not None:
            dec = simput[1].data['DEC']  # [deg]
            data = data[np.where((dec >= yrange[0]) & (dec < yrange[1]))[0], :]
            del dec
        elif xrange is not None and yrange is not None:
            ra = simput[1].data['RA']  # [deg]
            dec = simput[1].data['DEC']  # [deg]
            data = data[np.where((ra >= xrange[0]) & (ra < xrange[1]) &
                                 (dec >= yrange[0]) & (dec < yrange[1]))[0], :]
            del ra, dec

        if erange is not None:
            index_ecut =  np.where((energy >= erange[0]) & (energy < erange[1]))[0]
            energy = energy[index_ecut]
            data = data[:, index_ecut]
            del index_ecut

        spectrum = np.zeros(shape=(len(energy)), dtype=SP)  # [photons s^-1 cm^-2]
        nsp, nene = data.shape
        for isp in range(nsp):
            spectrum += data[isp, :]  # [photons s^-1 cm^-2]

        return energy, spectrum  # [keV], [photons s^-1 cm^-2]

    # Checking input type and determining energy and spectrum based on it
    input_type = type(inp)
    if input_type == dict:
        # Assuming it's a specmap ([keV], [photons s^-1 cm^-2])
        energy, spectrum = e_sp_from_spmap(inp, xrange=xrange, yrange=yrange, erange=erange)
    elif input_type == fits.hdu.hdulist.HDUList:
        # Assuming it's a Simput HUDList ([keV], [photons s^-1 cm^-2])
        energy, spectrum = e_sp_from_simput(inp, xrange=xrange, yrange=yrange, erange=erange)
    elif input_type == str:
        try:
            # Trying with a file containing a specmap ([keV], [photons s^-1 cm^-2])
            energy, spectrum = e_sp_from_spmap(read_specmap(inp), xrange=xrange, yrange=yrange, erange=erange)
        except:
            # Trying with a Simput file ([keV], [photons s^-1 cm^-2])
            energy, spectrum = e_sp_from_simput(fits.open(inp), xrange=xrange, yrange=yrange, erange=erange)
    else:
        raise ValueError("Invalid input type. Must be a specmap dictionary, a Simput HUDList or a string with a file "
                         "name containing one of them.")

    # Checking arf input type and extracting data based on it
    type_arf = type(arf)
    if type_arf == str:
        instrument = sixte.instruments.get(arf)
        if instrument is not None:
            arf_hdulist = fits.open(instrument.path + "/" + instrument.arf[telescope - 1])
        else:
            try:
                arf_hdulist = fits.open(arf)
            except:
                raise ValueError("Invalid input: " + arf + " is not an instrument name or FITS file.")
    elif type_arf == fits.hdu.hdulist.HDUList:
        arf_hdulist = arf
    elif type_arf == sixte.Instrument:
        arf_hdulist = fits.open(arf.path + "/" + arf.arf[telescope - 1])
    else:
        raise ValueError("Invalid input type. Must be a FITS HUDList, sixte.Instrument or a string.")

    energy_arf = 0.5 * (arf_hdulist[1].data['ENERG_LO'] + arf_hdulist[1].data['ENERG_HI'])  # [keV]
    effarea_arf = arf_hdulist[1].data['SPECRESP']  # [cm^2]

    effarea = np.interp(energy, energy_arf, effarea_arf, left=0, right=0, period=None)  # [cm^2]

    return np.sum(spectrum * effarea)  # [counts s^-1]


def mosaic(n, center=(0, 0), fov=1) -> list:
    """
    Creates a square mosaic of pointings.
    :param n: (int) Number of sides of the square mosaic.
    :param center: (float x 2) Coordinate of the center of the mosaic [arbitrary units], default (0, 0).
    :param fov: (float) Field of view, default 1 [arbitrary units]
    :return: (list of dict) List of pointings containing the following keys:
            - x: (float) x-coordinate of the pointing center
            - y: (float) y-coordinate of the pointing center
            - ring: (int) ring index with the respect to the '00' pointing located in the center of the mosaic
            - tag: (str) a tag that identifies the pointing, being '00' the central pointing (rounded low/left when
                n is even) and with numbers 1, 2, 3, ... toward the up/right, and 9, 8, 7, ... towards the low/left.
                The uniqueness of the tag will fail for n > 10.
    """
    coord = np.linspace(-0.5 * (n - 1), 0.5 * (n - 1), n, endpoint = True, dtype=SP)
    zero_pixel = int(np.floor((n - 1) / 2))
    result = []
    for i in range(n):
        for j in range(n):
            result.append({'x': coord[i] * fov + center[0],
                           'y': coord[j] * fov + center[1],
                           'ring': max(abs(i - zero_pixel), abs(j - zero_pixel)),
                           'tag': str((i - zero_pixel) % 10) + str((j - zero_pixel) % 10)})

    return result


def ra_corr(ra, units=None, zero=False):
    """
    Converts right ascension coordinates in the interval [0, 2pi[
    :param ra: (float) Right ascension [rad] or [deg]
    :param units: (str) Units of the ra array, can be radians ('rad'), degrees ('deg') or acrmin ('arcmin'), default
        'rad'
    :param zero: (bool) If True coordinates are converted in zero-centered interval, i.e. [-pi, pi[, default False
    :return: (float) Corrected value of right ascension
    """
    units_ = units.lower() if units else 'rad'
    if units_ in ['rad', 'radians']:
        full = 2 * np.pi  # [rad]
    elif units_ in ['deg', 'degree']:
        full = 360  # [deg]
    elif units_ == 'arcmin':
        full = 21600  # [arcmim]
    else:
        raise ValueError("ERROR IN ra_corr. Invalid unit: ", units, "Must be one of 'rad', 'radians', 'deg', 'degree' "
                                                                    "'arcmin' or None")

    inp_type = type(ra)
    if inp_type in [tuple, list]:
        ra_ = np.asarray(ra)
    else:
        ra_ = ra
    result = ra_ % full  # in range [0, 2pi[ or [0, 360[

    if zero:
        corr = result >= 0.5 * full
        if type(corr) in [bool, np.bool_]:
            if corr:
                result -= full
        else:
            result[corr] -= full  # in range [-pi, pi[ (for rad)

    if inp_type in [tuple, list]:
        result = inp_type(result)

    return result
