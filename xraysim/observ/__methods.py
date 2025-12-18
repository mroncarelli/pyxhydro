"""
Set of methods connected to observational quantities.
"""

import numpy as np
from astropy.io import fits
from gadgetutils.phys_const import keV2erg
from xraysim import sixte
from xraysim.sphprojection.mapping import read_speccube

SP = np.float32

def countrate(inp, arf, telescope=1, xrange=None, yrange=None, erange=None) -> float:
    """
    Calculates the expected countrate of a spectral cube for a given response.
    :param inp: (fits.HDUList or str) Spectral cube. The input can be either a speccube (mapping module), or a Simput
        file or a string with the name of the file that contains them.
    :param arf: (fits.HDUList or sixte.Instrument or str) The response containing the effective area as a function of
        energy. The input can be either a response HDUList (or string with the file) or and Instrument of the sixte
        module (or a string with the instrument name).
    :param telescope: (int) The telescope number to use, considered only it the arf is provided via a sixte.Instrument.
        Default 1.
    :param xrange: (2 x float) Range in the x-axis [arcmin]. For spectral cube assumes 0 in the center. Default None.
    :param yrange: (2 x float) Range in the y-axis [arcmin]. For spectral cube assumes 0 in the center. Default None.
    :param erange: (2 x float) Energy range [keV]. Default None.
    :return: (float) The expected countrate [cts s^-1].
    """

    def e_sp_from_spcube(spcube: dict, xrange=None, yrange=None, erange=None) -> tuple:
        """
        Extracts the energy bins spectra from a spectral cube.
        :param spcube: (dict) Spectral cube.
        :param xrange: (2 x float) Range in the x-axis [arcmin]. Assumes 0 in the center. Default None.
        :param yrange: (2 x float) Range in the y-axis [arcmin]. Assumes 0 in the center. Default None.
        :param erange: (2 x float) Energy range [keV]. Default None.
        :return: (2 x float array) Central energy of the bins [keV] and total spectrum [photons s^-1 cm^-2].
        """

        energy = spcube["energy"]  # [keV]
        data = spcube["data"]
        d_ene = spcube["energy_interval"] # (energy[-1] - energy[0]) / (len(energy) - 1)  # [keV]

        if xrange is not None or yrange is not None:
            npix = data.shape[0]
            size = spcube["size"] * 60  # [arcmin]
            step = size / npix  # [arcmin]
            pvec = np.linspace(0.5 * (-size + step), 0.5 * (size - step), num=npix, endpoint=True)  # [arcmin]

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

        if spcube["flag_ene"]:
            spectrum /= energy  # [photons keV^-1 s^-1 cm^-2 arcmin^-2]

        spectrum *= d_ene * spcube["pixel_size"] ** 2 # [photons s^-1 cm^-2]

        return energy, spectrum  # [keV], [photons s^-1 cm^-2]


    def e_sp_from_simput(simput: fits.hdu.hdulist.HDUList, xrange=None, yrange=None, erange=None) -> tuple:
        """
        Extracts the energy bins spectra from a imput file HDUList. Assumes that the energy coordinate is the same for
        all spectra and that it is uniform.
        :param simput: (HDUList) Simput file HDUList.
        :param xrange: (2 x float) Range in the x-axis (RA) [arcmin].
        :param yrange: (2 x float) Range in the y-axis (DEC) [arcmin].
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
            ra_arcmin = simput[1].data['RA'] * 60.  # [arcmin]
            data = data[np.where((ra_arcmin >= xrange[0]) & (ra_arcmin < xrange[1]))[0], :]
            del ra_arcmin
        elif xrange is None and yrange is not None:
            dec_arcmin = simput[1].data['DEC'] * 60.  # [arcmin]
            data = data[np.where((dec_arcmin >= yrange[0]) & (dec_arcmin < yrange[1]))[0], :]
            del dec_arcmin
        elif xrange is not None and yrange is not None:
            ra_arcmin = simput[1].data['RA'] * 60.
            dec_arcmin = simput[1].data['DEC'] * 60.
            data = data[np.where((ra_arcmin >= xrange[0]) & (ra_arcmin < xrange[1]) &
                                 (dec_arcmin >= yrange[0]) & (dec_arcmin < yrange[1]))[0], :]
            del ra_arcmin, dec_arcmin

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
        # Assuming it's a speccube ([keV], [photons s^-1 cm^-2])
        energy, spectrum = e_sp_from_spcube(inp, xrange=xrange, yrange=yrange, erange=erange)
    elif input_type == fits.hdu.hdulist.HDUList:
        # Assuming it's a Simput HUDList ([keV], [photons s^-1 cm^-2])
        energy, spectrum = e_sp_from_simput(inp, xrange=xrange, yrange=yrange, erange=erange)
    elif input_type == str:
        try:
            # Trying with a file containing a speccube ([keV], [photons s^-1 cm^-2])
            energy, spectrum = e_sp_from_spcube(read_speccube(inp), xrange=xrange, yrange=yrange, erange=erange)
        except:
            # Trying with a Simput file ([keV], [photons s^-1 cm^-2])
            energy, spectrum = e_sp_from_simput(fits.open(inp), xrange=xrange, yrange=yrange, erange=erange)
    else:
        raise ValueError("Invalid input type. Must be a speccube dictionary, a Simput HUDList or a string with a file "
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
