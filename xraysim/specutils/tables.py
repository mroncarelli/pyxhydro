from astropy.io import fits
import numpy as np

sp = np.float32

import os
import sys

sys.path.append(os.environ.get("HEADAS") + "/lib/python")
# TODO: the three lines above are necessary only to make the code work in IntelliJ (useful for debugging)


def nearest_index_sorted(array, value) -> int:
    """
    Returns the index whose value is closest to the input value. Assumes the array is sorted in ascending order.
    :param array: array (sorted ascending) to search into
    :param value: value to search
    :return: index of the closest value
    """
    idx = int(np.searchsorted(array, value, side="left"))
    if idx > 0 and np.abs(value - array[idx - 1]) < np.abs(value - array[idx]):
        idx += -1
    return idx


def largest_index_smaller(array, value):
    """
    Returns the largest index whose value is smaller than the input value. Assumes the array is sorted in ascending
    order.
    :param array: array (sorted ascending) to search into
    :param value: value: value to search
    :return: index of the largest value smaller than the input value
    """
    idx = len(array) - 1
    while idx > 0 and array[idx] >= value:
        idx += -1

    if array[idx] < value:
        return idx
    else:
        return None


def smallest_index_larger(array, value):
    """
    Returns the smallest index whose value is larger than the input value. Assumes the array is sorted in ascending
    order.
    :param array: array (sorted ascending) to search into
    :param value: value: value to search
    :return: index of the smallest value larger than the input value
    """
    idx = 0
    while idx < len(array) - 1 and array[idx] <= value:
        idx += 1

    if array[idx] > value:
        return idx
    else:
        return None


def reversed_fits_axis_order(inp) -> bool:
    """
    Determines whether a spectable in a FITS file has been created following the column-major-order convention and,
    therefore, requires to transpose the input.
    :param inp: input file (FITS) or HDUList
    :return: (bool) True if it is in reversed axis order, False otherwise
    """

    input_type = type(inp)
    if input_type == str:
        hdulist = fits.open(inp)
        hdulist.close()
    elif input_type == fits.hdu.HDUList:
        hdulist = inp
    else:
        raise ValueError("Invalid input in reversed_fits_axis_order: must be a string or HDUList")

    if hdulist[0].header['NZ'] == hdulist[0].header['NENE']:
        # In this case I can't determine just from the array shape, I have to guess based on the header.
        # I know that IDL routines write tables in FITS files in column-major-order
        return hdulist[0].header.comments['SIMPLE'].lower().startswith('written by idl')
    else:
        # In this case I verify if NENE correspond to the 1st dimension and NZ to the last: in this case I have a
        # column-major-order
        return ((hdulist[0].header['NENE'], hdulist[0].header['NZ']) ==
                (hdulist[0].data.shape[0], hdulist[0].data.shape[-1]))


def read_spectable(filename: str, z_cut=None, temperature_cut=None, energy_cut=None) -> dict:
    """
    Reads a spectrum table from a file.
    :param filename: (str) input file (FITS)
    :param z_cut: (float 2) optional redshift interval where to cut the table [---]
    :param temperature_cut: (float 2) optional temperature interval where to cut the table [keV]
    :param energy_cut: (float 2) optional energy interval where to cut the table [keV]
    :return: a structure containing the spectrum table, in the 'data' key, with other information in the other keys.
    With standard Xspec parameters the units are [10^-14 photons s^-1 cm^3] or [10^-14 keV s^-1 cm^3] if the
    header of the file has the keyword FLAG_ENE = 1.
    """
    hdulist = fits.open(filename)
    spectable = hdulist[0].data
    if reversed_fits_axis_order(hdulist):
        spectable = spectable.transpose()
    z = hdulist[1].data
    temperature = hdulist[2].data
    energy = hdulist[3].data

    # Sanity check
    assert spectable.shape[0] == len(z) and spectable.shape[1] == len(temperature) and spectable.shape[2] == len(energy)

    if z_cut:
        try:
            z0, z1 = float(z_cut[0]), float(z_cut[1])
        except BaseException:
            raise ValueError("Invalid tcut: ", z_cut, "Must be a 2d number vector")
        i0, i1 = largest_index_smaller(z, z0), smallest_index_larger(z, z1)
        if i0 is None:
            i0 = 0  # TODO: WARNING
        if i1 is None:
            i1 = len(z) - 1  # TODO: WARNING
        z = z[i0:i1 + 1]
        spectable = spectable[i0:i1 + 1, :, :]

    if temperature_cut:
        try:
            t0, t1 = float(temperature_cut[0]), float(temperature_cut[1])
        except BaseException:
            raise ValueError("Invalid tcut: ", temperature_cut, "Must be a 2d number vector")
        i0, i1 = largest_index_smaller(temperature, t0), smallest_index_larger(temperature, t1)
        if i0 is None:
            i0 = 0  # TODO: WARNING
        if i1 is None:
            i1 = len(temperature) - 1  # TODO: WARNING
        if i0 == i1:
            i1 += 1
            if i1 == len(temperature):
                i0, i1 = len(temperature) - 2, len(temperature) - 1
        temperature = temperature[i0:i1 + 1]
        spectable = spectable[:, i0:i1 + 1, :]

    if energy_cut:
        try:
            e0, e1 = float(energy_cut[0]), float(energy_cut[1])
        except BaseException:
            raise ValueError("Invalid tcut: ", energy_cut, "Must be a 2d number vector")
        i0, i1 = largest_index_smaller(energy, e0), smallest_index_larger(energy, e1)
        if i0 is None:
            i0 = 0  # TODO: WARNING
        if i1 is None:
            i1 = len(energy) - 1  # TODO: WARNING
        energy = energy[i0:i1 + 1]
        spectable = spectable[:, :, i0:i1 + 1]

    result = {
        'data': spectable,
        'z': z,
        'temperature': temperature,
        'energy': energy,
        'units': hdulist[0].header['UNITS'],
        'flag_ene': hdulist[0].header['FLAG_ENE'] == 1,
        'model': hdulist[0].header['MODEL'],
        'temperature_units': hdulist[2].header['UNITS'],
        'energy_units': hdulist[3].header['UNITS']
    }
    if 'ABUND' in hdulist[0].header:
        result['abund'] = hdulist[0].header['ABUND']
    if 'TBROAD' in hdulist[0].header:
        result['tbroad'] = hdulist[0].header['TBROAD']
    if 'PARAM' in hdulist[0].header:
        result['param'] = hdulist[0].header['PARAM']
    if 'NH' in hdulist[0].header:
        result['nh'] = hdulist[0].header['NH']

    hdulist.close()
    return result


def write_spectable(spectable: dict, file: str, overwrite=True) -> int:
    """
    Writes a spectral table into a FITS file.
    :param spectable: (dict) Spectral table as from, i.e. apec_table
    :param file: (str) Output file
    :param overwrite:
    :return: Output of fits.HDUlist.writeto()
    """


    # Creagin HDU list
    hdulist = fits.HDUList()

    # Appending data
    # I transpose the array so that Fits View reads it correctly and for compatibility with old IDL tables. When read
    # with read_spectable the array is transposed back so that the shape matches the original one.
    hdulist.append(fits.PrimaryHDU(spectable.get('data').transpose()))
    hdulist.append(fits.ImageHDU(spectable.get('z'), name="Redshift"))
    hdulist.append(fits.ImageHDU(spectable.get('temperature'), name="Temperature"))
    hdulist.append(fits.ImageHDU(spectable.get('energy'), name="Energy"))

    # Setting headers
    hdulist[0].header.set('MODEL', spectable.get('model'))
    hdulist[0].header.set('ABUND', spectable.get('abund'))
    hdulist[0].header.set('METAL', str(spectable.get('metallicity')))
    hdulist[0].header.set('TBROAD', 1 if spectable.get('tbroad') else 0)
    hdulist[0].header.set('FLAG_ENE', 1 if spectable.get('flag_ene') else 0)
    hdulist[0].header.set('NZ', len(spectable.get('z')))
    hdulist[0].header.set('NTEMP', len(spectable.get('temperature')))
    hdulist[0].header.set('NENE', len(spectable.get('energy')))
    hdulist[0].header.set('UNITS', spectable.get('units'))
    hdulist[1].header.set('NZ', len(spectable.get('z')))
    hdulist[1].header.set('UNITS', "[---]")
    hdulist[2].header.set('NTEMP', len(spectable.get('temperature')))
    hdulist[2].header.set('UNITS', spectable.get('temperature_units'))
    hdulist[3].header.set('NENE', len(spectable.get('energy')))
    hdulist[3].header.set('UNITS', spectable.get('energy_units'))

    return hdulist.writeto(file, overwrite=overwrite)


def calc_spec(spectable: dict, z: float, temperature: float, no_z_interp=False, flag_ene=False) -> np.ndarray:
    """
    Calculates a spectrum from a table for a given redshift and temperature
    :param spectable: structure containing the spectrum table
    :param z: redshift where to compute the spectrum [---]
    :param temperature: temperature where to compute the spectrum [keV]
    :param no_z_interp: (boolean) if set to True redshift interpolation is turned off (useful to avoid line-emission
     smearing in high resolution spectra)
    :param flag_ene: (boolean) if True the spectrum is calculated in energy, if False in photons (default False)
    :return: array containing the spectrum. With standard Xspec parameters the units are [10^-14 photons s^-1 cm^3] or
    [10^-14 keV s^-1 cm^3] if flag_ene is set to True.
    """

    data = spectable.get('data')  # [10^-14 photons s^-1 cm^3] or [10^-14 keV s^-1 cm^3]
    nene = data.shape[2]
    z_table = spectable.get('z')
    temperature_table = spectable.get('temperature')  # [keV]
    flag_ene_table = spectable.get('flag_ene')

    # Redshift (index 0)
    if no_z_interp:
        iz = nearest_index_sorted(z_table, z)
        data = data[iz, :, :]
    else:
        iz0 = largest_index_smaller(z_table, z)
        if iz0 is None:
            iz0 = 0  # TODO: WARNING
        elif iz0 == len(z_table) - 1:
            iz0 = len(z_table) - 2  # TODO: WARNING
        iz1 = iz0 + 1
        fz = (z - z_table[iz0]) / (z_table[iz1] - z_table[iz0])
        data = (1 - fz) * data[iz0, :, :] + fz * data[iz1, :, :]

    # Temperature (index 1)
    it0 = largest_index_smaller(temperature_table, temperature)
    if it0 is None:
        it0 = 0  # TODO: WARNING
    elif it0 == len(temperature_table) - 1:
        it0 = len(temperature_table) - 2  # TODO: WARNING
    it1 = it0 + 1
    ft = (np.log(temperature) - np.log(temperature_table[it0])) / (
            np.log(temperature_table[it1]) - np.log(temperature_table[it0]))
    valid = np.where(data[it0, :] * data[it0, :] > 0.)
    result = np.zeros(nene, dtype=sp)
    result[valid] = np.exp((1 - ft) * np.log(data[it0, valid]) + ft * np.log(
        data[it1, valid]))  # [10^-14 photons s^-1 cm^3] or [10^-14 keV s^-1 cm^3]

    # Converting photons to energy or vice versa if required
    if flag_ene != flag_ene_table:
        energy = spectable.get('energy')  # [keV]
        if flag_ene:
            for ind, ene in enumerate(energy):
                result[:, :, ind] *= ene  # [10^-14 keV s^-1 cm^3]
        else:
            for ind, ene in enumerate(energy):
                result[:, :, ind] /= ene  # [10^-14 photons s^-1 cm^3]

    return result  # [10^-14 photons s^-1 cm^3] or [10^-14 keV s^-1 cm^3]


def apec_table(nz: int, zmin: float, zmax: float, ntemp: int, tmin: float, tmax: float, nene: int, emin: float,
               emax: float, metal=0, apecroot=None, tbroad=True, abund='angr', flag_ene=False) -> dict:
    """
    Creates a 3D Apec spectral table with fixed metallicity.
    :param nz: (int) Number of redshifts, first dimension of the output table
    :param zmin: (float) Minimum redshift
    :param zmax: (float) Maximum redshift
    :param ntemp: (int) Number of temperatures, second dimension of the output table
    :param tmin: (float) Minimum temperature [keV]
    :param tmax: (float) Maximum temperature [keV]
    :param nene: (int) Number of energy bins, third dimension of the output table
    :param emin: (float) Minimum energy
    :param emax: (float) Maximum energy
    :param metal: (float or tuple/list) Metallicity [Solar]. If a single value is provided it applies to all metals, if
    a tuple/list is provided each value will correspond to the 28 metals of the vvapec model of Xspec
    (https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/node134.html). Default 0
    :param apecroot: (str) Root table for Apec version. Default None, i.e. default table decide by Xspec
    :param tbroad: (bool) Thermal broadening turned on (True) or off (False). Default True
    :param abund: (str) Solar abundance reference for Xspec (see
    https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/node116.html), default 'angr', i.e. Anders & Grevesse 1989
    :param flag_ene: (bool) If set to True spectra are in [10^-14 keV s^-1 cm^3], if False in
    [10^-14 photons s^-1 cm^3], default False
    :return: (dict) Dictionary containing the spectable in the "data" key, and other properties.
    """

    import xspec as xsp

    # Saving current PyXspec settings to restore them at the end of the procedure
    chatter_ = xsp.Xset.chatter
    model_strings_ = xsp.Xset.modelStrings
    abund_ = xsp.Xset.abund[0:4]

    # General settings
    xsp.Xset.chatter = 0
    xsp.AllModels.setEnergies(str(emin)+ " " + str(emax) + " " + str(nene) +" lin")

    # Optional settings
    if apecroot:
        xsp.Xset.addModelString("APECROOT", apecroot)
    if type(tbroad) == bool:
        xsp.Xset.addModelString("APECTHERMAL", "yes" if tbroad else "no")
    if abund:
        xsp.Xset.abund = abund

    # Redshift, temperature and energy arrays
    z = np.linspace(zmin, zmax, num=nz, endpoint=True)  # [---]
    temperature = np.exp(np.linspace(np.log(tmin), np.log(tmax), num=ntemp, endpoint=True))  # [keV]
    # Middle point of the bin
    energy = np.linspace(emin, emax, num=nene, endpoint=False) + 0.5 * (emax - emin) / nene  # [keV]

    # Parameters setup through dictionary (common to all z and T)
    pars = {}
    # Metal abundance
    if isinstance(metal, float) or isinstance(metal, int):
        for ind in range(28):
            pars[4 + ind] = metal
    elif isinstance(metal, tuple) or isinstance(metal, list):
        for ind in range(28):
            pars[4 + ind] = metal[ind] if ind < len(metal) else 0
    pars[33] = 1.

    model = xsp.Model('vvapec', 'xraysim.specutils.apec_table', 0)

    table = np.ndarray([nz, ntemp, nene], dtype=sp)
    for index_z in range(nz):
        pars[32] = z[index_z]
        for index_t in range(ntemp):
            pars[1] = temperature[index_t]
            model.setPars(pars)
            if flag_ene:
                table[index_z, index_t, :] = np.array(model.values(0)) * energy  # [10^-14 keV s^-1 cm^3]
            else:
                table[index_z, index_t, :] = np.array(model.values(0))  # [10^-14 photons s^-1 cm^3]

    # Restoring PyXspec settings
    xsp.Xset.chatter = chatter_
    xsp.Xset.modelStrings = model_strings_
    xsp.Xset.abund = abund_
    xsp.AllModels.setEnergies("reset")  # Resets to the PyXspec default, not to the original

    result = {
        'data': table,
        'z': sp(z),
        'temperature': sp(temperature),
        'energy': sp(energy),
        'units': "[10^-14 keV s^-1 cm^3]" if flag_ene else "[10^-14 photons s^-1 cm^3]",
        'flag_ene': flag_ene,
        'model': "vvapec",
        'abund': abund,
        'tbroad': tbroad,
        'metallicity': metal,
        'temperature_units': "keV",
        'energy_units': "keV"
    }

    return result
