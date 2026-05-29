"""
Set of methods connected to observational quantities.
"""

import numpy as np
from astropy.io import fits
from matplotlib.path import Path

from ..gadgetutils.phys_const import keV2erg
from .. import sixte
from ..sphprojection.mapping import read_specmap

SP = np.float32
DP = np.float64


def countrate(inp, arf, telescope=1, xrange=None, yrange=None, polygon=None, erange=None) -> float:
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
    :param polygon: (2D float array) Coordinates
    :param erange: (2 x float) Energy range [keV]. Default None.
    :return: (float) The expected countrate [cts s^-1].
    """

    def __inside_fov(points: np.ndarray, xrange=None, yrange=None, polygon=None) -> np.ndarray:
        """
        Determines if a set of points is inside and x,y-range and is inside a polygon of given vertices.
        :param points: (n x 2 array) The points coordinates.
        :return: (array of bool) True if the point respects all conditions, False otherwise.
        """
        n = points.shape[0]
        flag = np.full(n, True)
        if xrange is not None:
            flag *= ((points[:, 0] >= xrange[0]) & (points[:, 0] < xrange[1]))
        if yrange is not None:
            flag *= ((points[:, 1] >= yrange[0]) & (points[:, 1] < yrange[1]))
        if polygon is not None:
            path = Path(polygon)
            flag *= path.contains_points(points)

        return flag

    def e_sp_from_spmap(spmap: dict, xrange=None, yrange=None, polygon=None, erange=None) -> tuple:
        """
        Extracts the energy bins spectra from a spectral map.
        :param spmap: (dict) Spectral map.
        :param xrange: (2 x float) Range in the x-axis [deg]. Assumes 0 in the center. Default None.
        :param yrange: (2 x float) Range in the y-axis [deg]. Assumes 0 in the center. Default None.
        :param erange: (2 x float) Energy range [keV]. Default None.
        :return: (2 x float array) Central energy of the bins [keV] and total spectrum [photons s^-1 cm^-2].
        """

        energy = spmap["energy"]  # [keV]
        nx, ny, nene = spmap["data"].shape
        assert nx == ny  # assumes square regular map
        # Reshaping array to allow easier filtering
        data = spmap["data"].reshape(nx * ny, nene)
        d_ene = spmap["energy_interval"]  # [keV]

        if xrange is not None or yrange is not None or polygon is not None:
            size = spmap["size"]  # [deg]
            step = size / nx  # [arcmin]
            pvec = np.linspace(0.5 * (-size + step), 0.5 * (size - step), num=nx, endpoint=True)  # [deg]
            coords = np.ndarray([nx * ny, 2], dtype=SP)
            for ipix in range(nx):
                for jpix in range(ny):
                    coords[nx * ipix + jpix, 0] = pvec[ipix]
                    coords[nx * ipix + jpix, 1] = pvec[jpix]

            data = data[np.where(__inside_fov(coords, xrange=xrange, yrange=yrange, polygon=polygon))[0], :]
            del coords

        if erange is not None:
            index_ecut =  np.where((energy >= erange[0]) & (energy < erange[1]))[0]
            energy = energy[index_ecut]
            d_ene = d_ene[index_ecut]
            data = data[:, index_ecut]
            del index_ecut

        spectrum = np.sum(data, axis=0, dtype=DP)

        if spmap["flag_ene"]:
            spectrum /= energy  # [photons keV^-1 s^-1 cm^-2 arcmin^-2]

        spectrum *= d_ene * spmap["pixel_size"] ** 2  # [photons s^-1 cm^-2]

        return energy, spectrum  # [keV], [photons s^-1 cm^-2]


    def e_sp_from_simput(simput: fits.hdu.hdulist.HDUList, xrange=None, yrange=None, polygon=None,
                         erange=None) -> tuple:
        """
        Extracts the energy bins spectra from an input file HDUList. Assumes that the energy coordinate is the same for
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
        nsp, nene = data.shape
        for isp in range(nsp):
            data[isp, :] *= d_ene  # [photons s^-1 cm^-2]

        # Renormalization. This is actually not necessary if the Simput file has been created with the sixte module.
        flux = simput[1].data['FLUX']  # [erg s^-1 cm^-2]
        for isp in range(len(data)):
            flux0 = np.sum(data[isp] * energy) * keV2erg  # [erg s^-1 cm^-2]
            data[isp] *= flux[isp] / flux0  # [photons s^-1 cm^-2]
        del flux

        if xrange is not None or yrange is not None or polygon is not None:
            coords = np.ndarray([nsp, 2], dtype=SP)
            coords[:, 0] = simput[1].data['RA']  # [deg]
            coords[:, 1] = simput[1].data['DEC']  # [deg]

            data = data[np.where(__inside_fov(coords, xrange=xrange, yrange=yrange, polygon=polygon))[0], :]
            del coords

        if erange is not None:
            index_ecut =  np.where((energy >= erange[0]) & (energy < erange[1]))[0]
            energy = energy[index_ecut]
            data = data[:, index_ecut]
            del index_ecut

        spectrum = np.sum(data, axis=0, dtype=DP) # [photons s^-1 cm^-2]
        # nsp, nene = data.shape
        # for isp in range(nsp):
        #     spectrum += data[isp, :]  # [photons s^-1 cm^-2]

        return energy, spectrum  # [keV], [photons s^-1 cm^-2]

    # Checking input type and determining energy and spectrum based on it
    input_type = type(inp)
    if input_type == dict:
        # Assuming it's a specmap ([keV], [photons s^-1 cm^-2])
        energy, spectrum = e_sp_from_spmap(inp, xrange=xrange, yrange=yrange, polygon=polygon, erange=erange)
    elif input_type == fits.hdu.hdulist.HDUList:
        # Assuming it's a Simput HUDList ([keV], [photons s^-1 cm^-2])
        energy, spectrum = e_sp_from_simput(inp, xrange=xrange, yrange=yrange, polygon=polygon, erange=erange)
    elif input_type == str:
        try:
            # Trying with a file containing a specmap ([keV], [photons s^-1 cm^-2])
            energy, spectrum = e_sp_from_spmap(read_specmap(inp), xrange=xrange, yrange=yrange, polygon=polygon,
                                               erange=erange)
        except:
            # Trying with a Simput file ([keV], [photons s^-1 cm^-2])
            energy, spectrum = e_sp_from_simput(fits.open(inp), xrange=xrange, yrange=yrange, polygon=polygon,
                                                erange=erange)
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


def mosaic(n, center=(0, 0), side=1, theta=0, hexagon=False, layout='h', force_center=False) -> list:
    """
    Creates a square or hexagonal mosaic of pointings. For hexagon mosaic it uses odd horizontal or vertical layout.
    :param n: (int) Number of sides of the mosaic.
    :param center: (float x 2) Coordinate of the center of the mosaic [arbitrary units], default (0, 0).
    :param side: (float) Side of the square (or hexagon), default 1 [arbitrary units]
    :param theta: (float) Rotation angle, default 0 [deg]
    :param hexagon: (bool) If set to True creates a hexagonal mosaic, default False
    :param layout: (str) Tyling type for exagonal mosaic: 'h', i.e. horizontal (default), or 'v', i.e. vertical
    :param force_center: (bool) If set to True the coordinates are set to have the pointing with tag '00' contered in
        the center position, default False
    :return: (list of dict) List of pointings containing the following keys:
            - x: (float) x-coordinate of the pointing center
            - y: (float) y-coordinate of the pointing center
            - index: (int tuple) indexes of the 2D coordinates
            - ring: (int) ring index with the respect to the '00' pointing located in the center of the mosaic
            - tag: (str) a tag that identifies the pointing, being '00' the central pointing (rounded low/left when
                n is even) and with numbers 1, 2, 3, ... toward the up/right, and 9, 8, 7, ... towards the low/left.
                The uniqueness of the tag will fail for n > 10.
    """

    def __hex_distance(col1: int, row1: int, col2: int, row2: int) -> int:
        """
        Calculates the distance between two hexagons.
        :param col1:
        :param row1:
        :param col2:
        :param row2:
        :return:
        """
        def offset_to_cube(col, row):
            x = col - (row - (row & 1)) // 2
            z = row
            y = -x - z
            return x, y, z

        x1, y1, z1 = offset_to_cube(col1, row1)
        x2, y2, z2 = offset_to_cube(col2, row2)

        # Cube distance
        return max(
            abs(x1 - x2),
            abs(y1 - y2),
            abs(z1 - z2)
        )

    coord = np.linspace(-0.5 * (n - 1), 0.5 * (n - 1), n, endpoint=True, dtype=SP)
    zero_pixel = int(np.floor((n - 1) / 2))
    result = []

    if hexagon:
        side_spacing = np.sqrt(3)  # spacing in the direction of hex_tyling
        orth_spacing = 1.5  # spacing in the direction orthogonal to hex_tyling
        layout_ = layout.strip().lower()
        if layout_ == 'h':
            for i in range(n):
                for j in range(n):
                    result.append({'x': ((coord[i] + (j % 2) * 0.5) * side_spacing) * side,
                                   'y': coord[j] * orth_spacing * side,
                                   'ring': __hex_distance(i, j, zero_pixel, zero_pixel)})
        elif layout_ == 'v':
            for i in range(n):
                for j in range(n):
                    result.append({'x': coord[i] * orth_spacing * side,
                                   'y': ((coord[j] + (i % 2) * 0.5) * side_spacing) * side,
                                   'ring': __hex_distance(j, i, zero_pixel, zero_pixel)})
        else:
            raise ValueError("Invalid input type: hex_tyling must be 'h' or 'v'.")
    else:
        for i in range(n):
            for j in range(n):
                result.append({'x': coord[i] * side,
                               'y': coord[j] * side,
                               'ring': max(abs(i - zero_pixel), abs(j - zero_pixel))})

    # Adding index and tag (common to all cases)
    for i in range(n):
        for j in range(n):
            result[j + n * i]['index'] = (i, j)
            result[j + n * i]['tag'] = str((i - zero_pixel) % 10) + str((j - zero_pixel) % 10)

    # Centering in the '00' pixel if required
    if force_center:
        i00 = np.where(np.asarray([p['tag'] for p in result]) == '00')[0][0]
        xc, yc = result[i00]['x'], result[i00]['y']
        for item in result:
            item['x'] -= xc
            item['y'] -= yc

    # Applying rotation and offset
    theta_rad = np.deg2rad(theta)
    for item in result:
        x_ = item['x'] * np.cos(theta_rad) - item['y'] * np.sin(theta_rad) + center[0]
        y_ = item['x'] * np.sin(theta_rad) + item['y'] * np.cos(theta_rad) + center[1]
        item['x'] = x_
        item['y'] = y_

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
