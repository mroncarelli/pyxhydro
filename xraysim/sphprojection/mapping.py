import numpy as np
from astropy.io import fits
from astropy import cosmology
import pygadgetreader as pygr
from tqdm import tqdm

from xraysim.gadgetutils.phys_const import Xp, m_p, Msun2g, kpc2cm
from xraysim.gadgetutils.readspecial import readtemperature, readvelocity
from xraysim.gadgetutils import convert, phys_const
from xraysim.sphprojection.kernel import intkernel, map2d_loop, map2d_loop2, speccube_loop, \
    map2d_alpha_weight_loop, map2d_alpha_weight_loop2
from xraysim.sphprojection.linkedlist import linkedlist2d
from xraysim.specutils import tables, absorption

intkernel_vec = np.vectorize(intkernel)

# PyGadgetReader options
pygro = {'units': 0, 'suppress': 1}

# Float data types
SP = np.float32
DP = np.float64


def get_proj_index(proj: str) -> int:
    """
    Returns the index corresponding to the projection direction
    :param proj: (str) Projection direction, either 'x', 'y' or 'z'.
    :return: (int) The corresponding index, either 0, 1 or 2
    """
    if proj == 'x' or proj == 0:
        return 0
    elif proj == 'y' or proj == 1:
        return 1
    elif proj == 'z' or proj == 2:
        return 2
    else:
        raise ValueError("Invalid projection axis: ", proj, "Choose between 'x' (or 0), 'y' (1) and 'z' (2)")


def get_map_coord(simfile: str, proj_index: int, z=False):
    """
    Reads the 2-d coordinates to map from a Gadget snapshot.
    :param simfile: (str) Snapshot file.
    :param proj_index: (int) Projection index, either 0, 1 or 2.
    :param z: (bool) If set to True the output includes the also z-coordinate. Default: False.
    :return: A tuple of ndarrays containing the x and y-coordinates in [h^-1 kpc]. If z=True also the z-coordinate is
    included.
    """
    if proj_index == 0:
        index_list = [1, 2, 0]
    elif proj_index == 1:
        index_list = [2, 0, 1]
    else:
        index_list = [0, 1, 2]

    pos = pygr.readsnap(simfile, 'pos', 'gas', **pygro)  # [h^-1 kpc] comoving
    x = pos[:, index_list[0]]
    y = pos[:, index_list[1]]

    if z:
        return x, y, pos[:, index_list[2]]
    else:
        return x, y


def map2d(simfile: str, quantity: str, npix=256, alpha=0, center=None, size=None, proj='z', zrange=None, tcut=0,
          nsample=None, struct=False, nosmooth=False, progress=False):
    """
    Creates a projected 2D map of a physical quantitiy from a Gadget snapshot.
    :param simfile: (str) simulation file (Gadget)
    :param quantity: (str) physical quantity to map (one of rho, rho2, Tmw, Tew, Tsl, Taw, vmw, vew, vaw, wmw, wew, waw)
    :param npix: (int) number of map pixels per side
    :param alpha: (float) exponent of the temperature weight, w = n_e^2*T^alpha. Default: 0 (i.e. emission-weighted)
    :param center: (float 2) comoving coord. of the map center [h^-1 kpc]. Default: median point of gas particles
    :param size: (float) map comoving size [h^-1 kpc]. Default: encloses all gas particles
    :param proj: (str/int) direction of projection ('x', 'y', 'z' or 0, 1, 2)
    :param zrange: (float 2) range in the l.o.s. axis
    :param tcut: (float) if set defines a temperature cut below which particles are removed [K]. Default: 0.
    :param nsample: (int), if set defines a sampling for the particles (useful to speed up). Default: 1 (no sampling)
    :param struct: (bool) if set outputs a structure (dictionary) containing relevant info. Default: False
                    - norm: normalization map
                    - x(y)range: map range in the x(y) direction
                    - pixel_size: pixel size
                    - units: units of the map
                    - norm_units: units of the normalization map
                    - coord_units: units of x(y) range, i.e. h^-1 kpc comoving
                    - other info present for some specific options
    :param nosmooth: (bool) if set the SPH smoothing is turned off, and particles ar treated as points. Default: False
    :param progress: (bool) if set the progress bar is shown in output. Default: False
    :return: map or structure if struct keyword is set to True
    """

    # Initialization
    if nsample is None:
        nsample = 1

    quantity_ = quantity.lower()

    # Reading header variables
    ngas = pygr.readhead(simfile, 'gascount')
    f_cooling = pygr.readhead(simfile, 'f_cooling')

    # Reading positions of particles
    proj_index = get_proj_index(proj)
    if zrange:
        x, y, z = get_map_coord(simfile, proj_index, True)
    else:
        x, y = get_map_coord(simfile, proj_index)
        z = None

    # Reading smoothing length or assigning it to zero if smoothing is turned off
    hsml = np.full(ngas, 1.e-30, dtype=SP) if nosmooth else pygr.readsnap(simfile, 'hsml', 'gas',
                                                                          **pygro)  # [h^-1 kpc] comoving

    # Defining center and map size
    if center is None:
        xmin, xmax = min(x - hsml), max(x + hsml)
        ymin, ymax = min(y - hsml), max(y + hsml)
        xc = 0.5 * (xmin + xmax)
        yc = 0.5 * (ymin + ymax)
        if size is None:
            delta_x, delta_y = xmax - xmin, ymax - ymin
            if delta_x >= delta_y:
                size = delta_x
                xmap0, ymap0 = xmin, ymin - 0.5 * (delta_x - delta_y)
            else:
                size = delta_y
                xmap0, ymap0 = xmin - 0.5 * (delta_y - delta_x), ymin
        else:
            xmap0, ymap0 = xc - 0.5 * size, yc - 0.5 * size
    else:
        try:
            xc, yc = float(center[0]), float(center[1])
        except BaseException:
            raise ValueError("Invalid center: ", center, "Must be a 2d number vector")

        if size is None:
            xmin, xmax = min(x - hsml), max(x + hsml)
            ymin, ymax = min(y - hsml), max(y + hsml)
            if not (xmin <= xc <= xmax and ymin <= yc <= ymax):
                print("WARNING: Map center is outside the simulation box")
            size = 2. * max(abs(xc - xmin), abs(xc - xmax), abs(yc - ymin), abs(yc - ymax))

        xmap0, ymap0 = xc - 0.5 * size, yc - 0.5 * size

    pixsize = size / npix  # [h^-1 kpc] comoving

    # Normalizing coordinates in pixel units (0 = left/bottom border, npix = right/top border)
    x = (x - xmap0) / size * npix  # [pixel units]
    y = (y - ymap0) / size * npix  # [pixel units]
    hsml_z = hsml if zrange else None  # saving hsml in comoving coordinates [h^-1 kpc]
    hsml = hsml / size * npix  # [pixel units]

    # Reading temperature if necessary
    if tcut > 0 or quantity_ in ['tmw', 'tew', 'tsl', 'taw', 'vaw', 'waw']:
        temp = readtemperature(simfile, f_cooling=f_cooling, suppress=1)  # [K]

    # Cutting out particles outside the f.o.v. and for other conditions
    valid_mask = (x + hsml > 0) & (x - hsml < npix) & (y + hsml > 0) & (y - hsml < npix)
    if tcut > 0 or quantity_ in ['tmw', 'tew', 'tsl', 'taw', 'vaw', 'waw']:
        valid_mask = valid_mask & (temp > tcut)
        if quantity_ not in ['tmw', 'tew', 'tsl', 'taw', 'vaw', 'waw']:
            del temp

    if zrange:
        valid_mask = valid_mask & (z + hsml_z > zrange[0]) & (z - hsml_z < zrange[1])

    valid = np.where(valid_mask)[0]
    del valid_mask

    # Creating linked list
    particle_list = valid[linkedlist2d(x[valid], y[valid], npix, npix)]
    del valid

    # Calculating quantity (q) to integrate and weight (w)
    mass = pygr.readsnap(simfile, 'mass', 'gas', **pygro)  # [10^10 h^-1 M_Sun]
    if zrange:
        # If a l.o.s. range is defined I modify the particle mass according to the smoothing kernel
        for ipart in particle_list[::nsample]:
            mass[ipart] *= intkernel_vec((zrange[1] - z[ipart]) / hsml_z[ipart]) - intkernel_vec((zrange[0] - z[ipart]))
        del z, hsml_z

    multi_alpha = quantity_ in ['taw', 'vaw', 'waw'] and type(alpha) in [tuple, list, np.ndarray]
    if multi_alpha:
        alpha_arr = np.asarray(alpha, dtype=SP)
        nalpha = len(alpha_arr)

    # Case select for quantity to map
    if quantity_ == 'rho':  # Int(rho*dl)
        nrm = np.full(ngas, 0., dtype=SP)  # [---]
        qty = mass / pixsize ** 2  # comoving [10^10 h M_Sun kpc^-2]
    elif quantity_ == 'nh':  # Int(n_H*dl) after conversion factor is applied
        nrm = np.full(ngas, 0., dtype=SP)  # [---]
        qty = mass / pixsize ** 2  # comoving [10^10 h M_Sun kpc^-2]
        conv_factor = 1e10 * Msun2g * Xp / m_p / kpc2cm ** 2
    elif quantity_ == 'rho2':  # Int(rho2*dl)
        nrm = np.full(ngas, 0., dtype=SP)  # [---]
        qty = mass * pygr.readsnap(simfile, 'rho', 'gas',
                                   **pygro) / pixsize ** 2  # comoving [10^20 h^3 M_Sun^2 kpc^-5]
    elif quantity_ in ['ne', 'nenh', 'ne2', 'tmw', 'tew', 'tsl', 'taw', 'vmw', 'vew', 'vaw', 'wmw', 'wew', 'waw']:
        x_e = pygr.readsnap(simfile, 'ne', 'gas', **pygro)  # n_e / n_H [---]
        if quantity_ == 'ne':  # Int(ne*dl) after conversion factor is applied
            nrm = np.full(ngas, 0., dtype=SP)  # [---]
            qty = mass * x_e / pixsize ** 2  # comoving [10^10 h M_Sun kpc^-2]
            conv_factor = 1e10 * Msun2g * Xp / m_p / kpc2cm ** 2
        elif quantity_ == 'nenh':  # Int(ne*nH*dl) after conversion factor is applied
            nrm = np.full(ngas, 0., dtype=SP)
            qty = (mass * pygr.readsnap(simfile, 'rho', 'gas', **pygro) *
                   x_e / pixsize ** 2)  # comoving [10^20 h^3 M_Sun^2 kpc^-5]
            conv_factor = 1e20 * (Msun2g * Xp / m_p) ** 2 / kpc2cm ** 5
        elif quantity_ == 'ne2':  # Int(ne2*dl) after conversion factor is applied
            nrm = np.full(ngas, 0., dtype=SP)  # [---]
            qty = (mass * pygr.readsnap(simfile, 'rho', 'gas', **pygro) ** 2 *
                   x_e / pixsize ** 2)  # comoving [10^20 h^3 M_Sun^2 kpc^-5]
            conv_factor = 1e20 * (Msun2g * Xp / m_p) ** 2 / kpc2cm ** 5
        elif quantity_ in ['tmw', 'tew', 'tsl', 'taw']:
            if multi_alpha:
                nrm = mass * pygr.readsnap(simfile, 'rho', 'gas',
                                           **pygro) * x_e ** 2 / pixsize ** 2  # [10^20 h^3 M_Sun^2 kpc^-5]
                qty = temp  # [K]
            else:
                if quantity_ == 'tmw':  # Weighted by electron density: Int(n_e*T*dl) / Int(ne*dl)
                    nrm = mass * x_e / pixsize ** 2  # comoving [10^10 h M_Sun kpc^-2]
                elif quantity_ == 'tew':  # Weighted by electron density squared: Int(n_e^2*T*dl) / Int(ne^2*dl)
                    rho = pygr.readsnap(simfile, 'rho', 'gas', **pygro)  # [10^10 h^2 M_Sun kpc^-3]
                    nrm = mass * rho * x_e ** 2 / pixsize ** 2  # [10^20 h^3 M_Sun^2 kpc^-5]
                    del rho
                elif quantity_ == 'tsl':  # Spectroscopic-like: Int(n_e^2*T^0.25*dl) / Int(ne^2*T^-0.75*dl)
                    rho = pygr.readsnap(simfile, 'rho', 'gas', **pygro)  # [10^10 h^2 M_Sun kpc^-3]
                    nrm = mass * rho * x_e ** 2 * temp ** (-0.75) / pixsize ** 2  # [10^20 h^3 M_Sun^2 kpc^-5 K^-0.75]
                    del rho
                elif quantity_ == 'taw':  # Alpha-weighted: Int(n_e^2*T^(alpha+1)*dl) / Int(ne^2*T^alpha*dl)
                    rho = pygr.readsnap(simfile, 'rho', 'gas', **pygro)  # [10^10 h^2 M_Sun kpc^-3]
                    nrm = mass * rho * x_e ** 2 * temp ** alpha / pixsize ** 2  # [10^20 h^3 M_Sun^2 kpc^-5 K^alpha]
                    del rho
                qty = nrm * temp  # [nrm units] * [K]
                del temp
        elif quantity_ in ['vmw', 'vew', 'vaw']:
            vel = readvelocity(simfile, units='km/s', suppress=1)[:, proj_index]  # [km s^-1]
            if multi_alpha:
                nrm = mass * pygr.readsnap(simfile, 'rho', 'gas',
                                           **pygro) * x_e ** 2 / pixsize ** 2  # [10^20 h^3 M_Sun^2 kpc^-5]
                qty = vel  # [km s^-1]
            else:
                if quantity_ == 'vmw':  # Weighted by electron density: Int(n_e*v*dl) / Int(n_e*dl)
                    nrm = mass * x_e / pixsize ** 2  # [10^10 h M_Sun kpc^-2]
                elif quantity_ == 'vew':  # Weighted by electron density squared: Int(n_e^2*v*dl) / Int(n_e^2*dl)
                    rho = pygr.readsnap(simfile, 'rho', 'gas', **pygro)  # [10^10 h^2 M_Sun kpc^-3]
                    nrm = mass * rho * x_e ** 2 / pixsize ** 2  # [10^20 h^3 M_Sun^2 kpc^-5]
                    del rho
                elif quantity_ == 'vaw':  # Alpha-weighted: Int(n_e^2*T^alpha*v*dl) / Int(n_e^2*T^alpha*dl)
                    rho = pygr.readsnap(simfile, 'rho', 'gas', **pygro)  # [10^10 h^2 M_Sun kpc^-3]
                    nrm = mass * rho * x_e ** 2 * temp ** alpha / pixsize ** 2  # [10^20 h^3 M_Sun^2 kpc^-5 K^alpha]
                    del rho, temp
                qty = nrm * vel  # [nrm units] * [km s^-1]
            del mass, vel
        elif quantity_ in ['wmw', 'wew', 'waw']:
            vel = readvelocity(simfile, units='km/s', suppress=1)[:, proj_index]  # [km s^-1]
            if multi_alpha:
                nrm = mass * pygr.readsnap(simfile, 'rho', 'gas',
                                           **pygro) * x_e ** 2 / pixsize ** 2  # [10^20 h^3 M_Sun^2 kpc^-5]
                qty = vel ** 2  # [km^2 s^-2]
                qty2 = vel  # [km s^-1]
            else:
                if quantity_ == 'wmw':  # Weighted by electron density: Int(n_e*v^2*dl) / Int(n_e*dl) - v_mw^2
                    nrm = mass * x_e / pixsize ** 2  # [10^10 h M_Sun kpc^-2]
                elif quantity_ == 'wew':  # Weighted by electron density squared: Int(n_e^2*v^2*dl) / Int(n_e^2*dl) - v_ew^2
                    rho = pygr.readsnap(simfile, 'rho', 'gas', **pygro)  # [10^10 h^2 M_Sun kpc^-3]
                    nrm = mass * rho * x_e ** 2 / pixsize ** 2  # [10^10 h M_Sun kpc^-2]
                    del rho
                elif quantity_ == 'waw':  # Alpha-weighted: Int(n_e^2*T^alpha*v^2*dl) / Int(n_e^2*T^alpha*dl) - v_aw^2
                    rho = pygr.readsnap(simfile, 'rho', 'gas', **pygro)  # [10^10 h^2 M_Sun kpc^-3]
                    nrm = mass * rho * x_e ** 2 * temp ** alpha / pixsize ** 2  # [10^20 h^3 M_Sun^2 kpc^-5 K^alpha]
                    del rho
                qty = nrm * vel ** 2  # [nrm units] * [km^2 s^-2]
                qty2 = nrm * vel  # [nrm units] * [km s^-1]
            del mass, vel
        del x_e
    else:
        raise ValueError("Invalid mapping quantity: " + quantity +
                         " . Must be one of 'rho', 'ne', 'nh', 'rho2', 'nenH', 'Tmw', 'Tew', 'Tsl', 'Taw', 'vmw', "
                         "'vew', 'vaw', 'wmw', 'wew' 'waw'")

    # Mapping
    if multi_alpha:
        qty_map = np.full((npix, npix, nalpha), 0., dtype=DP)
        nrm_map = np.full((npix, npix, nalpha), 0., dtype=DP)
    else:
        qty_map = np.full((npix, npix), 0., dtype=DP)
        nrm_map = np.full((npix, npix), 0., dtype=DP)

    iter_ = tqdm(particle_list[::nsample]) if progress else particle_list[::nsample]

    if quantity_ in ['wmw', 'wew', 'waw']:
        if multi_alpha:
            qty2_map = np.full((npix, npix, nalpha), 0., dtype=DP)
            map2d_alpha_weight_loop2(qty_map, qty2_map, nrm_map, iter_, x, y, hsml, qty, qty2, nrm, temp, alpha_arr)
        else:
            qty2_map = np.full((npix, npix), 0., dtype=DP)
            map2d_loop2(qty_map, qty2_map, nrm_map, iter_, x, y, hsml, qty, qty2, nrm)
    else:
        if multi_alpha:
            map2d_alpha_weight_loop(qty_map, nrm_map, iter_, x, y, hsml, qty, nrm, temp, alpha_arr)
        else:
            map2d_loop(qty_map, nrm_map, iter_, x, y, hsml, qty, nrm)
    if quantity_ in ['ne', 'nh', 'nenh', 'ne2']:
        qty_map *= conv_factor

    qty_map[np.where(nrm_map != 0.)] /= nrm_map[np.where(nrm_map != 0.)]
    if quantity_ in ['wmw', 'wew', 'waw']:
        qty2_map[np.where(nrm_map != 0.)] /= nrm_map[np.where(nrm_map != 0.)]
        # Numerical noise may cause some pixels of qty_map to have smaller values than the corresponding ones, squared,
        # in qty2_map: this would cause the presence of nan in the result. The loops below puts 0 in those pixels.
        if multi_alpha:
            for ipix in range(npix):
                for jpix in range(npix):
                    for k in range(nalpha):
                        qty_map[ipix, jpix, k] = np.sqrt(max(qty_map[ipix, jpix, k] - qty2_map[ipix, jpix, k] ** 2, 0.))
        else:
            for ipix in range(npix):
                for jpix in range(npix):
                    qty_map[ipix, jpix] = np.sqrt(max(qty_map[ipix, jpix] - qty2_map[ipix, jpix] ** 2, 0.))


    # Conversion to float32 for output
    qty_map = SP(qty_map)
    nrm_map = SP(nrm_map)
    if quantity_ in ['wmw', 'wew', 'waw']:
        qty2_map = SP(qty2_map)

    # Output
    if struct:

        units = {
            'rho': {'map': '10^10 h M_Sun kpc^-2', 'norm': '---'},
            'ne': {'map': 'h cm^-2', 'norm': '---'},
            'nh': {'map': 'h cm^-2', 'norm': '---'},
            'rho2': {'map': '10^20 h^3 M_Sun^2 kpc^-5', 'norm': '---'},
            'nenh': {'map': 'h^3 cm^-5', 'norm': '---'},
            'ne2': {'map': 'h^3 cm^-5', 'norm': '---'},
            'tmw': {'map': 'K', 'norm': '10^10 h M_Sun kpc^-2'},
            'tew': {'map': 'K', 'norm': '10^20 h^3 M_Sun^2 kpc^-5'},
            'tsl': {'map': 'K', 'norm': '10^20 h^3 M_Sun^2 kpc^-5 K^-0.75'},
            'taw': {'map': 'K', 'norm': '10^20 h^3 M_Sun^2 kpc^-5 K^alpha'},
            'vmw': {'map': 'km s^-1', 'norm': '10^10 h M_Sun kpc^-2'},
            'vew': {'map': 'km s^-1', 'norm': '10^20 h^3 M_Sun^2 kpc^-5'},
            'vaw': {'map': 'km s^-1', 'norm': '10^20 h^3 M_Sun^2 kpc^-5 K^alpha'},
            'wmw': {'map': 'km s^-1', 'map2': 'km s^-1', 'norm': '10^10 h M_Sun kpc^-2'},
            'wew': {'map': 'km s^-1', 'map2': 'km s^-1', 'norm': '10^20 h^3 M_Sun^2 kpc^-5'},
            'waw': {'map': 'km s^-1', 'map2': 'km s^-1', 'norm': '10^20 h^3 M_Sun^2 kpc^-5 K^alpha'}
        }

        result = {
            'map': qty_map,
            'norm': nrm_map,
            'xrange': (xmap0, xmap0 + size),  # [h^-1 kpc] comoving
            'yrange': (ymap0, ymap0 + size),  # [h^-1 kpc] comoving
            'pixel_size': pixsize,  # [h^-1 kpc] comoving
            'units': units[quantity_]['map'],
            'norm_units': units[quantity_]['norm'],
            'coord_units': 'h^-1 kpc'
        }
        if quantity_ in ['taw', 'vaw', 'waw']:
            result['alpha'] = alpha
        if nosmooth:
            result['smoothing'] = 'OFF'
        if quantity_ in ['wmw', 'wew', 'waw']:
            result['map2'] = qty2_map
            result['map2_units'] = units[quantity_]['map2']
        if zrange:
            result['zrange'] = zrange  # [h^-1 kpc] comoving

        return result

    else:

        return qty_map


def specmap(snapfile: str, sptable, size: float, npix=256, redshift=None, center=None, proj='z', zrange=None,
            energy_cut=None, tcut=0., flag_ene=False, nsample=None, isothermal=None, novel=None, gaussvel=None,
            seed=0, nosmooth=False, nh=None, simulation_type=None, progress=False):
    """
    Creates a spectral map (spectral-cube) from a Gadget snapshot.
    :param snapfile: (str) Simulation snapshot file (Gadget)
    :param sptable: (dict or str) Spectrum table or spectrum table file (FITS)
    :param size: (float) Angular size of the map [deg]
    :param npix: (int) Number of pixels per map side. Default: 256.
    :param redshift: (float) Redshift where to place the simulation. Default: the redshift of the Gadget snapshot file
    :param center: (float 2) Comoving coord. of the map center [h^-1 kpc]. Default: median point of gas particles
    :param proj: (str/int) Direction of projection ('x', 'y', 'z' or 0, 1, 2)
    :param zrange: (float 2) Range (comoving) in the l.o.s. axis [h^-1 kpc]
    :param energy_cut: (float 2) Energy interval to compute [keV]. Default: assumes the one from the sptable.
    :param tcut: (float) If set defines a temperature cut below which particles are removed [K]. The temperature is the
        one from the simulation snapshot even if the isothermal keyword is active. Default: 0.
    :param flag_ene: (bool) If set to True forces the computation to be in energy units, i.e. [keV keV^-1 s^-1 cm^-2
        arcmin^-2], with False in photon units, i.e. [photons keV^-1 s^-1 cm^-2 arcmin^-2]. Default: False
    :param nsample: (int) If set defines a sampling for the particles (useful to speed up). Default: 1 (no sampling)
    :param isothermal: (float) If set gas is assumed isothermal gas with temperature fixed to the input value [K]. Does
        not apply to the temperature cut (see tcut). Default: the temperature is read from the Gadget file.
    :param novel: (bool) If set to True peculiar velocities are turned off. Default: False
    :param gaussvel: (float 2) If set velocities are set randomly with a gaussian pdf, with mean and standard deviation
        given by the first and second argument in [km/s]. If the second argument is not present it is considered to be
        0. Applies only if novel=False. Default: None.
    :param seed: (int) Seed for the random generator (used for gaussvel). Default: 0.
    :param nosmooth: (bool) If set the SPH smoothing is turned off, and particles ar treated as points. Default: False
    :param nh: (float) Hydrogen column density [10^22 cm^-2], overrides the value from the spectral table. Default:
        assumes the value in sofile.
    :param simulation_type: (str) The name of the simulation set of the snapshot file. Default: None.
    :param progress: (bool) If set the progress bar is shown in output. Default: False.
    :return: A structure (dictionary) containing some info, including:
                    - data: spectral cube [photons keV^-1 s^-1 cm^-2 arcmin^-2] (or [keV keV^-1 s^-1 cm^-2 arcmin^-2]
                        if flag_ene is True)
                    - x(y)range: map range in the x(y) direction in Gadget units [h^-1 kpc]
                    - size: map size [deg]
                    - pixel_size: pixel size [arcmin]
                    - energy: energy of the spectrum [keV]
                    - energy_interval: energy bin size of the spectrum [keV]
                    - units: units of the spectral cube contained in 'data'
                    - coord_units: units of the coordinates of the map, i.e. x(y)range
                    - energy_units: units of energy and energy_interval
    """

    # Initialization
    pixsize = size / npix * 60.  # [arcmin]

    if nsample is None:
        nsample = 1

    # Reading header variables
    if redshift is None:
        redshift = pygr.readhead(snapfile, 'redshift')
    h_hubble = pygr.readhead(snapfile, 'hubble')
    ngas = pygr.readhead(snapfile, 'gascount')
    f_cooling = pygr.readhead(snapfile, 'f_cooling')

    # Reading positions of particles
    proj_index = get_proj_index(proj)
    if zrange:
        x, y, z = get_map_coord(snapfile, proj_index, True)  # [h^-1 kpc] comoving
    else:
        x, y = get_map_coord(snapfile, proj_index)  # [h^-1 kpc] comoving
        z = None

    # Reading smoothing length or assigning it to zero if smoothing is turned off
    hsml = np.full(ngas, 1.e-30, dtype=SP) if nosmooth else pygr.readsnap(snapfile, 'hsml', 'gas',
                                                                          **pygro)  # [h^-1 kpc] comoving

    # Geometry conversion
    cosmo = cosmology.FlatLambdaCDM(H0=100., Om0=0.3)
    gadget2deg = cosmo.arcsec_per_kpc_comoving(redshift).to_value() / 3600.  # 1 deg / 1 h^-1 kpc (comoving)
    size_gadget = size / gadget2deg  # [h^-1 kpc]

    # Defining center
    if center is None:
        xmin, xmax = min(x - hsml), max(x + hsml)  # [h^-1 kpc]
        ymin, ymax = min(y - hsml), max(y + hsml)  # [h^-1 kpc]
        xc = 0.5 * (xmin + xmax)  # [h^-1 kpc]
        yc = 0.5 * (ymin + ymax)  # [h^-1 kpc]
    else:
        try:
            xc, yc = float(center[0]), float(center[1])  # [h^-1 kpc]
        except BaseException:
            raise ValueError("Invalid center: ", center, "Must be a 2d number vector")

    xmap0, ymap0 = xc - 0.5 * size_gadget, yc - 0.5 * size_gadget  # [h^-1 kpc]

    # Normalizing coordinates in pixel units (0 = left/bottom border, npix = right/top border)
    x = (x - xmap0) / size_gadget * npix  # [pixel units]
    y = (y - ymap0) / size_gadget * npix  # [pixel units]
    hsml_z = hsml if zrange else None  # saving hsml in comoving coordinates [h^-1 kpc]
    hsml = hsml / size_gadget * npix  # [pixel units]

    # Reading temperature or assigning it to a single value if isothermal is set
    if isothermal:
        temp_kev = np.full(ngas, isothermal / phys_const.keV2K)  # [keV]
    else:
        temp_kev = readtemperature(snapfile, f_cooling=f_cooling, units='keV', suppress=1)  # [keV]

    # Cutting out particles outside the f.o.v.
    valid_mask = (x + hsml > 0) & (x - hsml < npix) & (y + hsml > 0) & (y - hsml < npix)

    # If tcut is set, cutting out particles with temperature lower than the limit. IMPORTANT: the cut on the simulation
    # temperature even in the isothermal case, to eliminate particles with extremely high densities that would dominate
    # the emission.
    if tcut > 0.:  # [K]
        valid_mask = valid_mask & (readtemperature(snapfile, f_cooling=f_cooling, suppress=1) > tcut)

    # If zrange is set, cutting out particles outside the l.o.s. range
    if zrange:
        valid_mask = valid_mask & (z + hsml_z > zrange[0]) & (z - hsml_z < zrange[1])

    valid = np.where(valid_mask)[0]
    del valid_mask

    # Creating linked list with valid particles only
    particle_list = valid[linkedlist2d(x[valid], y[valid], npix, npix)]
    del valid

    # Calculating quantity (q) to integrate and weight (w)
    mass = pygr.readsnap(snapfile, 'mass', 'gas', **pygro)  # [10^10 h^-1 M_Sun]
    if zrange:
        # If a l.o.s. range is defined I modify the particle mass according to the smoothing kernel
        for ipart in particle_list[::nsample]:
            mass[ipart] *= intkernel_vec((zrange[1] - z[ipart]) / hsml_z[ipart]) - intkernel_vec((zrange[0] - z[ipart]))
        del z, hsml_z

    # Calculating effective redshift (Hubble + peculiar velocity) of the particles
    if novel:
        # If peculiar velocities are switched off all particles effective redshift are set to the cosmological one
        z_eff = np.full(ngas, redshift)
    elif gaussvel:
        # Random gaussian velocity distribution
        try:
            v_mean = float(gaussvel[0])  # [km/s]
            v_std = float(gaussvel[1]) if len(gaussvel) > 1 else 0.  # [km/s]
            rng = np.random.default_rng(np.abs(seed))
            vel = rng.normal(loc=v_mean, scale=v_std, size=ngas)
            z_eff = convert.vpec2zobs(vel,  # [km/s]
                                      redshift,
                                      units='km/s')
            del vel
        except BaseException:
            raise ValueError("Invalid value for gaussvel: ", gaussvel, "Must be a 2d number vector")

    else:
        z_eff = convert.vpec2zobs(readvelocity(snapfile, units='km/s', suppress=1)[:, proj_index],  # [km/s]
                                  redshift,
                                  units='km/s')

    # Reading comoving density and converting to physical [10^10 h^2 M_Sun kpc^-3]
    rho = pygr.readsnap(snapfile, 'rho', 'gas', **pygro) * (1 + redshift) ** 3

    # Reading ionization fraction if f_cooling is on
    ne = pygr.readsnap(snapfile, 'ne', 'gas', **pygro) if f_cooling else None

    # Calculating Xspec normalization [10^14 cm^-5]
    norm = convert.gadget2xspecnorm(mass, rho, 1.e3 * cosmo.comoving_distance(z_eff).to_value(), h_hubble, ne)
    del mass, rho, ne

    # Reading emission table [10^-14 photons s^-1 cm^3]
    if type(sptable) == dict:
        spectable = sptable
    elif type(sptable) == str:
        spectable = tables.read_spectable(sptable, z_cut=(np.min(z_eff[particle_list]), np.max(z_eff[particle_list])),
                                          temperature_cut=(
                                              np.min(temp_kev[particle_list]), np.max(temp_kev[particle_list])),
                                          energy_cut=energy_cut)
    else:
        raise ValueError("Invalid sptable type: must be a string or dictionary")

    # In nh is provided the spectral table is converted
    if nh is not None:
        spectable = absorption.convert_nh(spectable, nh)

    energy = spectable.get('energy')  # [keV]
    nene = len(energy)
    # TODO: include d_ene while generating the table
    d_ene = (energy[-1] - energy[0]) / (nene - 1)  # [keV] Assuming uniform energy interval.

    # Converting photons to energy o viceversa, if necessary
    if flag_ene != spectable.get('flag_ene'):
        if flag_ene:
            for iene in range(0, nene):
                spectable['data'][:, :, iene] *= energy[iene]  # [10^-14 keV s^-1 cm^3]
        else:
            for iene in range(0, nene):
                spectable['data'][:, :, iene] /= energy[iene]  # [10^-14 photons s^-1 cm^3]

    # Defining iterable to iterate through particles
    iter_ = tqdm(particle_list[::nsample]) if progress else particle_list[::nsample]

    # Initializing spcube with double precision
    spcube = np.full((npix, npix, nene), 0., dtype=DP)

    # Cython loop for mapping
    speccube_loop(spcube, iter_, x, y, hsml, spectable, norm, z_eff, temp_kev)

    spcube /= d_ene * pixsize ** 2  # [photons s^-1 cm^-2 arcmin^-2 keV^-1]

    # Output
    result = {
        'data': SP(spcube),
        'simulation_file': snapfile,
        'spectral_table': sptable if type(sptable) == str else "(Computed)",
        'proj': proj,
        'z_cos': redshift,
        'd_c': 1e3 * cosmo.comoving_distance(redshift).to_value(),  # [h^-1 kpc]
        'xrange': (xmap0, xmap0 + size_gadget),  # [h^-1 kpc]
        'yrange': (ymap0, ymap0 + size_gadget),  # [h^-1 kpc]
        'size': SP(size),  # [deg]
        'size_units': 'deg',
        'pixel_size': SP(pixsize),  # [arcmin]
        'pixel_size_units': 'arcmin',
        'energy': SP(energy),
        'energy_interval': SP(np.full(nene, d_ene)),
        'units': 'keV keV^-1 s^-1 cm^-2 arcmin^-2' if flag_ene else 'photons keV^-1 s^-1 cm^-2 arcmin^-2',
        'coord_units': 'h^-1 kpc',
        'energy_units': 'keV',
        'flag_ene': flag_ene
    }
    if simulation_type:
        result['simulation_type']: simulation_type
    if tcut:
        result['tcut'] = tcut  # [K]
    if isothermal:
        result['isothermal'] = isothermal  # [K]
    result['smoothing'] = 'OFF' if nosmooth else 'ON'
    result['velocities'] = 'OFF' if novel else 'ON'
    if zrange:
        result['zrange'] = zrange  # [h^-1 kpc] comoving
    if nsample and nsample != 1:
        result['nsample'] = nsample
    if nh is not None:
        result['nh'] = nh  # [10^22 cm^-2]
        result['nh_units'] = '10^22 cm^-2'
    else:
        if 'nh' in spectable:
            result['nh'] = spectable.get('nh')  # [10^22 cm^-2]
            result['nh_units'] = '10^22 cm^-2'

    return result


def write_specmap(spec_cube: dict, outfile: str, overwrite=True):
    """
    Writes a spectral map (spectral-cube) into a FITS file.
    :param spec_cube: (dict) Spectral-cube, i.e. output of specmap.
    :param outfile: (str) FITS file.
    :param overwrite: (bool) If set to True the file is overwritten. Default: True.
    :return: System output of the writing operation (usually None)
    """
    hdulist = fits.HDUList()
    data = spec_cube.get('data')
    simulation_type = spec_cube.get('simulation_type')
    xrange = spec_cube.get('xrange')
    yrange = spec_cube.get('yrange')
    zrange = spec_cube.get('zrange')
    nh = spec_cube.get('nh')  # [10^22 cm^-2]
    nsample = spec_cube.get('nsample')

    # Primary
    hdulist.append(fits.PrimaryHDU(data.transpose()))
    hdulist[-1].header.set('INFO', 'Created with Python xraysim and astropy')
    if simulation_type:
        hdulist[-1].header.set('SIM_TYPE', simulation_type)
    hdulist[-1].header.set('SIM_FILE', spec_cube.get('simulation_file'))
    hdulist[-1].header.set('SP_FILE', spec_cube.get('spectral_table'))
    hdulist[-1].header.set('PROJ', spec_cube.get('proj'))
    hdulist[-1].header.set('Z_COS', spec_cube.get('z_cos'))
    hdulist[-1].header.set('D_C', spec_cube.get('d_c'), '[' + spec_cube.get('coord_units') + ']')
    if nsample:
        hdulist[-1].header.set('NSAMPLE', nsample)
    hdulist[-1].header.set('NPIX', data.shape[0])
    hdulist[-1].header.set('NENE', data.shape[2])
    hdulist[-1].header.set('ANG_PIX', spec_cube.get('pixel_size'), '[' + spec_cube.get('pixel_size_units') + ']')
    hdulist[-1].header.set('ANG_MAP', spec_cube.get('size'), '[' + spec_cube.get('size_units') + ']')
    hdulist[-1].header.set('E_MIN', spec_cube.get('energy')[0] - 0.5 * spec_cube.get('energy_interval')[0])
    hdulist[-1].header.set('E_MAX', spec_cube.get('energy')[-1] + 0.5 * spec_cube.get('energy_interval')[-1])
    hdulist[-1].header.set('FLAG_ENE', 1 if spec_cube.get('flag_ene') else 0)
    hdulist[-1].header.set('UNITS', '[' + spec_cube.get('units') + ']')
    hdulist[-1].header.set('X_MIN', xrange[0], '[' + spec_cube.get('coord_units') + ']')
    hdulist[-1].header.set('X_MAX', xrange[1], '[' + spec_cube.get('coord_units') + ']')
    hdulist[-1].header.set('Y_MIN', yrange[0], '[' + spec_cube.get('coord_units') + ']')
    hdulist[-1].header.set('Y_MAX', yrange[1], '[' + spec_cube.get('coord_units') + ']')
    if zrange:
        hdulist[-1].header.set('Z_MIN', zrange[0], '[' + spec_cube.get('coord_units') + ']')
        hdulist[-1].header.set('Z_MAX', zrange[1], '[' + spec_cube.get('coord_units') + ']')
    if spec_cube.get('tcut'):
        hdulist[-1].header.set('T_CUT', spec_cube.get('tcut'), '[K]')
    if spec_cube.get('isothermal'):
        hdulist[-1].header.set('ISO_T', spec_cube.get('isothermal'), '[K]')
    hdulist[-1].header.set('SMOOTH', spec_cube.get('smoothing'))
    hdulist[-1].header.set('VEL', spec_cube.get('velocities'))
    if nh:
        hdulist[-1].header.set('N_H', nh, '[' + spec_cube.get('nh_units') + ']')

    # Extension 1
    hdulist.append(fits.ImageHDU(spec_cube.get('energy'), name='Energy'))
    hdulist[-1].header.set('NENE', data.shape[2])
    hdulist[-1].header.set('UNITS', '[' + spec_cube.get('energy_units') + ']')

    # Extension 2
    hdulist.append(fits.ImageHDU(spec_cube.get('energy_interval'), name='En. interval'))
    hdulist[-1].header.set('NENE', data.shape[2])
    hdulist[-1].header.set('UNITS', '[' + spec_cube.get('energy_units') + ']')

    # Writing FITS file (returns None)
    return hdulist.writeto(outfile, overwrite=overwrite)


def read_specmap(infile: str):
    hdulist = fits.open(infile)
    header0 = hdulist[0].header
    header1 = hdulist[1].header
    result = {
        'data': hdulist[0].data.transpose(),
        'xrange': (SP(header0.get('X_MIN')), SP(header0.get('X_MAX'))),  # [h^-1 kpc]
        'yrange': (SP(header0.get('Y_MIN')), SP(header0.get('Y_MAX'))),  # [h^-1 kpc]
        'size': SP(header0.get('ANG_MAP')),  # [deg]
        'size_units': header0.comments['ANG_MAP'].replace('[', '').replace(']', ''),
        'pixel_size': SP(header0.get('ANG_PIX')),  # [arcmin]
        'pixel_size_units': header0.comments['ANG_PIX'].replace('[', '').replace(']', ''),
        'energy': hdulist[1].data,
        'energy_interval': hdulist[2].data,
        'units': header0.get('UNITS').replace('[', '').replace(']', ''),
        'coord_units': header0.comments['X_MIN'].replace('[', '').replace(']', ''),
        'energy_units': header1.get('UNITS').replace('[', '').replace(']', ''),
        'simulation_file': header0.get('SIM_FILE'),
        'spectral_table': header0.get('SP_FILE'),
        'proj': header0.get('PROJ'),
        'z_cos': header0.get('Z_COS'),
        'd_c': header0.get('D_C'),  # h^-1 Mpc
        'flag_ene': header0.get('FLAG_ENE') == 1
    }
    if 'SIM_TYPE' in header0:
        result['simulation_type'] = header0.get('SIM_TYPE')
    if 'T_CUT' in header0:
        result['tcut'] = header0.get('T_CUT')  # [K]
    if 'ISO_T' in header0:
        result['isothermal'] = header0.get('ISO_T')  # [K]
    result['smoothing'] = header0.get('SMOOTH')
    result['velocities'] = header0.get('VEL')
    if 'Z_MIN' in header0:
        result['zrange'] = (header0.get('Z_MIN'), header0.get('Z_MAX'))
    if 'N_H' in header0:
        result['nh'] = header0.get('N_H')
        result['nh_units'] = header0.comments['N_H'].replace('[', '').replace(']', '')

    return result
