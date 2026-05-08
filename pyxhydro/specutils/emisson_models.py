import os
import json
import numpy as np
import xspec as xsp

# ── Anders & Grevesse solar abundance table (mass fractions) ──────────
ABUNDANCE_TABLE = {
    'Symbols': np.array(['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si',
                         'P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn']),
    'Values':  np.array([7.06534941e-01, 2.74100525e-01, 7.05343364e-11, 8.90682759e-11,
                         3.01565655e-09, 3.05603908e-03, 1.09960388e-03, 9.54323263e-03,
                         4.83378824e-07, 1.73980024e-03, 3.44846527e-05, 6.47369649e-04,
                         5.57916578e-05, 6.98837004e-04, 6.12236919e-06, 3.64042128e-04,
                         7.85193027e-06, 1.01642369e-04, 3.61744208e-06, 6.43301606e-05,
                         3.97037310e-08, 3.27796178e-06, 3.57066498e-07, 1.70564600e-05,
                         9.43435125e-06, 1.83190632e-03, 3.43680576e-06, 7.32283794e-05,
                         7.21566472e-07, 1.82390032e-06])
}

Z_SOLAR          = float(np.sum(ABUNDANCE_TABLE['Values'][2:]))
SYMBOLS_TO_IDX   = {s: i for i, s in enumerate(ABUNDANCE_TABLE['Symbols'])}
SYMBOLS_TO_SOLAR = dict(zip(ABUNDANCE_TABLE['Symbols'], ABUNDANCE_TABLE['Values']))
SYMBOLS_TO_SLOT  = {s: i + 2 for i, s in enumerate(ABUNDANCE_TABLE['Symbols'])}

# fixed cosmological values
H_PRIMORDIAL  = 0.7517
HE_PRIMORDIAL = 0.2453

H_SLOT        = 2
HE_SLOT       = 3
REDSHIFT_SLOT = 32
NORM_SLOT     = 33
METAL_SLOTS   = set(range(4, 32))   # Li(4) through Zn(31)

models_config_file = os.path.join(os.path.dirname(__file__), 'em_reference.json')
with open(models_config_file) as file:
    json_data = json.load(file)


class EmissionModel:

    def __init__(self, energy: np.ndarray, sim_config: str) -> None:
        """
        :param energy:     energy bin edges in keV, shape (n_bins+1,)
        :param sim_config: simulation name matching 'name' in em_reference.json
        """
        self.energy      = energy
        self.json_record = next((i for i in json_data if i['name'] == sim_config), None)

        if self.json_record is None:
            raise ValueError(f"Model '{sim_config}' not found in em_reference.json.")

        # ensure 'tracked' key always exists (GADGET-X has none)
        self.json_record.setdefault('tracked', [])

        self._validate_config()

        # { element_symbol -> pz_column_index }
        self._tracked_elements = {
            e['name']: e['index']
            for e in self.json_record['tracked']
            if e['type'] == 'element'
        }

        # untracked_fill descriptor
        self._fill = self.json_record['untracked_fill']

        # metal slots not covered by tracked elements
        tracked_slots         = {SYMBOLS_TO_SLOT[s] for s in self._tracked_elements}
        self._untracked_slots = METAL_SLOTS - tracked_slots

        # xspec setup
        xsp.Xset.chatter = 0
        xsp.AllModels.setEnergies(f"{energy.min()} {energy.max()} {len(energy) - 1} lin")
        for cmd in self.json_record.get('xset', []):
            if cmd['method'] == 'abund':
                xsp.Xset.abund = cmd['arg']
            elif cmd['method'] == 'addModelString':
                xsp.Xset.addModelString(cmd['arg'][0], cmd['arg'][1])

        self.model = xsp.Model("vvapec")

    # ------------------------------------------------------------------
    def _validate_config(self) -> None:
        """
        Validates the model configuration and raises errors if something is wrong.
        :return: None
        """
        forbidden    = {'H', 'He'}
        tracked      = self.json_record['tracked']
        fill         = self.json_record['untracked_fill']
        seen_names   = set()
        seen_indices = set()

        for e in tracked:
            for key in ('index', 'type', 'name'):
                if key not in e:
                    raise ValueError(f"Tracked entry missing '{key}'.")
            if e['type'] != 'element':
                raise ValueError("Tracked entries must have type='element'.")
            if e['name'] not in SYMBOLS_TO_IDX:
                raise ValueError(f"Unknown element '{e['name']}'.")
            if e['name'] in forbidden:
                raise ValueError(f"'{e['name']}' is fixed to primordial — mark as 'ignored'.")
            if e['name'] in seen_names:
                raise ValueError(f"Duplicate tracked element '{e['name']}'.")
            if e['index'] in seen_indices:
                raise ValueError(f"Duplicate tracked index '{e['index']}'.")
            seen_names.add(e['name'])
            seen_indices.add(e['index'])

        for key in ('type', 'name', 'index'):
            if key not in fill:
                raise ValueError(f"untracked_fill missing '{key}'.")

        if fill['type'] == 'total':
            if fill['name'] != 'Z':
                raise ValueError("Total fill must have name='Z'.")

        elif fill['type'] == 'element':
            if fill['name'] not in SYMBOLS_TO_IDX:
                raise ValueError(f"Unknown fill element '{fill['name']}'.")
            if fill['name'] in forbidden:
                raise ValueError(f"'{fill['name']}' cannot be used as fill.")
            if fill['name'] not in seen_names:
                raise ValueError(f"Fill element '{fill['name']}' must also appear in tracked.")
        else:
            raise ValueError(f"Unknown untracked_fill type '{fill['type']}'.")

    # ------------------------------------------------------------------
    def _build_params(self, pz: np.ndarray) -> dict:
        """
        TODO Change this in order to get as an output a 28 element array with the indexes where each element has to be
        TODO found in the simulation.
        TODO The input should be something like this  "new": {"C": 2, "N": 2, "O": 2, ..., "Co": -1, "other": "total"}
  }

        Map one particle array to a parameter dict for metals.
        H  (fixed)
        He (fixed)
        tracked metals   → from gas simulations, normalised to angr solar values
        untracked metals → either by total metallicity or some metal, such as Fe

        :param pz: 1-D metallicity array for one particle, [Z] for gadgetX or [Z,.....] for multi-species
        :return:   {vvapec_param_slot: solar_normalised_value}
        """
        all_indices  = [e['index'] for e in self.json_record['tracked']]
        all_indices += [self._fill['index']]
        n_expected   = max(all_indices) + 1

        if len(pz) < n_expected:
            raise ValueError(
                f"[{self.json_record['name']}] pz too short: "
                f"need at least {n_expected} columns, got {len(pz)}."
            )

        params = {}

        # H and He:
        params[H_SLOT]  = H_PRIMORDIAL  / SYMBOLS_TO_SOLAR['H']
        params[HE_SLOT] = HE_PRIMORDIAL / SYMBOLS_TO_SOLAR['He']

        # tracked metals
        for sym, col in self._tracked_elements.items():
            params[SYMBOLS_TO_SLOT[sym]] = pz[col] / SYMBOLS_TO_SOLAR[sym]

        # untracked metals
        if self._fill['type'] == 'total':
            fill_ratio = pz[self._fill['index']] / Z_SOLAR
        else:
            sym        = self._fill['name']
            fill_ratio = pz[self._fill['index']] / SYMBOLS_TO_SOLAR[sym]

        for slot in self._untracked_slots:
            params[slot] = fill_ratio

        return params

    # ------------------------------------------------------------------
    def calculate_spectrum(
            self,
            redshift:    float,
            temperature: float,
            pz:          np.ndarray,
            norm:        float,
            flag_ene:    bool = False,
    ) -> np.ndarray:
        """
        :param redshift:    cosmological redshift
        :param temperature: temperature in keV
        :param pz:          P['z'][i, :] metallicity array for particle i
        :param norm:        PyXspec normalisation (10^-14 cm^-5)
        :param flag_ene:    if True → energy flux (multiply by bin centres)
        :return:            array containing the spectrum. With standard Xspec parameters the units are
            [10^-14 photons s^-1 cm^3] or [10^-14 keV s^-1 cm^3] if flag_ene is set to True.
        """
        params                = self._build_params(pz)
        params[1]             = temperature
        params[REDSHIFT_SLOT] = redshift
        params[NORM_SLOT]     = norm
        params                = {k: np.float64(v) for k, v in params.items()}

        self.model.setPars(params)
        result = np.array(self.model.values(0), dtype=np.float32)

        if flag_ene:
            bin_centres = 0.5 * (self.energy[1:] + self.energy[:-1])
            result     *= bin_centres

        return result

    # ------------------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for f in os.listdir(os.getcwd()):
            if f.endswith('.dum'):
                os.remove(f)




############################################# TEST ####################################################################
# energy_bins = np.linspace(0.1, 2.4, 101)
#
# # GADGET-X — single total metallicity
# T      = 9.0404          # keV
# Z      = np.array([0.000192149])   # total metallicity (mass fraction)
#
# model_gadget = EmissionModel(energy_bins, 'GADGET-X')
# spectrum_gadget = model_gadget.calculate_spectrum(redshift=0.2, temperature=T, pz=Z, norm= 1.0)
# print("GADGET-X spectrum:", spectrum_gadget)
#
# # GIZMO-SIMBA — 11 species
# simba = np.array([
#    0.000192149,   # index 0 — Z_total
#    2.44e-01,      # index 1 — He
#    3.06e-05,      # index 2 — C
#    1.10e-05,      # index 3 — N
#    9.54e-05,      # index 4 — O
#    1.74e-05,      # index 5 — Ne
#    6.47e-06,      # index 6 — Mg
#    6.99e-06,      # index 7 — Si
#    3.64e-06,      # index 8 — S
#    6.43e-07,      # index 9 — Ca
#    1.83e-05,      # index 10 — Fe
# ])
#
# model_simba = EmissionModel(energy_bins, 'GIZMO-SIMBA')
# spectrum_simba = model_simba.calculate_spectrum(
#    redshift    = 0.2,
#    temperature = T,
#    pz          = simba,
#    norm        = 1.0,
# )
# print("\nGIZMO-SIMBA spectrum:", spectrum_simba)
# import matplotlib.pyplot as plt
#
# bin_centres = 0.5 * (energy_bins[1:] + energy_bins[:-1])
#
# fig, ax = plt.subplots(figsize=(8, 5))
# ax.plot(bin_centres, spectrum_gadget, label='GADGET-X (single Z)',  lw=1.5)
# ax.plot(bin_centres, spectrum_simba,  label='GIZMO-SIMBA (9 metals)', lw=1.5, ls='--')
# ax.set_xlabel('Energy (keV)')
# ax.set_ylabel(r'Flux [$10^{-14}$ photons s$^{-1}$ cm$^{-2}$]')
# ax.set_title(f'vvapec  |  T = {T} keV  |  z = 0.2')
# ax.set_yscale('log')
# plt.xscale('log')
# ax.legend()
# ax.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()
