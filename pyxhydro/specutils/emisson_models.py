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

    def __init__(self, energy: np.ndarray, sim_config: str, flag_ene: bool) -> None:
        """
        :param energy:     energy bin edges in keV, shape (n_bins+1)
        :param sim_config: simulation name matching 'name' in em_reference.json
        """
        self.energy      = energy
        self.json_record = next((i for i in json_data if i['name'] == sim_config), None)

        if self.json_record is None:
            raise ValueError(f"Model '{sim_config}' not found in em_reference.json.")

        self._validate_config()

        self._tracked_elements = {}

        self._untracked_elements = {}

        for i in self.json_record['metals']:
            if i!="other":
                self._tracked_elements.update({i:self.json_record['metals'][i]})

            if i == "other":
                if self.json_record['metals'][i]=="total":
                    #print("This",self.json_record['metals'][i])
                    self._untracked_elements.update({self.json_record['metals'][i]:0})
                else:
                    #print("Second",self.json_record['metals'][i])
                    self._untracked_elements.update(self.json_record['metals'][i])

        # print(self._tracked_elements)
        # print(self._untracked_elements)
        # print('total' in self._untracked_elements.keys())

        # metal slots not covered by tracked elements
        tracked_slots         = {SYMBOLS_TO_SLOT[s] for s in self._tracked_elements}
        # print(tracked_slots)
        self._untracked_slots = METAL_SLOTS - tracked_slots
        # print(self._untracked_slots)
        # xspec setup
        xsp.Xset.chatter = 0
        xsp.AllModels.setEnergies(f"{energy.min()} {energy.max()} {len(energy)} lin")
        for cmd in self.json_record.get('xset', []):
            if cmd['method'] == 'abund':
                xsp.Xset.abund = cmd['arg']
            elif cmd['method'] == 'addModelString':
                xsp.Xset.addModelString(cmd['arg'][0], cmd['arg'][1])

        self.model = xsp.Model("vvapec")
        self.flag_ene = flag_ene

    # ------------------------------------------------------------------
    def _validate_config(self) -> None:
        forbidden = {'H', 'He'}
        metals    = self.json_record.get('metals')

        if metals is None:
            raise ValueError(f"[{self.json_record['name']}] Missing 'metals' key.")

        if not isinstance(metals, dict):
            raise ValueError(f"[{self.json_record['name']}] 'metals' must be a dict.")

        if 'other' not in metals:
            raise ValueError(f"[{self.json_record['name']}] 'metals' must contain 'other' key.")

        seen_indices = set()

        for sym, val in metals.items():

            if sym == 'other':
                continue

            if sym not in SYMBOLS_TO_IDX:
                raise ValueError(
                    f"[{self.json_record['name']}] Unknown element '{sym}' in metals."
                )

            # ── H and He forbidden
            if sym in forbidden:
                raise ValueError(
                    f"[{self.json_record['name']}] '{sym}' is fixed at primordial "
                    f"— remove it from metals."
                )

            if not isinstance(val, int) or val < 0:
                raise ValueError(
                    f"[{self.json_record['name']}] '{sym}' index must be a "
                    f"non-negative integer, got '{val}'."
                )

            # ── no duplicate indices
            if val in seen_indices:
                raise ValueError(
                    f"[{self.json_record['name']}] Duplicate column index "
                    f"'{val}' for element '{sym}'."
                )

            seen_indices.add(val)

        # validate 'other'
        other = metals['other']

        if isinstance(other, str):
            # 'other': 'total'  — scale all untracked by total Z
            if other != 'total':
                raise ValueError(
                    f"[{self.json_record['name']}] 'other' string value must be "
                    f"'total', got '{other}'."
                )

        elif isinstance(other, dict):
            # 'other': {{'Fe': 10}}  — scale by a specific tracked element
            if len(other) != 1:
                raise ValueError(
                    f"[{self.json_record['name']}] 'other' dict must have exactly "
                    f"one entry, got {len(other)}."
                )

            sym, idx = next(iter(other.items()))

            if sym not in SYMBOLS_TO_IDX:
                raise ValueError(
                    f"[{self.json_record['name']}] Unknown element '{sym}' in 'other'."
                )

            if sym in forbidden:
                raise ValueError(
                    f"[{self.json_record['name']}] '{sym}' cannot be used in 'other'."
                )

            if not isinstance(idx, int) or idx < 0:
                raise ValueError(
                    f"[{self.json_record['name']}] '{sym}' index in 'other' must be "
                    f"a non-negative integer, got '{idx}'."
                )

            # the reference element must also appear in tracked
            if sym not in metals:
                raise ValueError(
                    f"[{self.json_record['name']}] 'other' reference element '{sym}' "
                    f"must also appear as a tracked element in metals."
                )

            # index must match
            if metals[sym] != idx:
                raise ValueError(
                    f"[{self.json_record['name']}] 'other' index {idx} for '{sym}' "
                    f"does not match tracked index {metals[sym]}."
                )

        else:
            raise ValueError(
                f"[{self.json_record['name']}] 'other' must be 'total' (string) "
                f"or a single-element dict like {{'Fe': 10}}, got '{other}'."
            )

    def _build_params(self, pZ: np.ndarray) -> dict:
        """
        Map one particle array to a parameter dict for metals.
        H  (fixed)
        He (fixed)
        tracked metals   → from gas simulations, normalised to angr solar values
        untracked metals → either by total metallicity or some metal, such as Fe

        :param pZ: 1-D metallicity array for one particle, [Z] for gadgetX or [Z,.....] for multi-species
        :return:   {vvapec_param_slot: solar_normalised_value}
        """

        required_index = max(
            list(self._tracked_elements.values()) +
            list(self._untracked_elements.values())
        )

        if len(pZ) <= required_index:
            raise ValueError(
                f"[{self.json_record['name']}] pZ too short: "
                f"highest required index is {required_index}, "
                f"but pZ has length {len(pZ)}."
            )

        fill_ratio = None

        # H and He: initialized to the bbn ratio
        params = {H_SLOT: H_PRIMORDIAL / SYMBOLS_TO_SOLAR['H'], HE_SLOT: HE_PRIMORDIAL / SYMBOLS_TO_SOLAR['He']}

        # tracked metals
        if len(self._tracked_elements)>0:
            for sym, col in self._tracked_elements.items():
                params[SYMBOLS_TO_SLOT[sym]] = pZ[col] / SYMBOLS_TO_SOLAR[sym]

        # untracked metals
        if 'total' in self._untracked_elements.keys():
            if len(self._tracked_elements.values())>0:
                test_pz = np.asarray(pZ, dtype=np.float64)
                idx = list(self._tracked_elements.values())
                tracked_total = np.sum(test_pz[idx])
                fill_ratio = (pZ[self._untracked_elements['total']] - tracked_total) / Z_SOLAR
                if fill_ratio<0:
                    fill_ratio = 0.0
            else:
                fill_ratio = (pZ[self._untracked_elements['total']]) / Z_SOLAR
        else:
            chem = next(iter(self._untracked_elements.keys()))
            sim_idx  = next(iter(self._untracked_elements.values()))
            fill_ratio = pZ[sim_idx] / SYMBOLS_TO_SOLAR[chem]

        for slot in self._untracked_slots:
            params[slot] = fill_ratio

        return params

    # ------------------------------------------------------------------
    def calculate_spectrum(
            self,
            redshift:    float,
            temperature: float,
            pZ:          np.ndarray,
            norm:        float,
    ) -> np.ndarray:
        """
        :param redshift:    cosmological redshift
        :param temperature: temperature in keV
        :param pZ:          P['z'][i, :] metallicity array for particle i
        :param norm:        PyXspec normalisation (10^-14 cm^-5)
        :return:            spectrum shape (n_bins,)
        """
        params                = self._build_params(pZ)
        params[1]             = temperature
        params[REDSHIFT_SLOT] = redshift
        params[NORM_SLOT]     = norm
        params                = {k: np.float64(v) for k, v in params.items()}

        self.model.setPars(params)

        result = np.array(self.model.values(0), dtype=np.float32)

        if self.flag_ene:
            bin_centres = 0.5 * (self.energy[1:] + self.energy[:-1])
            result     *= bin_centres

        return result

    def get(self, name):
        return getattr(self, name)

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
# model_gadget = EmissionModel(energy_bins, 'GADGET-X',False)
# spectrum_gadget = model_gadget.calculate_spectrum(redshift=0.2, temperature=T, pZ=Z, norm= 1.0)
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
# model_simba = EmissionModel(energy_bins, 'GIZMO-SIMBA',False)
# spectrum_simba = model_simba.calculate_spectrum(
#    redshift    = 0.2,
#    temperature = T,
#    pZ          = simba,
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
