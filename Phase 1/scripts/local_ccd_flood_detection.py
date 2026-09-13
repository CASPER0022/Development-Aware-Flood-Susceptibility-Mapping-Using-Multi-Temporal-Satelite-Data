import os
import sys
import json
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.stats import f as f_dist

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

script_dir = os.path.dirname(os.path.abspath(__file__))
phase1_dir = os.path.abspath(os.path.join(script_dir, ".."))
os.chdir(phase1_dir)

print("=" * 80)
print("LOCAL CCD-STYLE FLOOD DETECTION (CFAR RATIO-OF-LOCAL-MEANS TEST)")
print("Replaces the fixed global dB threshold with a per-pixel statistical")
print("significance test whose decision boundary adapts to local speckle/")
print("texture variance instead of applying one cutoff everywhere.")
print("=" * 80)

BBOX = [76.15, 9.90, 76.55, 10.25]
lon_min, lat_min, lon_max, lat_max = BBOX
calc_area_sqkm = 1696.64

WINDOW = 7          # local averaging window (pixels) -- both speckle suppression and the test's sample size
PFA = 1e-3          # target false-alarm probability per tail (CFAR design parameter)

# -------------------------------------------------------------------------
# Load UNFILTERED, registered linear intensities (see test_urban_double_bounce.py
# for why the 3x3 median filter is intentionally NOT used as input here: this
# test's own windowed averaging is the speckle suppression step, and
# pre-smoothing with a median filter would distort the Gamma-distributed
# multiplicative-speckle assumption the F-test below relies on).
# -------------------------------------------------------------------------
dry_dn = np.load("data/raw/sentinel1_2018_dry_dn_raw.npy").astype(np.float64)
flood_dn = np.load("data/raw/sentinel1_2018_flood_dn_raw_registered.npy").astype(np.float64)
builtup_mask = np.load("data/raw/builtup_mask_10m.npy")

valid_mask = (dry_dn > 10.0) & (flood_dn > 10.0)
print(f"[+] Loaded raw registered intensities: {dry_dn.shape}, valid pixels={np.sum(valid_mask)}")

# -------------------------------------------------------------------------
# Estimate the Equivalent Number of Looks (ENL) PER LAND-COVER CLASS, not as
# one global constant.
#
# A first version of this test estimated ENL once, from deep interior
# permanent water only (mean²/var = 23.7), and applied that single value
# everywhere. That reproduced the exact same mistake the PIF radiometric
# correction made earlier in this pipeline: it treated one unusually
# homogeneous reference target's noise level as representative of every
# land-cover class. Built-up land has real, non-speckle structural texture
# (rooftops, walls, layover, shadow) that adds variance beyond pure
# multiplicative speckle -- so its true mean²/var is measurably lower than
# calm water's, and using water's ENL there made the significance test far
# too sensitive, flagging most of the built-up footprint as "change." This
# is the same failure mode, one level down the pipeline: a homogeneous-
# reference assumption applied to a heterogeneous target class.
#
# Fix: estimate ENL separately for built-up (WorldCover class 50) and for
# open/vegetated land (everything else, excluding the dry-scene permanent
# water mask), and test each pixel against its own class's empirical noise
# level. Permanent water pixels are excluded from testing entirely (they're
# already water; the fixed-threshold pipeline defines them the same way).
# -------------------------------------------------------------------------
dry_db_tmp = 20.0 * np.log10(np.maximum(dry_dn, 1.0)) - 55.0
perm_water_raw = (dry_db_tmp <= -15.5) & valid_mask

wc_path = "data/raw/esa_worldcover_2021_resampled.npy"
worldcover = np.load(wc_path)
builtup_class = (worldcover == 50) & valid_mask & (~perm_water_raw)
open_class = valid_mask & (~builtup_class) & (~perm_water_raw)

def estimate_enl(mask, label):
    vals = dry_dn[mask]
    enl_val = float((np.mean(vals) ** 2) / np.var(vals))
    print(f"    ENL[{label:9s}] = {enl_val:.3f}  (n={np.sum(mask)})")
    return enl_val

print("[+] Empirically estimated ENL per land-cover group (mean²/var of raw linear intensity):")
enl_builtup = estimate_enl(builtup_class, "built-up")
enl_open = estimate_enl(open_class, "open-land")
print("    NOTE: built-up ENL is measured over the whole class, including any pixels")
print("    genuinely flooded at acquisition time -- that biases it lower (more")
print("    conservative test), not higher, so this approximation errs on the side")
print("    of under-, not over-, detection.")

# -------------------------------------------------------------------------
# Local windowed means (this is the speckle-reduction step for this test)
# -------------------------------------------------------------------------
print(f"[+] Computing {WINDOW}x{WINDOW} local windowed means...")
local_mean_dry = uniform_filter(dry_dn, size=WINDOW, mode='reflect')
local_mean_flood = uniform_filter(flood_dn, size=WINDOW, mode='reflect')

N = WINDOW * WINDOW

# Ratio of two independent, identically-shaped Gamma-distributed window means
# (same true sigma0, no change) follows F(2*N*L, 2*N*L). See e.g. Bruzzone &
# Bovolo-style CFAR SAR change detectors for the same construction. L (and so
# the critical ratio) is now assigned PER PIXEL from its own land-cover
# group's empirical ENL, not one global value.
ratio = local_mean_flood / np.maximum(local_mean_dry, 1e-6)

dof_builtup = 2.0 * N * enl_builtup
dof_open = 2.0 * N * enl_open
upper_crit_builtup = f_dist.ppf(1.0 - PFA, dof_builtup, dof_builtup)
lower_crit_builtup = f_dist.ppf(PFA, dof_builtup, dof_builtup)
upper_crit_open = f_dist.ppf(1.0 - PFA, dof_open, dof_open)
lower_crit_open = f_dist.ppf(PFA, dof_open, dof_open)

print(f"[+] Testing H0: no change, at Pfa={PFA:.1e} per tail, with class-specific degrees of freedom:")
print(f"    Built-up  (dof={dof_builtup:.1f}): flag increase if ratio > {upper_crit_builtup:.4f}, "
      f"decrease if ratio < {lower_crit_builtup:.4f}")
print(f"    Open-land (dof={dof_open:.1f}): flag increase if ratio > {upper_crit_open:.4f}, "
      f"decrease if ratio < {lower_crit_open:.4f}")

# -------------------------------------------------------------------------
# Apply the same land-cover restriction as the legacy method, so this is an
# apples-to-apples comparison of "adaptive local test" vs "fixed global
# threshold" and not also a change in what land cover is eligible.
# -------------------------------------------------------------------------
perm_water = perm_water_raw
urban_zone = builtup_mask & valid_mask & (~perm_water)
open_zone = valid_mask & (~urban_zone) & (~perm_water)

increase_sig = np.zeros_like(ratio, dtype=bool)
decrease_sig = np.zeros_like(ratio, dtype=bool)
increase_sig[urban_zone] = ratio[urban_zone] > upper_crit_builtup
decrease_sig[urban_zone] = ratio[urban_zone] < lower_crit_builtup
increase_sig[open_zone] = ratio[open_zone] > upper_crit_open
decrease_sig[open_zone] = ratio[open_zone] < lower_crit_open

open_flood_ccd = decrease_sig & open_zone
urban_double_bounce_ccd = increase_sig & urban_zone
total_event_flood_ccd = open_flood_ccd | urban_double_bounce_ccd

total_valid = np.sum(valid_mask)
perm_sqkm = round((np.sum(perm_water) / total_valid) * calc_area_sqkm, 2)
open_sqkm = round((np.sum(open_flood_ccd) / total_valid) * calc_area_sqkm, 2)
urban_sqkm = round((np.sum(urban_double_bounce_ccd) / total_valid) * calc_area_sqkm, 2)
total_sqkm = round((np.sum(total_event_flood_ccd) / total_valid) * calc_area_sqkm, 2)
builtup_sqkm = round((np.sum(urban_zone) / total_valid) * calc_area_sqkm, 2)
urban_pct_of_builtup = round((urban_sqkm / builtup_sqkm) * 100.0, 2) if builtup_sqkm > 0 else 0.0

print("\n" + "=" * 80)
print("     LOCAL CCD-STYLE FLOOD DETECTION RESULTS")
print("=" * 80)
print(f"  • Permanent water                                : {perm_sqkm} sq km")
print(f"  • Total built-up land                             : {builtup_sqkm} sq km")
print(f"  • Open-land specular attenuation (CCD-significant): {open_sqkm} sq km")
print(f"  • Urban double-bounce surge (CCD-significant)     : {urban_sqkm} sq km")
print(f"    -> as % of total built-up land in AOI           : {urban_pct_of_builtup}%")
print(f"  • TOTAL event flood footprint (CCD)               : {total_sqkm} sq km "
      f"({(total_sqkm/calc_area_sqkm)*100:.2f}%)")
print("=" * 80)

# -------------------------------------------------------------------------
# Same hotspot-vs-elsewhere check used in the raw delta_db diagnostic, now
# run against the CCD detection instead of eyeballing the map.
# -------------------------------------------------------------------------
h, w = dry_dn.shape
lat_axis = np.linspace(lat_max, lat_min, h)
lon_axis = np.linspace(lon_min, lon_max, w)
lon_grid, lat_grid = np.meshgrid(lon_axis, lat_axis)

landmarks = [
    {"name": "Aluva", "lat": 10.1076, "lon": 76.3516},
    {"name": "Kalamassery", "lat": 10.0528, "lon": 76.3264},
    {"name": "N. Kochi", "lat": 10.0245, "lon": 76.3078},
    {"name": "COK Airport", "lat": 10.1520, "lon": 76.4019},
    {"name": "Eloor", "lat": 10.0805, "lon": 76.2990},
]

print("  Hotspot check (built-up double-bounce detection rate, near vs elsewhere):")
for lm in landmarks:
    dist_deg = np.hypot(lon_grid - lm["lon"], lat_grid - lm["lat"])
    near_bu = urban_zone & (dist_deg < 0.02)
    far_bu = urban_zone & (dist_deg >= 0.05) & (dist_deg < 0.15)
    near_rate = round((np.sum(urban_double_bounce_ccd & near_bu) / np.sum(near_bu)) * 100.0, 2) if np.sum(near_bu) > 50 else float('nan')
    far_rate = round((np.sum(urban_double_bounce_ccd & far_bu) / np.sum(far_bu)) * 100.0, 2) if np.sum(far_bu) > 50 else float('nan')
    print(f"    • {lm['name']:14s}: near-hotspot={near_rate}%  elsewhere={far_rate}%")
print("=" * 80)

# Save outputs
np.save("data/raw/open_flood_mask_2018_ccd.npy", open_flood_ccd)
np.save("data/raw/urban_double_bounce_mask_2018_ccd.npy", urban_double_bounce_ccd)
np.save("data/raw/new_flood_inundation_mask_2018_ccd.npy", total_event_flood_ccd)
np.save("data/raw/ccd_ratio.npy", ratio)

ccd_diagnostics = {
    "method": "CFAR local ratio-of-means test: ratio of WINDOWxWINDOW local means "
              "(flood/dry) of raw registered linear intensity, tested against "
              "F(2*N*L, 2*N*L) under H0 of no change (N=window pixel count, L=ENL). "
              "L is class-adaptive (built-up vs open-land), not a single global value "
              "-- see enl_per_class and the note below.",
    "window_size_px": WINDOW,
    "enl_per_class": {
        "built_up": round(enl_builtup, 4),
        "open_land": round(enl_open, 4),
    },
    "enl_estimation_method": "mean^2 / variance of raw linear intensity over each "
                              "WorldCover-defined land-cover group in the dry scene",
    "enl_adaptivity_note": "An earlier version of this script used ONE globally-estimated "
                           "ENL from interior permanent water (23.7) for every pixel. That "
                           "reproduced the same flaw the PIF radiometric correction had: a "
                           "homogeneous reference target's noise level does not transfer to "
                           "built-up land, which has real structural texture beyond pure "
                           "speckle. Using water's ENL for built-up pixels flagged ~61% of "
                           "all built-up land as 'significant change' -- worse than the "
                           "original fixed +2.2dB threshold's 16.2%. Class-adaptive ENL "
                           "(built-up=own class, open-land=own class) fixes this.",
    "target_pfa_per_tail": PFA,
    "critical_ratio_builtup": {"increase": round(float(upper_crit_builtup), 4), "decrease": round(float(lower_crit_builtup), 4)},
    "critical_ratio_openland": {"increase": round(float(upper_crit_open), 4), "decrease": round(float(lower_crit_open), 4)},
    "results_sqkm": {
        "permanent_water": perm_sqkm,
        "built_up_total": builtup_sqkm,
        "open_flood": open_sqkm,
        "urban_double_bounce": urban_sqkm,
        "total_event_flood": total_sqkm,
        "urban_pct_of_builtup": urban_pct_of_builtup,
    },
}
with open("data/raw/local_ccd_diagnostics.json", "w", encoding="utf-8") as f:
    json.dump(ccd_diagnostics, f, indent=2)
print("[OK] Saved CCD masks and diagnostics -> data/raw/")
