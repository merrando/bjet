"""
Minimal fix for X-range truncation:
- Each model's total is built on the union of the grids from _ss, _cs, and _cs2.
- The final A+B sum is built on the union of the two totals' grids.
"""
import os, glob, shutil
import numpy as np
import matplotlib.pyplot as plt

import blazar_model
import blazar_utils
from blazar_properties import *

h = 4.135667662E-15
def v_to_e(val): return val * h
def e_to_v(val): return val / h

# ---- User inputs ----
Source = "Mrk421"
parameter_file_1 = "parameter_files/Mrk421_ext.par"
parameter_file_2 = "parameter_files/Mrk421_alt.par"
name_stem = "test_bj"
subdir_1 = "model_A_fix"
subdir_2 = "model_B_fix"

# ---- Plot/data config ----
if Source == "Mrk421":
    Ylim = [1e-15,1e-8]
    Xlim = [1e8,1e28]
    instrument_data_file = "real_data/Mrk421_SED_MAGIC_paper.dat"
else:
    raise ValueError("Only Mrk421 wired for this minimal example.")

def ensure_dir(d): os.makedirs(d, exist_ok=True)

def move_outputs(src_folder, dst_folder, stem=name_stem):
    ensure_dir(dst_folder)
    pats = [f"{stem}_ss.dat", f"{stem}_cs.dat", f"{stem}_cs2.dat", f"{stem}_nuc.dat"]
    for pat in pats:
        for src in glob.glob(os.path.join(src_folder, pat)):
            shutil.move(src, os.path.join(dst_folder, os.path.basename(src)))

def _load_component(path, assume_log=True):
    if not os.path.exists(path):
        return None
    arr = np.loadtxt(path)
    x, y = arr[:,0], arr[:,2]
    # If looks like log10 values:
    if assume_log and (0 < x.min() < 50) and (-100 < y.min() < 50):
        v = 10**x; f = 10**y
    else:
        v = x; f = y
    m = np.isfinite(v) & np.isfinite(f) & (v > 0) & (f >= 0)
    if m.sum() < 2: return None
    o = np.argsort(v[m])
    return v[m][o], f[m][o]

def load_total_union(data_folder, stem=name_stem):
    """Build total on the UNION of _ss, _cs, _cs2 grids (prevents truncation)."""
    paths = {
        "ss": os.path.join(BASE_PATH,  data_folder, f"{stem}_ss.dat"),
        "cs": os.path.join(BASE_PATH,  data_folder, f"{stem}_cs.dat"),
        "cs2":os.path.join(FOLDER_PATH,data_folder, f"{stem}_cs2.dat"),
    }
    comps = {}
    for k,p in paths.items():
        c = _load_component(p, assume_log=True)
        if c is not None:
            comps[k] = c
    if "ss" not in comps or "cs" not in comps:
        raise RuntimeError("Need at least _ss and _cs to build total.")
    # union grid
    logv_union = np.unique(np.concatenate([np.log10(v) for (v,_) in comps.values()]))
    v_grid = 10**logv_union
    f_total = np.zeros_like(v_grid)
    for v,f in comps.values():
        f_total += np.interp(np.log10(v_grid), np.log10(v), f, left=0.0, right=0.0)
    return v_grid, f_total, comps

if __name__ == "__main__":
    # data
    v_data, vFv_data, err_data, instrument_data, nubin_data = blazar_utils.read_data(instrument_data_file, instrument=True)
    uplims = [False]*len(v_data); lolims=[False]*len(v_data)
    for i in range(len(err_data[1])):
        if err_data[1][i] == -1:
            uplims[i] = True; err_data[0][i] = vFv_data[i]/4
        if err_data[0][i] == -1:
            lolims[i] = True; err_data[1][i] = vFv_data[i]/4

    # run model 1
    blazar_model.file_make_SED(parameter_file=parameter_file_1, data_folder=None, executable=None, prev_files=False, verbose=True)
    src = os.path.join(BASE_PATH, DATA_FOLDER)
    dstA = os.path.join(BASE_PATH, DATA_FOLDER, subdir_1)
    move_outputs(src, dstA, stem=name_stem)

    # run model 2
    blazar_model.file_make_SED(parameter_file=parameter_file_2, data_folder=None, executable=None, prev_files=False, verbose=True)
    dstB = os.path.join(BASE_PATH, DATA_FOLDER, subdir_2)
    move_outputs(src, dstB, stem=name_stem)

    # load totals on union grids (prevents 1e21 Hz cutoff)
    vA, fA, compsA = load_total_union(os.path.join(DATA_FOLDER, subdir_1), stem=name_stem)
    vB, fB, compsB = load_total_union(os.path.join(DATA_FOLDER, subdir_2), stem=name_stem)

    # sum on union of model grids
    logv_ab = np.unique(np.concatenate([np.log10(vA), np.log10(vB)]))
    v_ab = 10**logv_ab
    fA_u = np.interp(logv_ab, np.log10(vA), fA, left=0, right=0)
    fB_u = np.interp(logv_ab, np.log10(vB), fB, left=0, right=0)
    f_sum = fA_u + fB_u

    # plot
    fig, ax = plt.subplots(figsize=(7.25,5))

    # simple grouped data plotting
    cmap = plt.get_cmap("tab10")
    filled_markers = ("o","^","<",">","8","s","p","*","h","H","D","d","P","X")
    list_instr=[instrument_data[0]]
    vi=[v_data[0]]; yi=[vFv_data[0]]
    ei_dn=[err_data[0][0]]; ei_up=[err_data[1][0]]
    bi_lo=[nubin_data[0][0]]; bi_hi=[nubin_data[1][0]]
    ui=[uplims[0]]; li=[lolims[0]]
    tmp_c=0; tmp_m=0

    for i in range(1,len(instrument_data)):
        if len(list_instr)-tmp_c >= cmap.N: tmp_c += cmap.N
        if len(list_instr)-tmp_m >= len(filled_markers): tmp_m += len(filled_markers)
        color_index = len(list_instr)-tmp_c
        marker_index = len(list_instr)-tmp_m
        if instrument_data[i] != list_instr[-1]:
            ax.errorbar(vi, yi, xerr=(bi_lo, bi_hi), yerr=(ei_dn, ei_up),
                        uplims=ui, lolims=li, label=str(list_instr[-1]),
                        markersize=4, elinewidth=1, color=cmap(color_index),
                        fmt=filled_markers[marker_index-1])
            list_instr.append(instrument_data[i])
            vi=[v_data[i]]; yi=[vFv_data[i]]
            ei_dn=[err_data[0][i]]; ei_up=[err_data[1][i]]
            bi_lo=[nubin_data[0][i]]; bi_hi=[nubin_data[1][i]]
            ui=[uplims[i]]; li=[lolims[i]]
        else:
            vi.append(v_data[i]); yi.append(vFv_data[i])
            ei_dn.append(err_data[0][i]); ei_up.append(err_data[1][i])
            bi_lo.append(nubin_data[0][i]); bi_hi.append(nubin_data[1][i])
            ui.append(uplims[i]); li.append(lolims[i])

    ax.errorbar(vi, yi, xerr=(bi_lo, bi_hi), yerr=(ei_dn, ei_up),
                uplims=ui, lolims=li, label=str(list_instr[-1]),
                markersize=4, elinewidth=1, color=cmap(0), fmt="o")

    ax.plot(vA, fA, lw=2, label="Model 1 total")
    ax.plot(vB, fB, lw=2, ls="--", label="Model 2 total")
    ax.plot(v_ab, f_sum, lw=2, ls=":", label="Model 1 + 2")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(Xlim); ax.set_ylim(Ylim)
    ax.set_xlabel(r"Frequency $\nu$ (Hz)")
    ax.set_ylabel(r"Energy Flux $\nu F_{\nu}$ (erg cm$^{-2}$ s$^{-1})$")
    ax.legend(loc="best", fontsize=9)
    secax = ax.secondary_xaxis('top', functions=(v_to_e, e_to_v))
    secax.set_xlabel("Energy (eV)")
    plt.tight_layout()
    plt.show()
