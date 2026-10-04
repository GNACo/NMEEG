"""
Master script — generates all publication figures.

  .\\myenv\\Scripts\\python.exe run_figures.py

Output folder: FIGURES_PATH (change the single variable below).

-----------------------------------------------------------------
Fig results  fig_results.py
       PURPOSE : Single unified results figure (3 panels).
       Panel a : Centile curves — 7 bands, Occipital ROI.
       Panel b : Group z-score heatmaps (MCI | AD | ACr PSEN1 E280A).
       Panel c : Individual profiles — gauge cards per group.

Fig 2  blr_performance_figure.py
       PURPOSE : Validate the normative model on held-out test set.
       Panel a : MSLL heatmap (bands x ROIs).
       Panel b : Spearman rho heatmap with significance stars.
       Panel c : Q-Q plots per band (Gaussian calibration check).

Fig 3  fig3_spectra_zoom.py
       PURPOSE : Group-average spectra (HC/MCI/ACr/AD/PD/VD).
       3 panels: Unadjusted | Oscillatory Fit | Aperiodic, with
       zoomed call-outs on the theta/alpha peak and the 27-40 Hz tail.

Demographic figure is in figures/graphs_demographic.ipynb
(run separately in Jupyter).
-----------------------------------------------------------------
"""

import os
from run_config_params.paths import FIGURES_DIR
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent

# ── OUTPUT FOLDER — change this line to redirect all figures ──────────────────
FIGURES_PATH = (
    FIGURES_DIR
)

SCRIPTS = [
    ("Results figure (a: curves | b: heatmaps | c: profiles)",
     ROOT / "figures" / "fig_results.py"),

    ("Fig 2 — BLR model performance & calibration",
     ROOT / "figures" / "blr_performance_figure.py"),

    ("Fig 3 — Group spectra with zoom call-outs",
     ROOT / "figures" / "fig3_spectra_zoom.py"),
]

SEP = "-" * 68


def run_script(label: str, path: Path, env: dict) -> bool:
    print(f"\n{SEP}")
    print(f"  {label}")
    print(f"  {path.name}")
    print(SEP)
    t0 = time.time()
    result = subprocess.run([sys.executable, str(path)],
                            cwd=str(ROOT), env=env)
    elapsed = time.time() - t0
    ok = result.returncode == 0
    print(f"  {'[DONE]  ' if ok else '[FAILED]'}  ({elapsed:.1f} s)")
    return ok


if __name__ == "__main__":
    os.makedirs(FIGURES_PATH, exist_ok=True)
    env = {**os.environ, "NMEEG_FIGURES_PATH": FIGURES_PATH}

    print(f"\n{'=' * 68}")
    print("  Normative EEG -- figure generation")
    print(f"  -> {FIGURES_PATH}")
    print(f"{'=' * 68}")

    results = {label: run_script(label, path, env) for label, path in SCRIPTS}

    print(f"\n{SEP}")
    all_ok = all(results.values())
    for label, ok in results.items():
        print(f"  [{'OK    ' if ok else 'FAILED'}]  {label}")
    print(SEP)
    print(f"  {'All figures saved.' if all_ok else 'One or more scripts failed.'}")
    print()
