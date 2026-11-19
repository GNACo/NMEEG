"""
Tablas (.tex) y figuras (.pdf) para Overleaf a partir de los resultados de pipeline_v2.py.

Uso: python paper_outputs_v2.py
Salida: RESULTS_ROOT/pipeline_v2/overleaf/
  table_demographics_v2.tex     Tabla 1 (muestra analizada)
  table_classification_v2.tex   Tabla 2 (clasificación, con y sin gamma)
  table_normative_v2.tex        Tabla suplementaria (desempeño y calibración del modelo)
  table_flow_v2.tex             Tabla suplementaria (flujo de participantes)
  fig_age_distribution_v2.pdf   Figura 1
  fig_group_deviations_v2.pdf   Figura 4b (Δz clínico - HC de test, FDR)
  resumen_numeros.md            Cifras para abstract / resultados
"""
import os
import sys
import json
import ast
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from run_config_params.paths import RESULTS_ROOT  # noqa: E402

V2 = os.path.join(RESULTS_ROOT, "pipeline_v2")
OUT = os.path.join(V2, "overleaf")
os.makedirs(OUT, exist_ok=True)

ORDER = ["ACr", "MCI", "AD"]                      # orden clínico (AJJ)
COLORS = {"HC": "#2F6FBF", "ACr": "#27A6A0", "MCI": "#7B3FA0", "AD": "#E0A526"}
BANDS = ["delta", "theta", "alpha1", "alpha2", "beta1", "beta2", "beta3", "gamma", "exponent", "offset"]
BAND_LBL = {"delta": "Delta", "theta": "Theta", "alpha1": "Alpha1", "alpha2": "Alpha2", "beta1": "Beta1",
            "beta2": "Beta2", "beta3": "Beta3", "gamma": "Gamma", "exponent": "Exponent", "offset": "Offset"}
ROIS = ["F", "C", "P", "O", "PO"]
ROI_LBL = {"F": "F", "C": "C", "P": "P", "O": "O", "PO": "PO"}
REF = {"ACr": (20, 45), "MCI": (47, 200), "AD": (47, 200)}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})

H = pd.read_csv(os.path.join(V2, "features_roi_harmonized.csv"))
C = pd.read_csv(os.path.join(V2, "classification.csv"))
G = pd.read_csv(os.path.join(V2, "group_deviations_vs_HCtest.csv"))
M = pd.read_csv(os.path.join(V2, "metrics_flat.csv"))
LOG = json.load(open(os.path.join(V2, "log.json")))
CT = os.path.join(RESULTS_ROOT, "pipeline_v2_TFfilter", "log.json")
LOG_TF = json.load(open(CT)) if os.path.exists(CT) else None


def fmt_age(a):
    return f"{a.mean():.1f} $\\pm$ {a.std():.1f}", f"{a.median():.0f} [{a.min():.0f}--{a.max():.0f}]"


def sex_str(s):
    s = s.dropna()
    n_all = None
    return s


# ------------------------------------------------------------------ Tabla 1
def table_demographics():
    cols = {
        "HC (all)": H[H.group == "HC"],
        "HC test, 20--45 y": H[(H.set == "test") & H.age.between(20, 45)],
        "HC test, $\\geq$47 y": H[(H.set == "test") & (H.age >= 47)],
        "ACr": H[H.group == "ACr"], "MCI": H[H.group == "MCI"], "AD": H[H.group == "AD"],
    }
    rows = {"n": [], "Age, years": [], "median [range]": [], "Sex, F/M (\\% available)": [], "Sites": []}
    for k, d in cols.items():
        rows["n"].append(str(len(d)))
        m, r = fmt_age(d.age)
        rows["Age, years"].append(m)
        rows["median [range]"].append(r)
        s = d.sex.dropna()
        rows["Sex, F/M (\\% available)"].append(f"{(s == 'F').sum()}/{(s == 'M').sum()} ({100 * len(s) / len(d):.0f}\\%)")
        rows["Sites"].append(str(d.SITE.nunique()))
    hdr = " & ".join(cols.keys())
    body = "\n".join(f"{k} & " + " & ".join(v) + " \\\\" for k, v in rows.items())
    tex = f"""\\begin{{table}}[t]
\\centering
\\caption{{Demographic characteristics of the analyzed sample. Age as mean $\\pm$ SD and median [range].
HC test are the held-out healthy controls used as reference: 20--45 years for ACr and $\\geq$47 years for MCI and AD.
MCI and AD from the Medell\\'in sites (PSEN1 E280A kindred) are not included.}}
\\label{{tab:demographics}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{l{'c' * len(cols)}}}
\\toprule
 & {hdr} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}}}
\\end{{table}}
"""
    open(os.path.join(OUT, "table_demographics_v2.tex"), "w").write(tex)
    return cols


# ------------------------------------------------------------------ Tabla 2
def table_classification():
    name = {"SVM": "Linear SVM", "LogReg": "Logistic regression", "RF": "Random forest", "DT": "Decision tree"}
    lines = []
    for g in ORDER:
        for feats, lab in [("todas", "all features (50)"), ("sin_gamma", "without gamma (45)")]:
            d = C[(C.comparison == f"HC vs {g}") & (C.features == feats)].copy()
            cnt = ast.literal_eval(d["counts"].iloc[0])
            best = d.AUC_mean.idxmax()
            lines.append(f"\\midrule\n\\multicolumn{{6}}{{l}}{{\\textbf{{HC vs {g}}} ($n_{{HC}}$ = {cnt['HC']}, "
                         f"$n_{{{g}}}$ = {cnt[g]}), {lab}}} \\\\")
            for i, r in d.iterrows():
                b = (lambda s: f"\\textbf{{{s}}}") if i == best else (lambda s: s)
                lines.append(f"{name[r.model]} & {b(f'{r.AUC_mean:.3f} $\\pm$ {r.AUC_std:.3f}')} & "
                             f"{r.accuracy:.3f} & {r.f1_score:.3f} & {r.brier:.3f} & "
                             f"{r.boot_AUC_mean:.3f} $\\pm$ {r.boot_AUC_std:.3f} \\\\")
    tex = """\\begin{table}[t]
\\centering
\\caption{Classification of each clinical group against age-comparable held-out healthy controls using normative
deviation scores (aperiodic-adjusted power in 8 bands and aperiodic exponent and offset, 5 ROIs).
Stratified 5-fold cross-validation with in-fold undersampling; 50-iteration out-of-bag bootstrap.
Bold: best CV AUC per comparison.}
\\label{tab:classification}
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{lccccc}
\\toprule
Model & CV AUC (mean $\\pm$ SD) & Accuracy & Macro-F1 & Brier & Bootstrap AUC (mean $\\pm$ SD) \\\\
""" + "\n".join(lines) + """
\\bottomrule
\\end{tabular}}
\\end{table}
"""
    open(os.path.join(OUT, "table_classification_v2.tex"), "w").write(tex)


# ------------------------------------------------------------------ Tablas suplementarias
def table_normative():
    m = M.copy()
    m["roi"] = m.feature.str.split("_roi").str[1]
    m["band"] = m.feature.str.split("_roi").str[0].str.replace("flat_", "")
    lines = []
    for _, r in m.iterrows():
        lines.append(f"{BAND_LBL[r.band]} & {r.roi} & {r.msll_test:.3f} & {r.ev_test:.3f} & {r.rho_test:.2f}"
                     f"{'$^*$' if r.rho_q < 0.05 else ''} & {r.z_test_mean:.2f} & {r.z_test_sd:.2f} & "
                     f"{r['z_test_pct_abs_gt_1.96']:.1f} \\\\")
    tex = """\\small
\\begin{longtable}{llrrrrrr}
\\caption{Normative model performance and calibration on held-out healthy controls (all ages).
MSLL $<0$: better than a trivial baseline. $^*$Spearman $q<0.05$ (FDR). A calibrated model gives
z mean $\\approx 0$, SD $\\approx 1$ and $\\approx$5\\% of $|z|>1.96$.}
\\label{tab:normative_metrics} \\\\
\\toprule
Band & ROI & MSLL & EV & $\\rho$ & z mean & z SD & \\% $|z|>1.96$ \\\\
\\midrule
""" + "\n".join(lines) + """
\\bottomrule
\\end{longtable}
"""
    open(os.path.join(OUT, "table_normative_v2.tex"), "w").write(tex)


def table_flow():
    rows = []
    base = LOG.get("TF_IAFp") or LOG_TF["TF_IAFp"]  # conteos tras control de épocas
    for site, v in base.items():
        tf = LOG_TF["TF_IAFp"][site]["despues"] if LOG_TF else None
        r2 = sum(n for k, n in LOG["R2_excluidos"].items() if k.startswith(site + "|"))
        fin = sum(n for k, n in LOG["muestra_final"].items() if k.startswith(site + "|"))
        rows.append(f"{site.replace('_', ' ')} & {v['antes']} & {tf if tf is not None else '--'} & {r2} & {fin} \\\\")
    tex = """\\begin{table}[h]
\\centering
\\caption{Participant flow by site. Recordings after epoch quality control ($\\geq$24 clean 5-s epochs);
participants that the previous TF/IAFp criterion would have retained (not applied in the final analysis);
exclusions for FOOOF $R^2<0.9$ in any channel; final analyzed sample (HC, MCI, AD, ACr; PSEN1-kindred MCI/AD,
PD, VD and FTD not included).}
\\label{tab:flow}
\\begin{tabular}{lrrrr}
\\toprule
Site & After epoch QC & TF/IAFp criterion (not applied) & $R^2<0.9$ & Analyzed \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
\\end{table}
"""
    open(os.path.join(OUT, "table_flow_v2.tex"), "w").write(tex)


# ------------------------------------------------------------------ Figura 1
def fig_age():
    groups = ["HC", "ACr", "MCI", "AD"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1, 2.4]})
    ax = axes[0]
    rng = np.random.default_rng(0)
    for i, g in enumerate(groups):
        a = H[H.group == g].age.values
        v = ax.violinplot([a], positions=[i], widths=0.8, showextrema=False)
        for b in v["bodies"]:
            b.set_facecolor(COLORS[g]); b.set_alpha(0.45); b.set_edgecolor("none")
        ax.scatter(i + rng.normal(0, 0.06, len(a)), a, s=3, color=COLORS[g], alpha=0.5, lw=0)
        ax.boxplot([a], positions=[i], widths=0.18, showfliers=False,
                   medianprops=dict(color="black"), boxprops=dict(color="black"), whiskerprops=dict(color="black"),
                   capprops=dict(color="black"))
        ax.text(i, 101, f"n={len(a)}", ha="center", fontsize=8)
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylabel("Age (years)")
    ax.set_ylim(10, 106)
    ax.set_title("Pooled sample", loc="left", fontsize=10)
    ax = axes[1]
    sites = (H.groupby("SITE").age.median().sort_values().index.tolist())
    off = {"HC": -0.27, "ACr": -0.09, "MCI": 0.09, "AD": 0.27}
    for j, s in enumerate(sites):
        for g in groups:
            a = H[(H.SITE == s) & (H.group == g)].age.values
            if len(a) == 0:
                continue
            ax.scatter(j + off[g] + rng.normal(0, 0.03, len(a)), a, s=3, color=COLORS[g], alpha=0.5, lw=0)
            ax.boxplot([a], positions=[j + off[g]], widths=0.14, showfliers=False,
                       medianprops=dict(color=COLORS[g], lw=1.5), boxprops=dict(color=COLORS[g]),
                       whiskerprops=dict(color=COLORS[g]), capprops=dict(color=COLORS[g]))
    ax.set_xticks(range(len(sites)), [f"{s.replace('Medellin_', 'Medellín ')}\n(n={(H.SITE == s).sum()})".replace(" hd", "\nhd").replace(" ld", "\nld") for s in sites],
                  rotation=0, fontsize=7.5)
    ax.set_ylim(10, 106)
    ax.set_title("By recording site", loc="left", fontsize=10)
    handles = [plt.Line2D([], [], marker="o", ls="", color=COLORS[g], label=g) for g in groups]
    ax.legend(handles=handles, frameon=False, ncol=4, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_age_distribution_v2.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig_age_distribution_v2.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ Figura 4b
def fig_deviations():
    g = G.copy()
    g["band"] = g["band"].fillna(g["feature"])
    vmax = np.ceil(g.delta_z.abs().max() * 4) / 4
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), sharey=True)
    for ax, grp in zip(axes, ORDER):
        d = g[g.group == grp]
        Z = d.pivot(index="band", columns="roi", values="delta_z").loc[BANDS, ROIS]
        Q = d.pivot(index="band", columns="roi", values="q").loc[BANDS, ROIS]
        im = ax.imshow(Z.values, cmap="RdBu_r", norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                q = Q.values[i, j]
                star = "***" if q < .001 else "**" if q < .01 else "*" if q < .05 else ""
                val = Z.values[i, j]
                ax.text(j, i, f"{val:+.2f}\n{star}" if star else f"{val:+.2f}", ha="center", va="center",
                        fontsize=7, color="white" if abs(val) > 0.55 * vmax else "black")
        lo, hi = REF[grp]
        ref = f"HC {lo}–{hi} y" if hi < 200 else f"HC ≥{lo} y"
        ax.set_title(f"{grp} (n={int(d.n_clin.iloc[0])}) vs {ref} (n={int(d.n_ref.iloc[0])})", fontsize=9, loc="left")
        ax.set_xticks(range(len(ROIS)), [ROI_LBL[r] for r in ROIS], fontsize=7.5)
        ax.set_yticks(range(len(BANDS)), [BAND_LBL[b] for b in BANDS])
        ax.spines[:].set_visible(False)
        ax.axhline(7.5, color="white", lw=2)
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.01)
    cb.set_label("Δ mean z (clinical − HC test)")
    fig.savefig(os.path.join(OUT, "fig_group_deviations_v2.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig_group_deviations_v2.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ resumen
def summary(cols):
    best = C[C.features == "todas"].loc[lambda d: d.groupby("comparison").AUC_mean.idxmax()]
    nog = C[C.features == "sin_gamma"].loc[lambda d: d.groupby("comparison").AUC_mean.idxmax()]
    n = H.group.value_counts()
    lines = [
        "# Cifras para el manuscrito (pipeline v2)",
        "",
        f"- Muestra analizada: {len(H)} participantes de {H.SITE.nunique()} sitios en "
        f"{H.SITE.replace({'Medellin_hd': 'Colombia', 'Medellin_ld': 'Colombia', 'Seoul': 'Korea', 'Spain': 'Spain'}).nunique()} países: "
        f"{n.get('HC', 0)} HC, {n.get('ACr', 0)} ACr, {n.get('MCI', 0)} MCI, {n.get('AD', 0)} AD.",
        f"- AD de dos sitios: " + ", ".join(f"{s} {k}" for s, k in H[H.group == 'AD'].SITE.value_counts().items()) + ".",
        f"- Exclusiones por FOOOF R2 < 0.9: {LOG['R2_excluidos_total']}.",
        f"- HC entrenamiento: {(H.set == 'train').sum()}; HC test: {(H.set == 'test').sum()}.",
        "",
        "## Clasificación (mejor CV AUC)",
    ]
    for _, r in best.iterrows():
        rr = nog[nog.comparison == r.comparison].iloc[0]
        lines.append(f"- {r.comparison}: {r.AUC_mean:.3f} ± {r.AUC_std:.3f} ({r.model}); bootstrap "
                     f"{r.boot_AUC_mean:.3f}; sin gamma {rr.AUC_mean:.3f} ({rr.model}).")
    lines += ["", "## Desviaciones de grupo (Δz vs HC test, q < 0.05)"]
    g = G.copy(); g["band"] = g["band"].fillna(g["feature"])
    for grp in ORDER:
        d = g[(g.group == grp) & (g.q < 0.05)].sort_values("delta_z")
        lines.append(f"- {grp}: " + "; ".join(f"{r.band}-{r.roi} {r.delta_z:+.2f}" for _, r in d.iterrows()))
    lines += ["", "## Calibración del modelo normativo (HC test)",
              f"- DE de z mediana {M.z_test_sd.median():.2f}; % |z|>1.96 mediana {M['z_test_pct_abs_gt_1.96'].median():.1f}%.",
              "- Peor calibradas: " + ", ".join(M.sort_values('z_test_sd', ascending=False).head(6).feature)]
    open(os.path.join(OUT, "resumen_numeros.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    cols = table_demographics()
    table_classification()
    table_normative()
    table_flow()
    fig_age()
    fig_deviations()
    summary(cols)
    print("Salida:", OUT)
