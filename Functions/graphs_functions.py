import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr
from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import matplotlib.lines as mlines

def plot_pca(unharmonized_data, harmonized_data,directory, method, approach, filename):
    sns.set_style("whitegrid")
    sns.set_context("talk")

    # --- FIGURA PCA ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    # 2 filas x 3 columnas, con auto-ajuste para evitar superposiciones

    ###############################################################################
    # 1) Definir paletas y "handles" para SITE y GROUP (para dibujar centroides y leyendas)
    ###############################################################################

    # Paleta y handles para SITE
    unique_site = np.sort(unharmonized_data["SITE"].unique())
    pal_site = sns.color_palette("Set1", n_colors=len(unique_site))
    handles_site = [
        Line2D([], [], marker='o', linestyle='', color=pal_site[i],
            markersize=6, label=site)
        for i, site in enumerate(unique_site)
    ]

    # Paleta y handles para GROUP
    unique_group = np.sort(unharmonized_data["group"].unique())
    pal_group = sns.color_palette("Set2", n_colors=len(unique_group))
    handles_group = [
        Line2D([], [], marker='o', linestyle='', color=pal_group[i],
            markersize=6, label=g)
        for i, g in enumerate(unique_group)
    ]

    # --- Columna 1: color por SITE + centroides ---
    ax = axes[0, 0]
    sns.scatterplot(
        ax=ax,
        data=unharmonized_data,
        x="pca-one",
        y="pca-two",
        hue="SITE",
        palette=pal_site,
        s=30,
        alpha=0.5,
        legend=False  # quitamos la leyenda automática
    )
    ax.set_title("PCA - Unharmonized (SITE)")

    # Dibujar centroides para cada SITE
    for i, site in enumerate(unique_site):
        df_site = unharmonized_data[unharmonized_data["SITE"] == site]
        cx = df_site["pca-one"].mean()
        cy = df_site["pca-two"].mean()
        ax.scatter(cx, cy, color=pal_site[i], s=100, marker='o', edgecolor=pal_site[i])

    # Leyenda manual para SITE, fuera del gráfico
    ax.legend(handles=handles_site, title="SITE", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

    # --- Columna 2: color por GROUP + centroides ---
    ax = axes[0, 1]
    sns.scatterplot(
        ax=ax,
        data=unharmonized_data,
        x="pca-one",
        y="pca-two",
        hue="group",
        palette=pal_group,
        s=30,
        alpha=0.3,
        legend=False  # leyenda manual
    )
    ax.set_title("PCA - Unharmonized (Diagnosis)")

    # Dibujar centroides para cada group
    for i, grp in enumerate(unique_group):
        df_grp = unharmonized_data[unharmonized_data["group"] == grp]
        cx = df_grp["pca-one"].mean()
        cy = df_grp["pca-two"].mean()
        ax.scatter(cx, cy, color=pal_group[i], s=100, marker='o', edgecolor=pal_group[i])

    # Leyenda manual para GROUP
    ax.legend(handles=handles_group, title="Diagnosis", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    
    # --- Columna 1: color por SITE + centroides ---
    ax = axes[1, 0]
    sns.scatterplot(
        ax=ax,
        data=harmonized_data,
        x="harmonized-pca-one",
        y="harmonized-pca-two",
        hue="SITE",
        palette=pal_site,
        s=30,
        alpha=0.3,
        legend=False
    )
    ax.set_title("PCA - Harmonized (SITE)")

    # Centroides
    for i, site in enumerate(unique_site):
        df_site = harmonized_data[harmonized_data["SITE"] == site]
        cx = df_site["harmonized-pca-one"].mean()
        cy = df_site["harmonized-pca-two"].mean()
        ax.scatter(cx, cy, color=pal_site[i], s=100, marker='o', edgecolor=pal_site[i])

    ax.legend(handles=handles_site, title="SITE", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

    # --- Columna 2: color por GROUP + centroides ---
    ax = axes[1, 1]
    sns.scatterplot(
        ax=ax,
        data=harmonized_data,
        x="harmonized-pca-one",
        y="harmonized-pca-two",
        hue="group",
        palette=pal_group,
        s=30,
        alpha=0.3,
        legend=False
    )
    ax.set_title("PCA - Harmonized (Diagnosis)")

    # Centroides
    for i, grp in enumerate(unique_group):
        df_grp = harmonized_data[harmonized_data["group"] == grp]
        cx = df_grp["harmonized-pca-one"].mean()
        cy = df_grp["harmonized-pca-two"].mean()
        ax.scatter(cx, cy, color=pal_group[i], s=100, marker='o', edgecolor=pal_group[i])

    ax.legend(handles=handles_group, title="Diagnosis", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()
    plt.savefig(fr'{directory}/results_harmonize/{method}/graphs/{approach}/PCA_{filename}.png', dpi=300)
    plt.close()
    
def plot_umap(unharmonized_data, harmonized_data,directory, method, approach, filename):
        
    sns.set_style("whitegrid")
    sns.set_context("talk")

    fig_umap, axes_umap = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    # Paleta y handles para SITE (columna 1)
    unique_site = np.sort(unharmonized_data["SITE"].unique())
    pal_site = sns.color_palette("Set1", n_colors=len(unique_site))
    handles_site = [
        Line2D([], [], marker='o', linestyle='', color=pal_site[i],
            markersize=8, label=site)
        for i, site in enumerate(unique_site)
    ]

    # Paleta y handles para GROUP (columna 2)
    unique_group = np.sort(unharmonized_data["group"].unique())
    pal_group = sns.color_palette("Set2", n_colors=len(unique_group))
    handles_group = [
        Line2D([], [], marker='o', linestyle='', color=pal_group[i],
            markersize=8, label=grp)
        for i, grp in enumerate(unique_group)
    ]

    ###############################
    # FILA 1: UMAP Unharmonized
    ###############################

    ## Columna 1: coloreado por SITE + centroides
    ax = axes_umap[0, 0]
    sns.scatterplot(
        ax=ax,
        data=unharmonized_data,
        x="umap-one",
        y="umap-two",
        hue="SITE",
        palette=pal_site,
        s=30,
        alpha=0.3,
        legend=False
    )
    ax.set_title("UMAP - Unharmonized (SITE)")
    # Agregar centroides para cada SITE
    for i, site in enumerate(unique_site):
        df_site = unharmonized_data[unharmonized_data["SITE"] == site]
        cx = df_site["umap-one"].mean()
        cy = df_site["umap-two"].mean()
        ax.scatter(cx, cy, color=pal_site[i], s=100, marker='o', edgecolor=pal_site[i])
    # Leyenda manual fuera del área de trazado
    ax.legend(handles=handles_site, title="SITE", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

    ## Columna 2: coloreado por group + centroides
    ax = axes_umap[0, 1]
    sns.scatterplot(
        ax=ax,
        data=unharmonized_data,
        x="umap-one",
        y="umap-two",
        hue="group",
        palette=pal_group,
        s=30,
        alpha=0.3,
        legend=False
    )
    ax.set_title("UMAP - Unharmonized (Group)")
    # Centroides para cada group
    for i, grp in enumerate(unique_group):
        df_grp = unharmonized_data[unharmonized_data["group"] == grp]
        cx = df_grp["umap-one"].mean()
        cy = df_grp["umap-two"].mean()
        ax.scatter(cx, cy, color=pal_group[i], s=100, marker='o', edgecolor=pal_group[i])
    ax.legend(handles=handles_group, title="Group", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

    ###############################
    # FILA 2: UMAP Harmonized
    ###############################

    ## Columna 1: coloreado por SITE + centroides
    ax = axes_umap[1, 0]
    sns.scatterplot(
        ax=ax,
        data=harmonized_data,
        x="harmonized-umap-one",
        y="harmonized-umap-two",
        hue="SITE",
        palette=pal_site,
        s=30,
        alpha=0.3,
        legend=False
    )
    ax.set_title("UMAP - Harmonized (SITE)")
    # Agregar centroides para cada SITE
    for i, site in enumerate(unique_site):
        df_site = harmonized_data[harmonized_data["SITE"] == site]
        cx = df_site["harmonized-umap-one"].mean()
        cy = df_site["harmonized-umap-two"].mean()
        ax.scatter(cx, cy, color=pal_site[i], s=100, marker='o', edgecolor=pal_site[i])
    ax.legend(handles=handles_site, title="SITE", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

    ## Columna 2: coloreado por group + centroides
    ax = axes_umap[1, 1]
    sns.scatterplot(
        ax=ax,
        data=harmonized_data,
        x="harmonized-umap-one",
        y="harmonized-umap-two",
        hue="group",
        palette=pal_group,
        s=30,
        alpha=0.3,
        legend=False
    )
    ax.set_title("UMAP - Harmonized (Group)")
    # Centroides para cada group
    for i, grp in enumerate(unique_group):
        df_grp = harmonized_data[harmonized_data["group"] == grp]
        cx = df_grp["harmonized-umap-one"].mean()
        cy = df_grp["harmonized-umap-two"].mean()
        ax.scatter(cx, cy, color=pal_group[i], s=100, marker='o', edgecolor=pal_group[i])
    ax.legend(handles=handles_group, title="Group", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()
    plt.savefig(fr'{directory}/results_harmonize/{method}/graphs/{approach}/UMAP_{filename}.png', dpi=300)
    plt.close()

def plot_with_means(X_original, X_harmonized, y,directory, title, method, approach, filename):
    unique_classes = np.unique(y)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax, X, name in zip(axes, [X_original, X_harmonized], ["Original", "Harmonized"]):
        for i, cls in enumerate(unique_classes):
            indices = np.where(y == cls)
            ax.scatter(X[indices, 0], X[indices, 1], label=f'{cls}', alpha=0.3, color=colors[i])
            
            # Calcular y dibujar centroides
            centroid = X[indices].mean(axis=0)
            ax.scatter(centroid[0], centroid[1], marker='X', color=colors[i], edgecolor=colors[i], s=100)
        
        ax.set_title(f'{title} - {name}')
        ax.legend()
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
    
    plt.tight_layout()
    plt.savefig(fr'{directory}/results_harmonize/{method}/graphs/{approach}/{title}_{filename}.png', dpi=300)
    plt.close()

def compute_mean_distances(coords_by_site, sites):
    n_sites = len(sites)
    distance_matrix = np.zeros((n_sites, n_sites))
    
    for i, site_i in enumerate(sites):
        for j, site_j in enumerate(sites):
            if i != j:
                distances = pairwise_distances(coords_by_site[site_i], coords_by_site[site_j], metric='euclidean')
                distance_matrix[i, j] = distances.mean()
    return distance_matrix

def plot_distance(unharmonized_data, harmonized_data,pca=True, directory= str, method=str, approach=str, filename=str):

    sites = unharmonized_data['SITE'].unique()
    dist_matrix_unharm = pd.DataFrame(index=sites, columns=sites, dtype=float)
    dist_matrix_harm   = pd.DataFrame(index=sites, columns=sites, dtype=float)
    if pca:
        components = ['pca-one', 'pca-two']
        components_harm = ['harmonized-pca-one', 'harmonized-pca-two']
    else:
        components = ['umap-one', 'umap-two']
        components_harm = ['harmonized-umap-one', 'harmonized-umap-two']
    # 3) Calcular la distancia euclidiana mediana ("all-pairs") para cada par de sitios en PCA
    for s1 in sites:
        # Coordenadas para s1 (sin armonizar)
        coords_s1_unharm = unharmonized_data.loc[unharmonized_data['SITE'] == s1, components].values
        # Coordenadas para s1 (armonizado)
        coords_s1_harm = harmonized_data.loc[harmonized_data['SITE'] == s1, components_harm].values
        for s2 in sites:
            # Coordenadas para s2 (sin armonizar)
            coords_s2_unharm = unharmonized_data.loc[unharmonized_data['SITE'] == s2, components].values
            # Coordenadas para s2 (armonizado)
            coords_s2_harm   = harmonized_data.loc[harmonized_data['SITE'] == s2, components_harm].values
            
            # Calcular todas las distancias y tomar la mediana
            dists_unharm = cdist(coords_s1_unharm, coords_s2_unharm, metric='euclidean')
            dists_harm   = cdist(coords_s1_harm,   coords_s2_harm,   metric='euclidean')
            dist_matrix_unharm.loc[s1, s2] = np.median(dists_unharm)
            dist_matrix_harm.loc[s1, s2]   = np.median(dists_harm)

    # 4) Calcular la matriz de diferencia: (armonizado - sin armonizar)
    diff_matrix = dist_matrix_harm.astype(float) - dist_matrix_unharm.astype(float)

    # 5) Crear una matriz discreta de diferencia:
    #    - Si la diferencia < -epsilon --> -1 (disminuyó)
    #    - Entre -epsilon y +epsilon --> 0 (sin cambio)
    #    - Si la diferencia > epsilon --> 1 (aumentó)
    epsilon = 0.01  # Umbral para considerar "sin cambio"
    diff_sign = np.zeros_like(diff_matrix, dtype=int)
    diff_sign[diff_matrix < -epsilon] = -1
    diff_sign[np.abs(diff_matrix) <= epsilon] = 0
    diff_sign[diff_matrix > epsilon]  = 1

    # 6) Definir un colormap discreto para diff_sign: verde para -1, gris para 0, rojo para 1
    cmap_diff = mcolors.ListedColormap(['thistle', 'white', 'indigo'])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm_diff = mcolors.BoundaryNorm(bounds, cmap_diff.N)

    # 7) Crear una máscara para mostrar solo el triángulo inferior en los heatmaps
    mask_upper = np.triu(np.ones_like(dist_matrix_unharm, dtype=bool))
    mask_diff  = np.triu(np.ones_like(diff_matrix, dtype=bool))

    # 8) Definir un rango de color común (vmin y vmax) para los heatmaps de distancias
    vmin = min(dist_matrix_unharm.astype(float).min().min(), dist_matrix_harm.astype(float).min().min())
    vmax = max(dist_matrix_unharm.astype(float).max().max(),   dist_matrix_harm.astype(float).max().max())

    # 9) Crear una figura con 1 fila y 3 columnas
    annot_matrix = diff_matrix.map(lambda x: f"{x:.2f}" if x > epsilon else "")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    orig_map=plt.cm.get_cmap('viridis') 
    # Heatmap: Distancias sin armonizar (solo triángulo inferior)
    hm1 = sns.heatmap(dist_matrix_unharm.astype(float),
                mask=mask_upper,
                annot=True, fmt=".2f", annot_kws={"size": 14},
                cmap=orig_map.reversed() , square=True,
                linewidths=0.5,
                vmin=vmin, vmax=vmax,
                ax=ax1, cbar=False)
    ax1.set_title("Median Distance - Unharmonized")
    ax1.set_xlabel("SITE")
    ax1.set_ylabel("SITE")
    ax1.grid(False)
    
    # Heatmap: Distancias armonizadas (solo triángulo inferior)
    hm2 = sns.heatmap(dist_matrix_harm.astype(float),
                mask=mask_upper,
                annot=True, fmt=".2f", annot_kws={"size": 14},
                cmap=orig_map.reversed() , square=True,
                linewidths=0.5,
                vmin=vmin, vmax=vmax,
                ax=ax2, cbar=False)
    ax2.set_title("Median Distance - Harmonized")
    ax2.set_xlabel("SITE")
    ax2.set_ylabel("SITE")
    ax2.grid(False)
    # Crear una única barra de color común para los dos primeros heatmaps
    cbar = fig.colorbar(hm2.collections[0], ax=[ax1, ax2], orientation="vertical", fraction=0.05, pad=0.04)
    cbar.set_label("Distance")

    # Heatmap: Comparación (diferencia discreta; sin anotación)
    hm3 = sns.heatmap(diff_sign,
                    mask=mask_diff,
                    cmap=cmap_diff, norm=norm_diff,
                    annot=annot_matrix, fmt="",
                    annot_kws={"fontsize":14},
                    square=True,
                    linewidths=0,  # No gridlines
                    cbar=False, ax=ax3)
    ax3.set_title("Change in Distance\n(Harmonized - Unharmonized)")
    ax3.set_xlabel("SITE")
    ax3.set_ylabel("SITE")

    # Leyenda manual para el heatmap de comparación
    legend_elements = [Patch(facecolor='indigo', label='Increased'),
                        Patch(facecolor='thistle', label='Decreased'),
                    Patch(facecolor='white', label='No change')]
    ax3.legend(handles=legend_elements, title='Change', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(False)
    plt.tight_layout()
    if not pca:
        plt.savefig(rf'{directory}/results_harmonize/{method}/graphs/{approach}/UMAP_distance_{filename}.png', dpi=300)
    else:
        plt.savefig(rf'{directory}/results_harmonize/{method}/graphs/{approach}/PCA_distance_{filename}.png', dpi=300)
    plt.close()

def plot_spearman_correlation(df, features,directory,method,approach, filename="spearman_results"):
    features.drop(['pca-one','pca-two','harmonized-pca-one','harmonized-pca-two','umap-one','umap-two','harmonized-umap-one','harmonized-umap-two'], axis=1, errors='ignore', inplace=True)
    features = features.columns.tolist()

    # Separar los datos por grupo
    df_HC = df[df["group"] == "HC"]
    df_MCI = df[df["group"] == "MCI"]

    # Calcular número óptimo de filas y columnas
    n_features = len(features)
    cols = min(4, n_features)  # Máximo 4 columnas para mejor visualización
    rows = int(np.ceil(n_features / cols))  # Ajuste automático de filas

    SITE_colors = {SITE: ('crimson' if "Medellin_ld" in SITE else 'purple') for SITE in df["SITE"].unique()}
    
    results = []  # Lista para almacenar resultados

    # Función para asignar asteriscos según p-valor
    def significance_stars(p):
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return ""

    def plot_group(df_group, group_name, save_path):
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))  # Tamaño dinámico
        axes = np.array(axes).flatten()  # Asegurar acceso con índice
        
        for i, col in enumerate(features):
            if col != ['age', 'group', 'SITE', 'subject']:
                if i >= len(axes):  
                    break  # Evitar errores si hay más features que subgráficos
                
                ax = axes[i]
                
                # Calcular la correlación de Spearman
                r, p = spearmanr(df_group['age'], df_group[col])
                sig = significance_stars(p)
                
                # Almacenar resultado en la lista
                results.append({"group": group_name, "Metric": col, "Spearman_R": r, "p-value": p, "Significance": sig})
                
                # Graficar puntos con colores personalizados
                sns.scatterplot(
                    data=df_group, x='age', y=col, hue='SITE', palette=SITE_colors, s=20,
                    sizes=(30, 70), ax=ax, legend=False
                )

                # Graficar la línea de regresión
                sns.regplot(data=df_group, x='age', y=col, scatter=False, ax=ax, color='blue', ci=None)
                
                # Título con R y p-valor + significancia
                ax.set_title(f"Corr = {r:.2f}, p = {p:.2g} {sig}")

        # Título general
        fig.suptitle(f"Spearman Correlation - {group_name} - {filename}", fontsize=16)
        fig.tight_layout()
        
        # Guardar imagen
        plt.savefig(save_path, dpi=300)
        plt.close()  # Liberar memoria
    
    # Graficar y guardar cada grupo
    plot_group(df_HC, "HC", os.path.join(directory, "results_harmonize",method, "statical_tests",  approach, "spearman_correlation", f"{filename}_HC.png"))
    plot_group(df_MCI, "MCI", os.path.join(directory, "results_harmonize",method ,"statical_tests", approach, "spearman_correlation", f"{filename}_MCI.png"))

    # Guardar CSV con los resultados de ambos grupos
    results_df = pd.DataFrame(results)
    if not 'osc' in filename:
        file_path = os.path.join(directory, "results_harmonize", method, "statical_tests", approach, 'spearman_correlation',"spearman_correlation.xlsx")
    else:
        file_path = os.path.join(directory, "results_harmonize", method, "statical_tests", approach, 'spearman_correlation','spearman_correlation_fooof.xlsx')
    from openpyxl import Workbook

    if not os.path.exists(file_path):
        wb = Workbook()
        wb.save(file_path)

    mode = "a" if os.path.exists(file_path) else "w"
    with pd.ExcelWriter(file_path, engine="openpyxl", mode=mode) as writer:
        results_df.to_excel(writer, sheet_name=filename, index=False)


import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

def get_spearman_by_site(df, feature, subject_col='subject', site_col='SITE'):
    """
    Computes Spearman correlation between 'age' and the specified feature
    within each group defined by (subject, SITE).
    
    Returns a DataFrame with columns: [subject_col, site_col, 'r', 'p']
    """
    def compute_corr(subdf):
        # Compute Spearman correlation for the group; 
        # subdf must have multiple rows to compute correlation.
        r, p = spearmanr(subdf['age'], subdf[feature], nan_policy='omit')
        return pd.Series({'r': r, 'p': p})
    
    grouped = df.groupby([subject_col, site_col]).apply(compute_corr).reset_index()
    return grouped


def significance_stars(p):
    """Devuelve la cadena de asteriscos según el p-valor."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""

def find_best_correlation(corr_info):
    """
    Dado un DataFrame corr_info con columnas:
      ['Data', 'Feature', 'Group', 'Spearman_r', 'p_value']
    Filtra los que tengan p<0.05 y retorna (feature, group) con el mayor |r|.
    Si ninguno es significativo, retorna (None, None).
    """
    significant = corr_info[corr_info['p_value'] < 0.05]
    if len(significant) == 0:
        return None, None
    # Ordenar por |Spearman_r| descendente
    significant_copy = significant.copy()
    significant_copy['abs_r'] = significant_copy['Spearman_r'].abs()
    best_row = significant_copy.sort_values('abs_r', ascending=False).iloc[0]
    return best_row['Feature'], best_row['Group']

def plot_spearman_scatter_single_winner(
    df_unharm, df_harm, features, 
    directory, method, 
    filename=str,
    excel_filename="spearman_correlations.xlsx",
    sheet_name="spearman_data",family_features= str, covars = str
    ):

    out_dir = os.path.join(directory, "results_harmonize", method, "statical_tests")
    os.makedirs(out_dir, exist_ok=True)
    
    group_colors = {"HC": "crimson", "MCI": "purple"}
    groups = ["HC", "MCI"]
    
    n_features = len(features)
    fig, axes = plt.subplots(2, n_features, figsize=(5*n_features, 10))
    if n_features == 1:
        axes = np.array(axes).reshape(2,1)
    
    # 1) Calculamos correlaciones para UNHARM (fila 1) y HARM (fila 2)
    results_list = []
    
    # --- Recolectar info de correlaciones unharm (todas features, ambos grupos)
    unharm_data = []
    for feat in features:
        data_u = df_unharm[['age','group',feat]].dropna()
        for grp in groups:
            sub = data_u[data_u['group'] == grp]
            if len(sub) >= 2:
                r, p = spearmanr(sub['age'], sub[feat])
            else:
                r, p = np.nan, np.nan
            unharm_data.append({
                "category": family_features,
                "Data": "Unharmonized",
                "covars": covars,
                "Feature": feat,
                "Group": grp,
                "Spearman_r": r,
                "p_value": p
            })
    
    unharm_df = pd.DataFrame(unharm_data)
    best_feat_unharm, best_group_unharm = find_best_correlation(unharm_df)
    
    # --- Recolectar info de correlaciones harm
    harm_data = []
    for feat in features:
        col_harm = "harm_" + feat
        if col_harm not in df_harm.columns:
            # no se puede correlacionar
            harm_data.append({
                "category": family_features,
                "Data": "Harmonized",
                "covars": covars,
                "Feature": feat,
                "Group": None,
                "Spearman_r": np.nan,
                "p_value": np.nan
            })
            continue
        
        data_h = df_harm[['age','group',col_harm]].dropna().rename(columns={col_harm: feat})
        for grp in groups:
            sub = data_h[data_h['group'] == grp]
            if len(sub) >= 2:
                r, p = spearmanr(sub['age'], sub[feat])
            else:
                r, p = np.nan, np.nan
            harm_data.append({
                "category": family_features,
                "Data": "Harmonized",
                "Feature": feat,
                "Group": grp,
                "Spearman_r": r,
                "p_value": p
            })
    
    harm_df = pd.DataFrame(harm_data)
    best_feat_harm, best_group_harm = find_best_correlation(harm_df)
    
    # 2) Graficar
    for j, feat in enumerate(features):
        # --- Fila 1: unharm
        ax_u = axes[0, j]
        df_u = df_unharm[['age','group',feat]].dropna()
        
        # Por cada grupo
        lines_for_legend = []
        for grp in groups:
            sub = df_u[df_u['group'] == grp]
            row_match = unharm_df[(unharm_df['Feature']==feat)&(unharm_df['Group']==grp)]
            if len(row_match)==1:
                r = row_match.iloc[0]['Spearman_r']
                p = row_match.iloc[0]['p_value']
            else:
                r, p = np.nan, np.nan
            stars = significance_stars(p)
            
            # Determinar alpha
            if feat == best_feat_unharm and grp == best_group_unharm and p<0.05:
                alpha_val = 1.0
            else:
                alpha_val = 0.3
            
            sns.scatterplot(
                data=sub, x='age', y=feat,
                ax=ax_u, color=group_colors[grp],
                s=20, alpha=alpha_val, label=None
            )
            sns.regplot(
                data=sub, x='age', y=feat,
                ax=ax_u, scatter=False,
                color=group_colors[grp], ci=None,
                line_kws={'alpha': alpha_val, 'linewidth':3.5}
            )
            lbl = f"{grp} {stars} (r={r:.2f}, p={p:.2g})"
            line = mlines.Line2D([], [], color=group_colors[grp], label=lbl, linewidth=2, alpha=alpha_val)
            lines_for_legend.append(line)
        
        ax_u.set_title(f"Unharmonized", fontweight="bold")
        ax_u.set_xlabel("Age", fontweight="bold")
        ax_u.set_ylabel(feat, fontweight="bold")
        ax_u.grid(False)
        ax_u.legend(handles=lines_for_legend, title="Group", loc='best', prop={'weight':'bold'})
        plt.setp(ax_u.get_xticklabels(), fontweight="bold")
        plt.setp(ax_u.get_yticklabels(), fontweight="bold")
        
        # --- Fila 2: harm
        ax_h = axes[1, j]
        col_harm = "harm_" + feat
        if col_harm not in df_harm.columns:
            ax_h.set_title(f"{feat} (Harmonized)\n[No col {col_harm}]", fontweight="bold")
            ax_h.set_xlabel("Age", fontweight="bold")
            ax_h.set_ylabel(feat, fontweight="bold")
            ax_h.grid(False)
            continue
        
        df_h = df_harm[['age','group',col_harm]].dropna().rename(columns={col_harm: feat})
        
        lines_for_legend_h = []
        for grp in groups:
            sub = df_h[df_h['group']==grp]
            row_match = harm_df[(harm_df['Feature']==feat)&(harm_df['Group']==grp)]
            if len(row_match)==1:
                r = row_match.iloc[0]['Spearman_r']
                p = row_match.iloc[0]['p_value']
            else:
                r, p = np.nan, np.nan
            stars = significance_stars(p)
            
            if feat == best_feat_harm and grp == best_group_harm and p<0.05:
                alpha_val = 1.0
            else:
                alpha_val = 0.3
            
            sns.scatterplot(
                data=sub, x='age', y=feat,
                ax=ax_h, color=group_colors[grp],
                s=20, alpha=alpha_val, label=None
            )
            sns.regplot(
                data=sub, x='age', y=feat,
                ax=ax_h, scatter=False,
                color=group_colors[grp], ci=None,
                line_kws={'alpha': alpha_val, 'linewidth':3.5}
            )
            
            lbl = f"{grp} {stars} (r={r:.2f}, p={p:.2g})"
            line = mlines.Line2D([], [], color=group_colors[grp], label=lbl, linewidth=2, alpha=alpha_val)
            lines_for_legend_h.append(line)
        
        ax_h.set_title(f"Harmonized", fontweight="bold")
        ax_h.set_xlabel("Age", fontweight="bold")
        ax_h.set_ylabel(feat, fontweight="bold")
        ax_h.grid(False)
        ax_h.legend(handles=lines_for_legend_h, title="Group", loc='best', prop={'weight':'bold'})
        plt.setp(ax_h.get_xticklabels(), fontweight="bold")
        plt.setp(ax_h.get_yticklabels(), fontweight="bold")
    
    # fig.suptitle(
    #     "Scatter Plot + Regression Lines (Age vs. Feature)\n",
    #     fontsize=13, fontweight="bold"
    # )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    # Guardar figura
    fig_path = os.path.join(out_dir, f"{filename}.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    # 3) Guardar resultados en Excel
    # Unificamos unharm_df y harm_df
    combined_df = pd.concat([unharm_df, harm_df], ignore_index=True)
    # Añadir columna de Significance
    combined_df['Significance'] = combined_df['p_value'].apply(significance_stars)
    
    excel_path = os.path.join(out_dir, excel_filename)
    
    if os.path.exists(excel_path):
        # Intenta cargar el archivo existente
        try:
            existing_df = pd.read_excel(excel_path, sheet_name=sheet_name)
            combined_df = pd.concat([existing_df, combined_df], ignore_index=True)
            with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                combined_df.to_excel(writer, sheet_name=sheet_name, index=False)
           
        except IndexError:
            print(f"[!] El archivo '{excel_path}' está corrupto (sin hojas visibles). Se sobrescribirá.")
            os.remove(excel_path)
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                combined_df.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        # Si no existe, lo crea desde cero
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            combined_df.to_excel(writer, sheet_name=sheet_name, index=False)




def plot_average_spectrum_by_group(data, freqs_ref, component, component_name, electrode, ax):
    """
    Gráfica el promedio del componente (psd, aperiodic_fit, oscillatory_fit)
    para un electrodo específico, separando por grupo (solo HC y MCI).
    """
    # Normalizar nombres de grupos
    
    # Filtrar solo los grupos HC y MCI
    data = data[data['group'].isin(['HC', 'MCI'])]
    
    groups = data['group'].unique()
    
    for group in groups:
        group_data = data[data['group'] == group]
        subjects = group_data['subject'].unique()
        all_group_data = []
        
        for subject in subjects:
            subject_data = group_data[group_data['subject'] == subject]
            
            if electrode in subject_data['Sensors'].values:
                electrode_data = subject_data[subject_data['Sensors'] == electrode]
                freqs = electrode_data['freqs'].values[0]
                component_data = electrode_data[component].values[0]
                interpolated_data = np.interp(freqs_ref, freqs[:len(component_data)], component_data)
                all_group_data.append(interpolated_data)
        
        if len(all_group_data) > 0:
            average_group_data = np.mean(all_group_data, axis=0)
            
            # Trazar cada sujeto
            for subject_data in all_group_data:
                if len(all_group_data) > 100:
                    alpha = 0.05
                else:
                    alpha = 0.2
                ax.plot(freqs_ref, subject_data, color='gray', alpha=alpha)
            
            # Trazar el promedio del grupo
            ax.plot(freqs_ref, average_group_data, label=f'group {group} (n={len(all_group_data)})', linewidth=3)
    
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power')
    ax.set_title(f'{component_name} - {electrode}')
    ax.set_xlim([1.5, 40])
    if component == 'psd':
        ax.set_yscale('log')
    ax.grid(False)
    ax.legend()


def plot_component_grid_by_group(data, db_name):
    """
    Genera una figura con 4 filas (canales) y 3 columnas (componentes):
    PSD, Oscillatory Fit y Aperiodic Fit.
    """
    # Componentes y sus nombres
    components = [
        ('psd', 'PSD'),
        ('oscilatory_psd', 'Oscillatory Fit'),
        ('aperiodic_comp', 'Aperiodic Fit')
    ]

    # Frecuencias de referencia
    freqs_ref = np.linspace(1, 40, 200)

    # Electrodos de interés (en orden específico si lo necesitas)
    electrodes = ['FP1','FP2','C3','C4','O1', 'O2', 'P7', 'P8']

    # Crear figura
    fig, axes = plt.subplots(len(electrodes), len(components), figsize=(18, 12))
    axes = axes.reshape(len(electrodes), len(components))  # Asegurar forma consistente

    print(f"Generando figura para: {db_name}")
    for row_idx, electrode in enumerate(electrodes):
        for col_idx, (component, component_name) in enumerate(components):
            ax = axes[row_idx, col_idx]
            plot_average_spectrum_by_group(data, freqs_ref, component, component_name, electrode, ax)

    # Título general
    fig.suptitle(f'{db_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Deja espacio para el título
    plt.savefig(rf'D:\MulticentersEEG\graphs_thesis_LMZS\Spectrum_{db_name}.pdf', dpi=600, format='pdf', bbox_inches='tight', transparent=True)
    plt.show()
