import pandas as pd
import numpy as np
from neuroCombat import neuroCombat
from neuroHarmonize import harmonizationLearn, harmonizationApply
from reCombat import reComBat # you need python <3.10 to use this library

# Función principal para aplicar la armonización
def apply_harmonization(method, data, covars, batch_column='batch'):
    """
    Aplica el método de armonización especificado a los datos.

    Parámetros:
    - method: str, 'neurocombat', 'neuroharmonize', o 'recombat'.
    - data: DataFrame, contiene las características a armonizar.
    - covars: DataFrame, contiene las covariables (incluyendo 'batch').
    - batch_column: str, nombre de la columna que indica los lotes.

    Devuelve:
    - DataFrame con los datos armonizados.
    """
    if method.lower() == 'neurocombat':
        print("Usando neuroCombat...")
        batch = covars[batch_column]
        mod = covars.drop(columns=[batch_column])  # Covariables adicionales
        harmonized_data = neuroCombat(data=data.T, covars=mod, batch_col=batch_column)["data"]
        return harmonized_data.T

    elif method.lower() == 'neuroharmonize':
        print("Usando neuroHarmonize con covariables no lineales...")
        covars = covars.copy()  # Evitar modificaciones accidentales
        covars[batch_column] = covars[batch_column].astype(str)  # NeuroHarmonize requiere strings para batch
        model, params = harmonizationLearn(data.values, covars, batch_col=batch_column, smooth_terms=covars.columns)
        harmonized_data = harmonizationApply(data.values, model)
        return pd.DataFrame(harmonized_data, index=data.index, columns=data.columns)

    elif method.lower() == 'recombat':
        print("Usando reComBat con configuración avanzada...")
        model = reComBat(parametric=True,
                         model='ridge',
                         config={'alpha': 1e-9},
                         conv_criterion=1e-4,
                         max_iter=1000,
                         n_jobs=1,
                         mean_only=False,
                         optimize_params=True,
                         reference_batch=None,
                         verbose=True)
        batch = covars[batch_column].values
        mod = covars.drop(columns=[batch_column]).values  # Covariables adicionales
        harmonized_data = model.fit_transform(data.values, batch=batch, covars=mod)
        return pd.DataFrame(harmonized_data, index=data.index, columns=data.columns)

    else:
        raise ValueError("Método desconocido. Usa 'neurocombat', 'neuroharmonize' o 'recombat'.")

# Ejemplo de uso:
if __name__ == "__main__":
    # Datos de ejemplo
    np.random.seed(42)
    data = pd.DataFrame(np.random.rand(100, 10), columns=[f"Feature_{i+1}" for i in range(10)])
    covars = pd.DataFrame({
        'batch': np.random.choice(['A', 'B', 'C'], size=100),
        'age': np.random.randint(20, 70, size=100),
        'sex': np.random.choice(['M', 'F'], size=100)
    })

    # Seleccionar el método y aplicar armonización
    method = 'recombat'  # Cambiar a 'neurocombat' o 'neuroharmonize' según necesidad
    harmonized_data = apply_harmonization(method, data, covars)

    # Mostrar resultados
    print("Datos armonizados:")
    print(harmonized_data.head())
