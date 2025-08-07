'''
@Luisa María Zapata Saldarriaga
'''

import os
import mne
import pandas as pd

# Definir la ruta principal
base_raw_path = r"F:\EEG_MULTICENTER\CHMP"

# Obtener la lista de sujetos
subjects = [d for d in os.listdir(base_raw_path) if os.path.isdir(os.path.join(base_raw_path, d))]

# Procesar cada sujeto
for subject in subjects:
    try:
        # Definir las rutas para cada sujeto
        raw_file = os.path.join(base_raw_path, subject, "eeg", f"{subject}_task-protmap_eeg.edf")
        events_file = os.path.join(base_raw_path, subject, "eeg", f"{subject}_task-protmap_events.tsv")
        output_file = os.path.join(base_raw_path, subject, "eeg",  f"{subject}_task-CE_eeg.edf")

        # Verificar si los archivos existen
        if not (os.path.exists(raw_file) and os.path.exists(events_file)):
            print(f"Archivos faltantes para el sujeto {subject}.")
            continue

        # Cargar el archivo raw
        raw = mne.io.read_raw_edf(raw_file, preload=True)

        # Cargar los eventos
        events_data = pd.read_csv(events_file, sep='\t')
        events_data['end_time'] = events_data['onset'].shift(-1)

        # Filtrar eventos de "ojos cerrados"
        CE_events = events_data[events_data['trial_type'] == 'ojos cerrados']
        CE_events['duration_inferred'] = CE_events['end_time'] - CE_events['onset']

        # Verificar si hay eventos CE
        if CE_events.empty:
            print(f"No hay eventos 'ojos cerrados' para el sujeto {subject}.")
            continue

        # Extraer y concatenar segmentos de "ojos cerrados"
        concatenated_data = []
        for _, event in CE_events.iterrows():
            segment = raw.copy().crop(tmin=event['onset'], tmax=event['end_time'])
            concatenated_data.append(segment.get_data())

        # Concatenar todos los segmentos de "ojos cerrados"
        concatenated_data = mne.concatenate_raws(
            [mne.io.RawArray(data, raw.info) for data in concatenated_data]
        )

        # Guardar el archivo concatenado como EDF
        mne.export.export_raw(output_file, concatenated_data, fmt='edf', overwrite=True)
        print(f"Archivo procesado y concatenado guardado como EDF: {output_file}")

    except Exception as e:
        print(f"Error procesando al sujeto {subject}: {e}")
