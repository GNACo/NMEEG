import mne
import json
import os
import numpy as np

# Clase EegEyeClosedCrop
class EegEyeClosedCrop:
    def __init__(self, jitter=50):
        self.jitter = jitter

    def __call__(self, sample):
        signal = sample['signal']
        first_close_eyes = sample['first_close_eyes']
        first_open_eyes = sample['first_open_eyes']

        # Recortar entre el primer 'Eyes Closed' y el primer 'Eyes Open'
        crop_start = first_close_eyes
        crop_end = first_open_eyes
        cropped_signal = signal[:, crop_start:crop_end]

        if cropped_signal.shape[1] == 0:
            raise ValueError("El segmento de señal recortado está vacío.")

        # Guardar la señal recortada en sample
        sample['cropped_signal'] = cropped_signal
        return sample

def rename_channels(raw):
    new_channel_names = {}
    for ch_name in raw.info['ch_names']:
        new_name = ch_name.replace('-AVG', '').upper()  # Eliminar '-AVG' y poner en mayúsculas
        new_channel_names[ch_name] = new_name
    raw.rename_channels(new_channel_names)
    
# Función para procesar el EDF y las anotaciones del JSON
def process_eeg(edf_file, json_file, cropper, output_dir):
    # Cargar el archivo EDF utilizando MNE
    raw = mne.io.read_raw_edf(edf_file, preload=True)
    signal = raw.get_data()
    rename_channels(raw)

    # Cargar las anotaciones JSON
    with open(json_file, 'r') as f:
        events = json.load(f)

    # Buscar el primer evento "Eyes Closed" y el primer evento "Eyes Open"
    first_close_eyes = None
    first_open_eyes = None
    for i, event in enumerate(events):
        if event[1].lower() == 'eyes closed' and first_close_eyes is None:
            first_close_eyes = event[0]  # Guardar el tiempo de "Eyes Closed"
        elif event[1].lower() == 'eyes open' and first_close_eyes is not None:
            first_open_eyes = event[0]  # Guardar el tiempo de "Eyes Open"
            break

    if first_close_eyes is None or first_open_eyes is None:
        print(f"No se encontró un evento adecuado de 'Eyes Closed' seguido de 'Eyes Open' en {edf_file}.")
        return None

    # Crear la muestra con la señal y la información del evento
    sample = {
        'signal': signal,
        'first_close_eyes': first_close_eyes,  # Tiempo del primer "Eyes Closed"
        'first_open_eyes': first_open_eyes     # Tiempo del primer "Eyes Open"
    }

    try:
        # Aplicar la lógica de recorte
        cropped_sample = cropper(sample)

        # Crear un nuevo objeto RawArray con la señal recortada
        info = raw.info  # Reutilizar la información original (canales, frecuencia de muestreo, etc.)
        print(info.ch_names)
        # Corregir el problema de meas_date usando set_meas_date
        info.set_meas_date(None)  # Eliminar la fecha de medición para evitar el error de rango

        cropped_signal = cropped_sample['cropped_signal']
        
        # Crear nuevo RawArray con la señal recortada
        new_raw = mne.io.RawArray(cropped_signal, info)

        # Guardar el archivo .fif en el directorio de salida
        output_fif_file = os.path.join(output_dir, os.path.basename(edf_file).replace('.edf', '.fif'))
        new_raw.save(output_fif_file, overwrite=True)

        print(f"Procesado {edf_file} con éxito. Guardado en {output_fif_file}.")
        print(f"El primer evento 'Eyes Closed' comienza en {first_close_eyes} y termina en {first_open_eyes}, con una duración de {first_open_eyes - first_close_eyes} muestras.")
        return cropped_sample
    except ValueError as e:
        print(f"Error procesando {edf_file}: {e}")
        return None

# Directorios
input_dir = "D:\MulticentersEEG\caueeg-dataset\signal\edf"
json_dir = "D:\MulticentersEEG\caueeg-dataset\event"
output_dir = "D:\MulticentersEEG\caueeg-dataset-ce\signal"

# Crear el directorio de salida si no existe
os.makedirs(output_dir, exist_ok=True)

# Instanciar el cropper
cropper = EegEyeClosedCrop(jitter=50)

# Procesar todos los archivos EDF y sus correspondientes JSON
for edf_file in os.listdir(input_dir):
    if edf_file.endswith('.edf'):
        edf_path = os.path.join(input_dir, edf_file)
        json_path = os.path.join(json_dir, edf_file.replace('.edf', '.json'))
        
        if os.path.exists(json_path):
            process_eeg(edf_path, json_path, cropper, output_dir)
        else:
            print(f"No se encontró archivo JSON para {edf_file}.")
