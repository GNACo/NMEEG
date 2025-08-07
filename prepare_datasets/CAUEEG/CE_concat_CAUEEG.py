'''
@Author: Albeto Jaramillo Jimenez 
'''

import os
import json
import mne
import numpy as np

# Define paths
all_events_path = 'D:/MulticentersEEG/caueeg-dataset/event/'
all_signals_path = 'D:/MulticentersEEG/caueeg-dataset/signal/edf/'
output_path = 'D:/MulticentersEEG/cau_eyes_closed/'

# Ensure output path exists
os.makedirs(output_path, exist_ok=True)

# Function to process a single file pair
def process_file_pair(event_file, signal_file):
    # Load the JSON file
    with open(event_file, 'r') as f:
        events = json.load(f)

    # Extract "Eyes Closed" segments
    eyes_closed_segments = []
    is_eyes_closed = False
    start_time = 0
    sfreq = 200 

    for timestamp, event in events:
        if event == "Eyes Closed":
            start_time = timestamp/sfreq
            is_eyes_closed = True
        elif event == "Eyes Open" and is_eyes_closed:
            end_time = timestamp/sfreq
            eyes_closed_segments.append((start_time, end_time))
            is_eyes_closed = False

    # Load the EDF file
    raw = mne.io.read_raw_edf(signal_file, preload=True)


    # Extract "Eyes Closed" segments and concatenate them
    eyes_closed_data = []
    for start, end in eyes_closed_segments:
        # Events 00013.json has more timestamps than the 00013.edf total number of samples
        if end > raw.get_data().shape[1]/raw.info['sfreq']:
            pass
        else:
            segment = raw.copy().crop(tmin=start, tmax=end)  # Convert to seconds
            eyes_closed_data.append(segment)

    # Concatenate all eyes closed segments
    if eyes_closed_data:
        eyes_closed_raw = mne.concatenate_raws(eyes_closed_data)

        # Modify channel names and drop specific channels
        eyes_closed_raw.rename_channels({ch: ch.replace('-AVG', '') for ch in eyes_closed_raw.ch_names})
        #eyes_closed_raw.drop_channels(['EKG', 'Photic'])
        # Reset meas_date to avoid the RuntimeError
        eyes_closed_raw.set_meas_date(None)
        # Generate output file path
        base_name = os.path.basename(signal_file).replace('.edf', '.fif')
        output_file = os.path.join(output_path, base_name)

        # Save the concatenated segments to a new FIF file
        eyes_closed_raw.save(output_file, overwrite=True)

# Iterate over all files and process
for event_file in os.listdir(all_events_path):
    if event_file.endswith('.json'):
        signal_file = event_file.replace('.json', '.edf')
        event_file_path = os.path.join(all_events_path, event_file)
        signal_file_path = os.path.join(all_signals_path, signal_file)
        
        if os.path.exists(signal_file_path):
            process_file_pair(event_file_path, signal_file_path)
