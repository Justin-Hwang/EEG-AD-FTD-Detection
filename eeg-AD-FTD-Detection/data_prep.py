"""
This script is used to prepare the data for the model, and it is used to split the data into train and test.
Preprocess EEG Data (clean, crop, resample, segmentation) and create .set eeg files and JSON labels
"""


"""
1. Loads each subject’s EEG .set file.
2. Crops 30 seconds from beginning & end.
3. Resamples EEG from 500 Hz → 95 Hz.
4. Splits each subject's EEG into 15-second segments (1425 timepoints).
5. Randomly samples some training chunks into a test set (for within-subject testing).
6. Saves .set chunks into model-data/ and creates a labels.json file with:

- file_name
- label: 'A' (Alzheimer), 'C' (Control), 'F' (FTD)
- type: 'train', 'test_cross', or 'test_within'
"""

import os
import shutil
import csv
import json
import pandas as pd
import mne
import warnings
import random

# Ignore RuntimeWarning
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Enable CUDA
mne.utils.set_config('MNE_USE_CUDA', 'true')
mne.cuda.init_cuda(verbose=True)

# Set random seed
random.seed(42)


"""
This function is used to prepare the data for the model, and it is used to split the data into train and test.
:param participants_tsv: The path to the participants.tsv file
:param data_dir: The path to the derivatives directory
:param intermediate_dir: The path to the intermediate data directory (full EEGs will be copied + renamed + cleaned)
:param output_dir: The path to the model data directory (final 15 second EEG chunks go)
:param intm_train_dir: The path to the intermediate train data directory
:param intm_test_dir: The path to the intermediate test data directory
:param train_dir: The path to the train data directory (final train output directory for chunked data)
:param test_dir: The path to the test data directory (final test output directory for chunked data)
:param split_size: The size to split the data
:param flag: A boolean to indicate if the output directory should be deleted
"""



""" 
This function:
1. Reads the TSV file with subject info.
2. Loads each .set EEG file.
3. Preprocesses it.
4. Splits it into chunks of length split_size.
5. Outputs train/test EEG segments into .set files + a labels.json.
"""

def data_prep(participants_tsv, data_dir, intermediate_dir, output_dir, intm_train_dir, intm_test_dir
              , train_dir, test_dir, split_size, flag=True):
    # Clear old intermediate/output data if needed
    # If flag=False, this deletes any old files so you can regenerate from scratch.
    try:
        if not flag:
            if os.path.exists(intermediate_dir):
                shutil.rmtree(intermediate_dir)
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
        # Read participants.tsv and parse subject IDs + labels
        tsv_content = []
        with open(participants_tsv, 'r') as tsv_file:
            reader = csv.reader(tsv_file, delimiter='\t')
            for row in reader:
                tsv_content.append(row)
        # Loads the file into a DataFrame and keeps only participant_id and Group
        df = pd.DataFrame(tsv_content[1:], columns=tsv_content[0])
        df = df[['participant_id', 'Group']]

        # Split into Train and Test subjects (80%/20%)
        train = df.sample(frac=0.8, random_state=42)
        test = df.drop(train.index)

        # Print how many A/C/F samples in each split
        print(f'Intermediate Train data distribution:\n{train["Group"].value_counts()}')
        print(f'Intermediate Test data distribution:\n{test["Group"].value_counts()}')

        # Create folders if they don't exist
        if not os.path.exists(intermediate_dir):
            os.makedirs(intermediate_dir)
        if not os.path.exists(intm_train_dir):
            os.makedirs(intm_train_dir)
        if not os.path.exists(intm_test_dir):
            os.makedirs(intm_test_dir)

        # Copy and rename .set EEG files to intermediate folder
        create_model_dir(train, data_dir, intm_train_dir)
        print('Intermediate Train data saved successfully')

        create_model_dir(test, data_dir, intm_test_dir)
        print('Intermediate Test data saved successfully')

        # Generate initial labels.json with metadata
        train_json_data = create_json_structure(train, 'train', intermediate_dir)
        test_json_data = create_json_structure(test, 'test_cross', intermediate_dir)

        with open(os.path.join(intermediate_dir, 'labels.json'), 'w') as label_file:
            json.dump(train_json_data + test_json_data, label_file, indent=4)

        print("Intermediate JSON file created successfully")

        # Create final output folders (model-data/train/, etc.)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        # Load intermediate labels JSON
        with open(os.path.join(intermediate_dir, 'labels.json'), 'r') as file:
            intermediate_data = json.load(file)

        main_json = []
        train_count_a = []
        test_count_a = []
        test_count_w_a = []
        train_count_c = []
        test_count_c = []
        test_count_w_c = []
        train_count_f = []
        test_count_f = []
        test_count_w_f = []

        # Loop over each full-length EEG and segment it
        for data in intermediate_data:
            file_name = data['file_name']
            label = data['label']
            type_ = data['type']
            raw = mne.io.read_raw_eeglab(os.path.join(intermediate_dir, file_name)) # Load the EEG
            raw = raw.crop(tmin=30, tmax=raw.times[-30]) # Remove first & last 30 sec
            raw = raw.resample(95) # Downsample to 95 Hz
            data_points = raw.get_data().shape[1] # Total timepoints
            num_chunks = data_points // split_size # 1424 timepoints per chunk
            print(f'File: {file_name}, Label: {label}, Type: {type_}, Timepoints: {data_points}, '
                  f'Chunks: {num_chunks}, Discarded timepoints: {(data_points % split_size) + 5700}')
            for i in range(num_chunks):
                flag = False
                # Get the chunk boundaries
                start = i * split_size
                end = (i + 1) * split_size
                # Crop that time range
                chunk = raw.copy().crop(tmin=start / raw.info['sfreq'], tmax=end / raw.info['sfreq'])
                chunk_file_name = f'{file_name.split(".")[0]}_chunk_{i}.set'
                # Inject 10% of training chunks into "within-subject test"
                # Test on other chunks from the same person
                # Not used for training, but not fully cross-subject either
                if type_ == 'train' and random.random() < 0.1:
                    flag = True
                    chunk_file_name = chunk_file_name.replace('train', 'test')
                chunk.export(os.path.join(output_dir, chunk_file_name), overwrite=True) # Save chunk as .set file
                # Save metadata entry for JSON
                entry = {
                    "file_name": chunk_file_name,
                    "label": label,
                    "type": type_ if not flag else 'test_within',
                    "num_channels": chunk.info['nchan'],
                    "timepoints": chunk.get_data().shape[1],
                    "total_time (in seconds)": chunk.get_data().shape[1] / chunk.info['sfreq']
                }
                # This main_json stores metadata for every EEG chunk in a list called main_json
                main_json.append(entry)
                if entry.get("type") == 'train':
                    if label == 'A':
                        train_count_a.append(entry)
                    elif label == 'C':
                        train_count_c.append(entry)
                    else:
                        train_count_f.append(entry)
                elif entry.get("type") == 'test_cross':
                    if label == 'A':
                        test_count_a.append(entry)
                    elif label == 'C':
                        test_count_c.append(entry)
                    else:
                        test_count_f.append(entry)
                else:
                    if label == 'A':
                        test_count_w_a.append(entry)
                    elif label == 'C':
                        test_count_w_c.append(entry)
                    else:
                        test_count_w_f.append(entry)
        # Save all metadata into labels.json for the model to use later
        with open(os.path.join(output_dir, 'labels.json'), 'w') as label_file:
            json.dump(main_json, label_file, indent=4)

        print("\nSplit data and JSON saved successfully")

        print(f'\nTrain data distribution:\n'
              f'A: {len(train_count_a)}, C: {len(train_count_c)}, F: {len(train_count_f)}'
              f', Total: {len(train_count_a) + len(train_count_c) + len(train_count_f)}')
        print(f'Test data distribution (cross subject):\n'
              f'A: {len(test_count_a)}, C: {len(test_count_c)}, F: {len(test_count_f)}'
              , f'Total: {len(test_count_a) + len(test_count_c) + len(test_count_f)}')
        print(f'Test data distribution (within subject):\n'
              f'A: {len(test_count_w_a)}, C: {len(test_count_w_c)}, F: {len(test_count_w_f)}'
              , f'Total: {len(test_count_w_a) + len(test_count_w_c) + len(test_count_w_f)}')
        print(f'\nTotal distribution:\n'
              f'A: {len(train_count_a) + len(test_count_a) + len(test_count_w_a)}, '
              f'C: {len(train_count_c) + len(test_count_c) + len(test_count_w_c)}, '
              f'F: {len(train_count_f) + len(test_count_f) + len(test_count_w_f)}, '
              f'Total: {len(main_json)}')

    except Exception as e:
        print(f'Error: {e}')

# Creates the metadata entries for EEG file (before chunking) and helps generate part of the labels.json file
def create_json_structure(df, data_type, output_dir):
    json_data = [] # This will store the list of metadata entries
    file_data = data_type.split('_')[0] # 
    for _, row in df.iterrows():
        file_name = f"{file_data}/{row['participant_id']}_eeg.set" # Build the EEG file path:
        label = row['Group'] # 'A', 'C', or 'F'
        raw = mne.io.read_raw_eeglab(os.path.join(output_dir, file_name))
        # Create the metadata entry
        entry = {
            "file_name": file_name,
            "label": label,
            "type": data_type,
            "num_channels": raw.info['nchan'],
            "timepoints": raw.get_data().shape[1],
            "total_time (in seconds)": raw.get_data().shape[1] / raw.info['sfreq']
        }
        json_data.append(entry)
    return json_data

# collecting the raw/preprocessed EEG files for each participant 
# and organizing them into a simpler, cleaner directory structure.
def create_model_dir(data, main_dir, target_dir):
    for index, row in data.iterrows():
        participant_id = row['participant_id']
        file_path = f'{main_dir}/{participant_id}/eeg/{participant_id}_task-eyesclosed_eeg.set'
        target_name = f'{participant_id}_eeg.set'
        target_path = os.path.join(target_dir, target_name)

        if os.path.exists(file_path):
            shutil.copy(file_path, target_path)
        else:
            print(f'File not found: {file_path}')
            raise Exception(f'File not found: {file_path}')

# Define the directories
d1 = '../participants.tsv' # Contains participant metadata
d2 = '../derivatives' # Containing preprocessed EEG files
d3 = './intermediate-data' # Simplified EEG files
d4 = '../model-data' # Final preprocessed and chunked EEG files
d5 = os.path.join(d3, 'train') # Full-length EEGs for train subjects
d6 = os.path.join(d3, 'test') # Full-length EEGs for test subjects
d7 = os.path.join(d4, 'train') # Chunked EEG files for train
d8 = os.path.join(d4, 'test') # Chunked EEG files for test

# 95 * number of seconds you want - 1
d9 = 1424
data_prep(d1, d2, d3, d4, d5, d6, d7, d8, d9, False)
