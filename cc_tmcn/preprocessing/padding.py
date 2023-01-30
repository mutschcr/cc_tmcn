import pandas as pd
pd.options.mode.chained_assignment = None

import numpy as np
import xml.etree.ElementTree as ET
import os
import pickle
from pathlib import Path
from joblib import Parallel, delayed

def format(data):
    data_frame = pd.DataFrame()
    data_frame["A_ID"] = list(data["A_ID"])
    data_frame["B_ID"] = list(data["B_ID"])
    data_frame["TD"] = data["TD"]
    data_frame["td_offset"] = data["TD_OFFSET"]
    data_frame["CIR_R"] = list(data["CIR_R"])
    data_frame["CIR_I"] = list(data["CIR_I"])
    data_frame["Ref_x"] = data["POS_X"]
    data_frame["Ref_y"] = data["POS_Y"]
    data_frame["Time_stamp"] = data["TIME_STAMP"]

    return data_frame

def _get_params(config_file_path):
    """
    Gets the parameters of the input data of the given XML object.

    Parameters
    ----------
    config : string
        Path to config file

    Returns
    -------
    cir_time_res : float
        Time resolution of the channel impulse response (CIR)
    anchor_count : int
        Number of anchors.
    td_value_range : [min_td, max_td]
        Value range which defined the area of valid values of TDOA / TOA values.

    """
    config = ET.parse(config_file_path)
    config_root = config.getroot()

    cir_time_res = None
    anchor_count = None
    
    # Gets the attributes from the xml tree
    for child in config_root:
        if child.tag == "CIR_time_res":
            cir_time_res = float(child.attrib['value'])
            
        if child.tag == "Anchor_count":
            anchor_count = int(child.attrib['value'])

        if child.tag == "TD_value_range":
            td_value_range = (float(child.attrib['min']), float(child.attrib['max']))

    return cir_time_res, anchor_count, td_value_range

def preprocess_data(data, config_file_path, output_folder):
    """
    Preprocess data and store the preprocessed data in the given output folder.
    The data is stored as pickle file with tuples of (data, labels, record_times) for every dataset.

    Parameters
    ----------
    data : dictonary
        Dictionary of { dataset_name : formatted_dataset }
    config_file_path : String
        Path to the parameters file for the formatted data
    output_folder : String
        Path for the output directory
        
    Returns
    -------
    None.

    """
    # Make path strings OS independent
    config_file_path = Path(config_file_path)
    output_folder = Path(output_folder)

    # Create a output directory for every evaluation scenario
    if os.path.exists(output_folder) == False:
        os.makedirs(output_folder)
    
    preprocessed_data = {}

    # Store preprocessed subsets as pickle files with labels
    for name, dataset in data.items():
        pre_sub_data, sub_labels, sub_times = preprocess(dataset, config_file_path)

        preprocessed_data[name] = [pre_sub_data, sub_labels, sub_times]
    
    # Estimate normalization factor from given subsets
    normalization_factor = -float('inf')
    norm_factor_path = output_folder / Path("normalization_factor")

    if not norm_factor_path.is_file():
        for _,dataset in preprocessed_data.items():

            max_subset = np.amax(np.absolute(dataset[0]))
            if max_subset > normalization_factor:
                normalization_factor = max_subset
        
        pickle.dump(normalization_factor, open(norm_factor_path, "wb"))
    else:
        normalization_factor = pickle.load(open(norm_factor_path, "rb"))

    for name, preprocessed_subset in preprocessed_data.items():
        preprocessed_subset[0] = preprocessed_subset[0] / normalization_factor

        output_path = output_folder / Path(name)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as sub_data_file:
            pickle.dump(preprocessed_subset, sub_data_file)
    
    return preprocessed_data


def preprocess(data, config_file_path):
    """
    Preprocesses the data for DL positioning as 3D tensors for every burst.
    Every burst contains following data: [Anchor, CIR, Real/Imaginary]

    Parameters
    ----------
    data : Pandas dataframe
        Data in the standardized format
    config_file_path : string (Path to *.xml file)
        Path to the configuration file containing the parameters for preprocessing
    use_mag : Boolean
        Determine if real/imaginary part or Magnitude is used
    use_td : Boolean
        Determine if CIR has to be aligned by TOA/TDOA

    Returns
    -------
    processed_data : List of numpy arrays.
        Preprocessed data for DL positioning.

    labels : List [[x,y], ...]
        Labels for positioning. In this case 2D positions of the tag.

    times : List
        Timestamps of recording in seconds
        
    """

    # Get parameters for preprocessing
    cir_time_res, anchor_count, td_value_range = _get_params(config_file_path)

    first_item = data.iloc[0]
    cir_length = len(first_item.CIR_R)

    # Remove all samples, which are not within the valid range
    valid_data = data[(data.TD >= td_value_range[0]) & (data.TD <= td_value_range[1])]
    valid_data = valid_data[(valid_data.td_offset >= 0) & (valid_data.td_offset <= cir_length * cir_time_res)]

    # Remove incomplete bursts
    burst_groups = valid_data.groupby(by=["B_ID"], as_index=False)

    valid_data = burst_groups.filter(lambda x: len(x) == anchor_count)

    # Get IDs of the anchors
    anchors = valid_data.A_ID.unique()

    # Sort anchors to ensure the same order for every dataset
    anchors = sorted(anchors)

    # Get the window size of the output tensor
    # Window size is defined by the valid value range.
    window_size = int(round((td_value_range[1] - td_value_range[0]) / cir_time_res))
    
    processed_data = []
    labels = []
    times = []

    burst_groups = valid_data.groupby(by=["B_ID"], as_index=False)

    # Preprocess bursts in parallel
    result = Parallel(n_jobs=-1, verbose=10)(delayed(get_burst_data)(burst, anchors, window_size, td_value_range, cir_time_res) for _,burst in burst_groups)
    
    processed_data = np.array([item[0] for item in result])
    labels = np.array([item[1] for item in result])
    times = np.array([item[2] for item in result])

    return np.array(processed_data).astype('float32'), np.array(labels).astype('float32'), times

def get_burst_data(burst, anchors, window_size, td_value_range, cir_time_res):
    """
    Gets the burst as 2D or 3D array for fingerprinting. A padding is performed of the CIRs in a input tensor of dimension [2, num_anchors, window_size], wheres 2 is for Real/Imag.
    If the magnitude should be used, the tensor is of size [num_anchors, window_size].

    Parameters
    ----------
    burst : pandas Dataframe
        Anchor information within the burst
    use_td : Boolean
        Determine if CIR has to be aligned by TOA/TDOA
    cir_time_res : float
        Time resolution of the channel impulse response (CIR)
    anchors : list
        Anchor ids
    td_value_range : [min_td, max_td]
        Value range which defined the area of valid values of TDOA / TOA values.
    cir_time_res : float
        Resolution of the input CIR
    window_size : int
        Size of the window the CIR is embedded in the burst tensor

    Returns
    -------

    """

    anchor_count = len(anchors)

    burst_data = np.zeros((anchor_count, window_size))
        
    for a_idx, anchor in enumerate(anchors):
        
        # Get data of the current anchor within the burst
        anch_data = burst[burst.A_ID == anchor].iloc[0]
        
        # Get shift in indices of the target time resolution of the TOA peak within the CIR
        cir_td_offset = int(round((anch_data['td_offset'] * (1/cir_time_res))))

        # Get shift in time indices from TD
        td_offset = int(round((anch_data.TD - td_value_range[0]) / cir_time_res))

        cir_shifted = np.zeros(td_offset + len(anch_data.CIR_R)+window_size)

        cir_mag = np.linalg.norm(np.vstack((np.array(anch_data.CIR_R), np.array(anch_data.CIR_I))), axis=0)
        cir_shifted[td_offset:td_offset+len(anch_data.CIR_R)] = cir_mag

        burst_data[a_idx, :window_size] = cir_shifted[cir_td_offset:window_size+cir_td_offset]
    
    return (burst_data, np.array([burst.Ref_x.mean(), burst.Ref_y.mean()]), burst.Time_stamp.mean())
