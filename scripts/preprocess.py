import cc_tmcn.preprocessing.padding as padding
from pathlib import Path
import argparse
import h5py

ap = argparse.ArgumentParser()
ap.add_argument("--training-data-path", required=True, help="Path to the training data")
ap.add_argument("--test-data-path", required=True, help="Path to the test data")
ap.add_argument("--config-path", required=True, help="Path to the config for preprocessing")
ap.add_argument("--output-folder", required=True, help="Path to the folder, where the padded CSI measurements should be stored")

if __name__ == "__main__":

    args = ap.parse_args()

    # Load training and test data
    training_data = h5py.File(Path(args.training_data_path), "r")
    test_data = h5py.File(Path(args.test_data_path), "r")
    
    # Format data for preprocessing
    training_data = padding.format(training_data)
    test_data = padding.format(test_data)

    data = {'training_data':training_data,
            'test_data':test_data }
            
    # Preprocess data
    padding.preprocess_data(data, Path(args.config_path), Path(args.output_folder))