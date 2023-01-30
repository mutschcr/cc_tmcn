from cc_tmcn.model import model_factory
import pickle
from pathlib import Path
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--data-path", required=True, help="Path to the training data folder, with the distance matrix")
ap.add_argument("--model-name", required=True, help="Name of the model")
ap.add_argument("--model-path", required=True, help="Path to the model")

if __name__ == "__main__":

    args = ap.parse_args()

    data_path = args.data_path
    model_path = args.model_path

    training_data = pickle.load(open(data_path / Path("training_data"), "rb"))
    model = model_factory.create_model(args.model_name, model_path)

    model.train(training_data)