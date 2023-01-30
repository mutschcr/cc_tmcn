from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import argparse

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path
import time

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', required=True, type=str, metavar='PATH',
						help='Path to input data')

parser.add_argument('--output-path', required=True, type=str,   
						help='Name of the folder, the distance matrix file has to be stored.')

if __name__ == "__main__":
    args = parser.parse_args()

    data = pd.read_pickle(Path(args.data_path))

    # Vectorize input data
    data_flat = data[0].reshape((data[0].shape[0], data[0].shape[1]*data[0].shape[2]))
    metric = "cityblock"

    n_neighbors = 5

    start = time.time()
    # Get nearest neighbors
    nbrs = NearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm="auto",
        metric=metric,
        p=2,
        n_jobs=-1)

    nbrs.fit(data_flat)
    print("NN: %.8f" % (time.time() - start))

    start = time.time()
    # Create neighborhood graph
    nbg = kneighbors_graph(nbrs,
                            n_neighbors,
                            metric=metric,
                            p=2,
                            mode="distance",
                            n_jobs=-1)

    print("Graph: %.8f" % (time.time() - start))

    start = time.time()
    # Calculate the shortest geodesic distances
    dist_matrix = shortest_path(nbg, method="auto", directed=False)

    print("Shortest path: %.8f" % (time.time() - start))

    # Create folder for the report if not available
    output_path = Path(args.output_path)
    
    output_path = output_path / Path("training_data")
    output_path.parents[0].mkdir(parents=True, exist_ok=True)

    # Normalize distances for better training convergence
    dist_matrix = dist_matrix / np.percentile(dist_matrix, 90)

    # Store distance matrix with the corresponding CSI input tensors
    with open(output_path, "wb") as output_file:
        pickle.dump((dist_matrix, data[0]), output_file)

