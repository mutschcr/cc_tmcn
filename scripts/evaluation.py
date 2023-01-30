import cc_tmcn.model.model_factory as model_factory
import pickle
from pathlib import Path
import argparse

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

import numpy as np
from matplotlib.ticker import FormatStrFormatter
from sklearn.manifold import trustworthiness

def least_squares_transform(cc_data, pos_data):

    # Pad the data with ones, so that our transformation can do translations too
    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
    unpad = lambda x: x[:,:-1]
    X = pad(cc_data)
    Y = pad(pos_data)

    A, res, rank, s = np.linalg.lstsq(X, Y)

    transform = lambda x: unpad(np.dot(pad(x), A))

    return transform(cc_data)

def _make_cols(xs: np.ndarray, ys: np.ndarray):

    color1 = (xs - min(xs)) / max(xs - min(xs))
    color2 = (ys - min(ys)) / max(ys - min(ys))
    color3 = np.zeros(len(color2))
    return np.concatenate([color1[:, None], color2[:, None], color3[:, None]], axis=-1)

def plot_channel_chart(ue_xs, ue_ys, embedding_xs, embedding_ys, cc_suptitle=""):

    assert len(ue_xs) == len(ue_ys) and ue_xs.shape == ue_ys.shape and len(ue_ys.shape) == 1
    assert embedding_xs.shape == ue_xs.shape and embedding_ys.shape == ue_ys.shape

    c = _make_cols(ue_xs, ue_ys)
    fig1 = plt.figure(figsize=(4, 3))

    ax = fig1.add_subplot()
    ax.set_title("Radio environment")
    ax.scatter(ue_xs, ue_ys, c=c, s=4., marker='o', rasterized=True)

    ax.set_ylabel("y [m]")
    ax.set_xlabel('x [m]')
    
    fig1.tight_layout()

    fig2 = plt.figure(figsize=(4, 3))
    ax = fig2.add_subplot()

    ax.scatter(embedding_xs, embedding_ys, c=c, marker='o', s=4., rasterized=True)
    ax.set_title(cc_suptitle)
  
    ax.set_ylabel("y [m]")
    ax.set_xlabel("x [m]")
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    
    fig2.tight_layout()
    return fig1, fig2

ap = argparse.ArgumentParser()
ap.add_argument("--data-path", required=True, help="Path to the padded data")
ap.add_argument("--model-name", required=True, help="Name of the model")
ap.add_argument("--model-path", required=True, help="Path to the model")
ap.add_argument('--report-name', required=True, type=str, help='Report file path (*.pdf)')

if __name__ == "__main__":

    args = ap.parse_args()

    data_path = args.data_path
    model_path = args.model_path

    training_data = list(pickle.load(open(data_path, "rb")))

    model = model_factory.load_model(args.model_name, model_path)
    input_data = training_data[0]
    pos = training_data[1]

    X_embedded = model.predict(input_data)

    # 5% of the nearest neighbors should be used for KPIs
    k_mul = 0.05    
    tw = trustworthiness(pos, X_embedded, n_neighbors=int(len(X_embedded) * k_mul))
    ct = trustworthiness(X_embedded, pos, n_neighbors=int(len(X_embedded) * k_mul))

    print("CT: {:3f}, TW: {:3f}".format(ct, tw))

    # Plot the channel chart
    _, fig4 = plot_channel_chart(pos[:,0], pos[:,1], X_embedded[:,0], X_embedded[:,1])

    # Estimate and apply linear transformation based on ground truth
    X_embedded = least_squares_transform(X_embedded, pos)

    error = np.linalg.norm(X_embedded - pos, axis=1)
    CE90 = np.percentile(error, 90)
    print("CE90: %.2f" % CE90)

    # Plot the channel chart
    fig1, fig2 = plot_channel_chart(pos[:,0], pos[:,1], X_embedded[:,0], X_embedded[:,1], cc_suptitle="CT: {:.3f}, TW: {:.3f}, CE90: {:.3f}".format(ct, tw, CE90))

    # Create folder for the report if not available
    report_path = Path(args.report_name)
    report_path.parents[0].mkdir(parents=True, exist_ok=True)

    # Store report
    with PdfPages(report_path) as pdf:
        pdf.savefig(fig1)
        pdf.savefig(fig2)
        pdf.savefig(fig4)

    plt.show()