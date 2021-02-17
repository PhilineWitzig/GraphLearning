"""
Contains executable code for creation of embeddings.
"""
import argparse
import os
import numpy as np
from node2vec import train_node2vec


def store_embeddings(embeddings, out_file):
    """
    Stores embeddings and corresponding labels in suitable format under the given file name.
    """
    np.save(out_file, embeddings)


def load_embeddings(in_file):
    """
    Loads embeddings from the given file.
    """
    return np.load(in_file)


def main(args):
    out_dir = args.output_dir
    dataset = args.dataset
    param_return = args.param_return
    param_in_out = args.param_in_out
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    outfile = os.path.join(out_dir, dataset + "_p_" + str(param_return) + "_q_" + str(param_in_out) + '_embeddings')
    embeddings = train_node2vec(dataset, param_return, param_in_out)
    store_embeddings(embeddings, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['CiteSeer', 'Cora', 'Facebook', 'PPI'],
                        help="Dataset for which to create node embeddings.")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory in which the embeddings are stored.")
    parser.add_argument('--param_return', required=True, type=float,
                        help="Bias of going back to the old node (p)")
    parser.add_argument('--param_in_out', required=True, type=float,
                        help="Bias of moving forward to a new node (q)")
    args = parser.parse_args()
    main(args)
