"""
Command line interface for executing the code of assignment 2.
"""

import argparse
from train import train_eval


def main(args):
    train_eval(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, required=True,
                        help="Please specify the datasets you want the classifier to be trained on "
                             "as a comma separated list."
                             "Options are: CiteSeer, Cora, Enzymes, Nci1.")
    parser.add_argument('--classifier', type=str, required=True,
                        help="Please determine whether you want to train the graph or the node classifier. "
                             "Options are: Node and Graph.")
    args = parser.parse_args()
    args.datasets = args.datasets.split(",")
    main(args)
