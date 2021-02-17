import argparse
import config
from graphlet import compute_graphlets, load_graphlets
from closed_walk import compute_closed_walks
from svm_exec import load_datasets, run_svm
from wl_kernel import compute_wl


def main(args):
    names = [x.strip() for x in args.datasets.split(',')]
    names, datasets, labelsets = load_datasets(names)
    kernel = args.kernel

    if args.max_it is None:
        max_it = config.MAX_IT
    else:
        max_it = args.max_it

    if kernel == "graphlet":
        # vectors = compute_wl(datasets, names)
        vectors = load_graphlets(names)
    elif kernel == "wl":
        vectors = compute_wl(datasets, names)
    elif kernel == "closed_walk":
        vectors = compute_closed_walks(datasets, names)
    else:
        raise ValueError(f"{kernel} is not a supported kernel type. Use one out of 'graphlet', 'wl' and 'closed_walk'.")

    run_svm(names, vectors, labelsets, max_it=max_it)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('kernel', help="name of the kernel to use. Available are 'graphlet', 'wl' and 'closed_walk'")
    parser.add_argument('datasets', help="comma separated list of datasets to train and evaluate")
    parser.add_argument('-max_it', help="(optional) number of maximal iterations for svm in solver. "
                                        "Defaults to 100000, pick -1 to disable.", type=int)

    args = parser.parse_args()
    main(args)
