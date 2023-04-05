import argparse
import json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", help="Name of the experiment", type=str, default='tempSQ', required=False)
    parser.add_argument("-oe", "--overwrite-experiment", help="Overwrite existing experiment", type=bool, default=True, required=False)
    parser.add_argument("-s", "--data-sampling-technique", help="Data sampling technique (IID or Non-IID)", type=str, default="iid", required=False)
    parser.add_argument("-w", "--weights-file", help="Weights file path to load", type=str, required=False)
    parser.add_argument("-d", "--debug", help="Debugging", action="store_true", required=False)
    parser.add_argument("-lr", "--learning-rate", help="Learning rate", type=float, default=0.01, required=False)
    parser.add_argument("-b", "--batch-size", help="Batch Size", type=int, default=32, required=False)
    parser.add_argument("-ce", "--client-epochs", help="Number of epochs for the clients", type=int, default=1, required=False)
    parser.add_argument("-g", "--gpu", help="GPU to use (-1 is CPU)", type=int, default=0, required=False)

    parser.add_argument('--search_space', type=str, default='nasbench_201', help='nasbench_101, _201, or _301')
    parser.add_argument('--mf', type=bool, default=False, help='Multi fidelity: true or false (for nasbench101)')
    parser.add_argument('--dataset', type=str, default='cifar100', help='cifar10, 100, or imagenet (for nasbench201)')

    parser.add_argument('--metann_params', type=str, default='standard', help='which parameters to use')
    parser.add_argument("-ne", "--num_ensemble", help="number of NN in an ensemble", type=int, default=5, required=False)
    parser.add_argument("-ni", "--num_init", help="number of initial data in each client", type=int, default=10, required=False)
    parser.add_argument("-c", "--clients", help="Number of clients", type=int, default=10, required=False)
    parser.add_argument("-f", "--fraction", help="Client fraction to use", type=float, default=0.2, required=False)
    parser.add_argument("-k", "--num_k", help="number of real-evaluated Arch in each BO round", type=int, default=5, required=False)
    parser.add_argument("-e", "--global_epochs", help="Number of global (server) epochs", type=int, default=99, required=False)
    parser.add_argument("-no", "--trial", help="No.# of independent experiment", type=int, default=0, required=False)

    args = parser.parse_args()
    return args


def args_as_json(args):
    json_str = json.dumps(args.__dict__, sort_keys=True, indent=4)
    return json_str


def save_args_as_json(args, path):
    json_str = args_as_json(args)

    with open(str(path), "w") as f:
        f.write(json_str)
