import argparse
import time
import logging
import sys
import os
import pickle
import numpy as np
import copy

from naszilla.params import *
from naszilla.nas_benchmarks import Nasbench101, Nasbench201, Nasbench301
from naszilla.nas_algorithms import run_nas_algorithm, bananas, record_best_data


def run_experiments(args, save_dir):

    # set up arguments
    trials = args.trials
    queries = args.query
    out_file = args.output_filename
    save_specs = args.save_specs
    metann_params = meta_neuralnet_params(args.metann_params)
    ss = args.search_space
    dataset = args.dataset
    mf = args.mf
    algorithm_params = algo_params(args.algo_params, queries=queries)
    num_algos = len(algorithm_params)
    logging.info(algorithm_params)


    mp = copy.deepcopy(metann_params)

    start = time.time()
    if ss == 'nasbench_101':
        search_space = Nasbench101(mf=mf)
    elif ss == 'nasbench_201':
        search_space = Nasbench201(dataset=dataset)
    elif ss == 'nasbench_301':
        search_space = Nasbench301()
    else:
        print('Invalid search space')
        raise NotImplementedError()
    loadcost = time.time() - start
    print(f'--> Total cost: {loadcost:.2f} seconds.')


    for i in range(trials):
        results = []
        val_results = []
        walltimes = []
        run_data = []


        for j in range(num_algos):
            print('\n ====== * Running NAS algorithm: {} ======'.format(algorithm_params[j]))
            starttime = time.time()

            result, val_result, run_datum = run_nas_algorithm(algorithm_params[j], search_space, mp)

            result = np.round(result, 5)
            val_result = np.round(val_result, 5)


            for d in run_datum:
                if not save_specs:
                    d.pop('spec')
                for key in ['encoding', 'adj', 'path', 'dist_to_min']:
                    if key in d:
                        d.pop(key)


            walltimes.append(time.time()-starttime)
            results.append(result)
            val_results.append(val_result)
            run_data.append(run_datum)


        filename = os.path.join(save_dir, '{}_{}.pkl'.format(out_file, i))
        print('\n* Trial summary: (params, results, walltimes)')
        print(algorithm_params)
        print(ss)
        print(results)
        print(walltimes)
        print('\n* Saving to file {}'.format(filename))
        with open(filename, 'wb') as f:
            pickle.dump([algorithm_params, metann_params, results, walltimes, run_data, val_results], f)
            f.close()


def main(args):

    print('===================')
    print('query=', args.query)
    print('k=', args.k)
    print('No.', args.trial)
    print('===================')

    # make save directory
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    algo_params_t = args.algo_params
    save_path = save_dir + '/' + algo_params_t + '/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # set up arguments
    trials = args.trial
    queries = args.query
    out_file = args.output_filename
    save_specs = args.save_specs
    metann_params = meta_neuralnet_params(args.metann_params)
    ss = args.search_space
    dataset = args.dataset
    mf = args.mf
    algorithm_params = algo_params(args.algo_params, queries=queries)

    # set up search space
    mp = copy.deepcopy(metann_params)

    start = time.time()
    if ss == 'nasbench_101':
        search_space = Nasbench101(mf=mf)
    elif ss == 'nasbench_201':
        search_space = Nasbench201(dataset=dataset)
    elif ss == 'nasbench_301':
        search_space = Nasbench301()
    else:
        print('Invalid search space')
        raise NotImplementedError()
    loadcost = time.time() - start
    print(f'--> Total cost: {loadcost:.2f} seconds.')

    start_time = time.time()

    data = bananas(search_space=search_space,
                   metann_params=mp,
                   num_init=10,
                   k=args.k,
                   total_queries=args.query)
    res = record_best_data(data, args.k, args.query)

    filename = 'bananas100_k{}_d{}_N{}.pkl'.format(args.k,
                                                args.query,
                                                args.trial)
    with open(filename, 'wb') as f:
        pickle.dump(res, f)
        f.close()

    print('File saved: {}'.format(filename))

    total_time = (time.time() - start_time) / 60
    print('Total time cost: {:.1f}min.'.format(total_time))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args for BANANAS experiments')
    parser.add_argument('--trial', type=int, default=0, help='Number of trials')
    parser.add_argument('--k', type=int, default=5, help='Number of k')

    parser.add_argument('--query', type=int, default=500, help='Max number of queries/evaluations each NAS algorithm gets')
    parser.add_argument('--search_space', type=str, default='nasbench_201', help='nasbench_101, _201, or _301')
    parser.add_argument('--dataset', type=str, default='cifar100', help='cifar10, 100, or imagenet (for nasbench201)')
    parser.add_argument('--mf', type=bool, default=False, help='Multi fidelity: true or false (for nasbench101)')
    parser.add_argument('--metann_params', type=str, default='standard', help='which parameters to use')
    parser.add_argument('--algo_params', type=str, default='bananas', help='which parameters to use')
    parser.add_argument('--output_filename', type=str, default='round', help='name of output files')
    parser.add_argument('--save_dir', type=str, default='results_output', help='name of save directory')
    parser.add_argument('--save_specs', type=bool, default=False, help='save the architecture specs')    

    args = parser.parse_args()

    main(args)

    args.trial = args.trial + 10
    main(args)
