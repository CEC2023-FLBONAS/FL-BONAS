import json
from pathlib import Path

import numpy as np
from tensorflow import keras
from keras import datasets
from naszilla.params import *
import fed_learn
import copy, pickle, time
from naszilla.nas_benchmarks import Nasbench101, Nasbench201, Nasbench301


args = fed_learn.get_args()

print('===================')
print('k=', args.num_k)
print('R=', args.global_epochs)
print('No.', args.trial)
print('===================')

fed_learn.set_working_GPU(str(args.gpu))

experiment_folder_path = Path(__file__).resolve().parent / "experiments" / args.name
experiment = fed_learn.Experiment(experiment_folder_path, args.overwrite_experiment)
experiment.serialize_args(args)


metann_params = meta_neuralnet_params(param_str=args.metann_params,
                                      num_ensemble=args.num_ensemble)

ss = args.search_space
if ss == 'nasbench_101':
    search_space = Nasbench101(mf=args.mf)
elif ss == 'nasbench_201':
    search_space = Nasbench201(dataset=args.dataset)
elif ss == 'nasbench_301':
    search_space = Nasbench301()
else:
    print('Invalid search space')
    raise NotImplementedError()





def model_fn():
    mp = copy.deepcopy(metann_params)
    v = mp['ensemble_params']
    ensemble = []
    for i in range(args.num_ensemble):
        NN = fed_learn.create_model(loss_fn=v[i]['loss'],
                                    num_layers=v[i]['num_layers'],
                                    layer_width=v[i]['layer_width'],
                                    lr=v[i]['lr'],
                                    regularization=0)
        ensemble.append(NN)
    return ensemble


ensemble = model_fn()
print('ensemble:', ensemble)

weight_summarizer = fed_learn.FedAvg()


server = fed_learn.Server(search_space=search_space,
                          model_fn=model_fn,
                          weight_summarizer=weight_summarizer,
                          nb_clients=args.clients,
                          client_fraction=args.fraction,
                          num_ensemble=args.num_ensemble,
                          num_init=args.num_init,
                          k=args.num_k)

weight_path = args.weights_file
if weight_path is not None:
    server.load_model_weights(weight_path)

server.create_clients()

start_time = time.time()


for epoch in range(args.global_epochs):
    print("Global Epoch {0} is starting".format(epoch))
    server.init_for_new_epoch()
    selected_clients = server.select_clients()

    fed_learn.print_selected_clients(selected_clients)

    for client in selected_clients:
        print("Client {0} is starting the training".format(client.id))

        server.send_model(client)
        mp = copy.deepcopy(metann_params)
        train_error = client.edge_train(mp)
        server.epoch_train_error.append(train_error)

        server.receive_results(client)

    server.summarize_weights()

    epoch_mean_train_error = np.mean(server.epoch_train_error)
    server.global_train_errors.append(epoch_mean_train_error)
    print("Loss (client mean): {0}".format(server.global_train_errors[-1]))

    print("--- Global record ---")
    server.record_global_data(epoch)
    print('FL round {}: num_data {} num_time {} best_val {:.3f} best_test {:.3f}'.format(epoch,
                                                                                      server.global_data_metrics_dict['num_data'][-1],
                                                                                      server.global_data_metrics_dict['num_time'][-1],
                                                                                      server.global_data_metrics_dict['best_val_loss'][-1],
                                                                                      server.global_data_metrics_dict['best_test_loss'][-1]))



num_data = [len(client.data) for client in server.clients]
print('number of evaluated Arch in each client (Total number:{})'.format(sum(num_data)))
print(num_data)

print('===== global data record ====')
print('num_data :', server.global_data_metrics_dict['num_data'][-1])
print('num_time :', server.global_data_metrics_dict['num_time'][-1])
print('best_val :', server.global_data_metrics_dict['best_val_loss'][-1])
print('best_test:', server.global_data_metrics_dict['best_test_loss'][-1])
print('=============================')


filename = 'exp100_k{}_d{}_t{}_N{}.pkl'.format(server.k,
                                            server.global_data_metrics_dict['num_data'][-1],
                                            server.global_data_metrics_dict['num_time'][-1],
                                            args.trial)  # k data time No.
with open(filename, 'wb') as f:
    pickle.dump(server.global_data_metrics_dict, f)
    f.close()

print('File saved: {}'.format(filename))

total_time = (time.time() - start_time) / 60
print('Total time cost: {:.1f}min.'.format(total_time))
