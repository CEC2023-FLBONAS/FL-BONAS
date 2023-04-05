from typing import Callable

import numpy as np
from keras import models

import fed_learn
from fed_learn.weight_summarizer import WeightSummarizer


#.
class Server:
    def __init__(self,
                 search_space,
                 model_fn: Callable,
                 weight_summarizer: WeightSummarizer,
                 nb_clients: int = 10,
                 client_fraction: float = 0.2,
                 num_ensemble: int = 5,
                 num_init: int = 10,
                 k: int = 10):
        self.search_space = search_space
        self.nb_clients = nb_clients
        self.client_fraction = client_fraction
        self.weight_summarizer = weight_summarizer
        self.num_ensemble = num_ensemble
        self.num_init = num_init
        self.k = k


        self.model_fn = model_fn
        ensemble = self.model_fn()
        self.global_test_metrics_dict = {'loss': [], 'accuracy': []}
        self.global_data_metrics_dict = {'num_data':[],
                                         'num_time':[],
                                         'best_val_loss':[],
                                         'best_test_loss':[]}


        self.global_model_weights = []
        for i in range(self.num_ensemble):
            self.global_model_weights.append(ensemble[i].get_weights())

        for NN in ensemble:
            fed_learn.get_rid_of_the_models(NN)

        self.global_train_errors = []
        self.epoch_train_error = []

        self.clients = []
        self.client_model_weights = [[] for _ in range(self.num_ensemble)]  # List中每个元素是个List，保存ensemble中每个NN的信息


        self.client_train_params_dict = {"batch_size": 32,
                                         "epochs": 5,
                                         "verbose": 1,
                                         "shuffle": True}


    def _create_model_with_updated_weights(self) -> models.Model:
        model = self.model_fn()
        fed_learn.models.set_model_weights(model, self.global_model_weights)
        return model

    def send_model(self, client):
        client.receive_and_init_model(self.model_fn, self.global_model_weights)  # 当前Client重新构建一个模型并加载global权重

    def init_for_new_epoch(self):

        self.client_model_weights.clear()
        self.client_model_weights = [[] for _ in range(self.num_ensemble)]

        self.epoch_train_error.clear()


    def receive_results(self, client):
        for i in range(self.num_ensemble):
            client_weights = client.ensemble[i].get_weights()
            self.client_model_weights[i].append(client_weights)
        client.reset_model()


    def create_clients(self):
        # Create all the clients
        for i in range(self.nb_clients):
            client = fed_learn.Client(id=i,
                                      search_space=self.search_space,
                                      num_ensemble=self.num_ensemble,
                                      num_init=self.num_init,
                                      k=self.k)
            self.clients.append(client)


    def summarize_weights(self):
        for i in range(self.num_ensemble):
            new_weights = self.weight_summarizer.process(self.client_model_weights[i])
            self.global_model_weights[i] = new_weights


    def get_client_train_param_dict(self):
        return self.client_train_params_dict


    def update_client_train_params(self, param_dict: dict):
        self.client_train_params_dict.update(param_dict)


    def test_global_model(self, x_test: np.ndarray, y_test: np.ndarray):
        model = self._create_model_with_updated_weights()
        results = model.evaluate(x_test, y_test, batch_size=32, verbose=1)

        results_dict = dict(zip(model.metrics_names, results))
        for metric_name, value in results_dict.items():
            if 'loss' not in self.global_test_metrics_dict:
                self.global_test_metrics_dict['loss'] = []
            if 'accuracy' not in self.global_test_metrics_dict:
                self.global_test_metrics_dict['accuracy'] = []

            self.global_test_metrics_dict[metric_name].append(value)

        fed_learn.get_rid_of_the_models(model)

        return results_dict


    def record_global_data(self, epoch):
        num_data = 0
        best_val_loss = 1e6
        best_test_loss = 1e6

        for client in self.clients:
            num_data += len(client.data)
            best_arch = sorted(client.data, key=lambda i: i['val_loss'])[0]
            val_loss, test_loss = best_arch['val_loss'], best_arch['test_loss']
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            if test_loss < best_test_loss:
                best_test_loss = test_loss

        self.global_data_metrics_dict['num_data'].append(num_data)
        self.global_data_metrics_dict['num_time'].append(self.num_init + self.k * epoch)
        self.global_data_metrics_dict['best_val_loss'].append(best_val_loss)
        self.global_data_metrics_dict['best_test_loss'].append(best_test_loss)


    def select_clients(self):
        nb_clients_to_use = max(int(self.nb_clients * self.client_fraction), 1)
        client_indices = np.arange(self.nb_clients)
        np.random.shuffle(client_indices)
        selected_client_indices = client_indices[:nb_clients_to_use]
        return np.asarray(self.clients)[selected_client_indices]


    def save_model_weights(self, path: str):
        model = self._create_model_with_updated_weights()
        model.save_weights(str(path), overwrite=True)
        fed_learn.get_rid_of_the_models(model)


    def load_model_weights(self, path: str, by_name: bool = False):
        model = self._create_model_with_updated_weights()
        model.load_weights(str(path), by_name=by_name)
        self.global_model_weights = model.get_weights()
        fed_learn.get_rid_of_the_models(model)
