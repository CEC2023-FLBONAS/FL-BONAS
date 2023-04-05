from typing import Callable
from keras import models
import numpy as np
import tensorflow as tf
import fed_learn
from naszilla.acquisition_functions import acq_fn


DEFAULT_NUM_INIT = 10
DEFAULT_K = 10
DEFAULT_TOTAL_QUERIES = 100
DEFAULT_LOSS = 'val_loss'


#
class Client:

    def __init__(self,
                 id: int,
                 search_space,
                 num_ensemble=5,
                 num_init=DEFAULT_NUM_INIT,
                 k=DEFAULT_K,
                 loss=DEFAULT_LOSS,
                 total_queries=DEFAULT_TOTAL_QUERIES,
                 acq_opt_type='mutation',
                 num_arches_to_mutate=1,
                 max_mutation_rate=1,
                 explore_type='its',
                 predictor='bananas',
                 predictor_encoding='trunc_path',
                 cutoff=0,
                 mutate_encoding='adj',
                 random_encoding='adj',
                 deterministic=True,
                 verbose=1):

        self.id = id
        self.ensemble = None
        self.num_ensemble = num_ensemble
        self.search_space = search_space

        self.query = num_init
        self.total_queries = total_queries
        self.k = k

        self.data = self.search_space.generate_random_dataset(num=num_init,
                                                              predictor_encoding=predictor_encoding,
                                                              random_encoding=random_encoding,
                                                              deterministic_loss=deterministic,
                                                              cutoff=cutoff)

        self.acq_opt_type = acq_opt_type
        self.predictor_encoding = predictor_encoding
        self.mutate_encoding = mutate_encoding
        self.num_arches_to_mutate = num_arches_to_mutate
        self.max_mutation_rate = max_mutation_rate
        self.loss = loss
        self.deterministic = deterministic
        self.cutoff = cutoff
        self.verbose = verbose
        self.explore_type = 'its'


    def _init_model(self, model_fn: Callable, model_weights):
        ensemble = model_fn()
        for i in range(len(ensemble)):
            NN = ensemble[i]
            NN_weights = model_weights[i]
            fed_learn.set_model_weights(NN, NN_weights)  # 赋值权重！
        self.ensemble = ensemble


    def receive_data(self, x, y):
        self.x_train = x
        self.y_train = y


    def receive_and_init_model(self, model_fn: Callable, model_weights):
        self._init_model(model_fn, model_weights)


    def edge_train(self, metann_params):
        if self.ensemble is None:
            raise ValueError("Ensemble is not created for client: {0}".format(self.id))

        # hist = self.model.fit(self.x_train, self.y_train)
        xtrain = np.array([d['encoding'] for d in self.data])
        ytrain = np.array([d['val_loss'] for d in self.data])


        candidates = self.search_space.get_candidates(self.data,
                                                      acq_opt_type=self.acq_opt_type,
                                                      predictor_encoding=self.predictor_encoding,
                                                      mutate_encoding=self.mutate_encoding,
                                                      num_arches_to_mutate=self.num_arches_to_mutate,
                                                      max_mutation_rate=self.max_mutation_rate,
                                                      loss=self.loss,
                                                      deterministic_loss=self.deterministic,
                                                      cutoff=self.cutoff)

        xcandidates = np.array([c['encoding'] for c in candidates])
        candidate_predictions = []


        train_error = 0
        for i in range(len(self.ensemble)):
            NN = self.ensemble[i]
            net_params = metann_params['ensemble_params'][i]
            NN.fit(xtrain, ytrain,
                   batch_size=net_params['batch_size'],
                   epochs=net_params['epochs'],
                   verbose=net_params['verbose'])

            train_pred = np.squeeze(NN.predict(xtrain))
            train_error = np.mean(abs(train_pred - ytrain))

            candidate_predictions.append(np.squeeze(NN.predict(xcandidates)))

        train_error /= len(self.ensemble)

        #
        candidate_indices = acq_fn(candidate_predictions, ytrain=ytrain, explore_type=self.explore_type)

        #
        for j in candidate_indices[:self.k]:
            arch_dict = self.search_space.query_arch(candidates[j]['spec'],
                                                     predictor_encoding=self.predictor_encoding,
                                                     deterministic=self.deterministic,
                                                     cutoff=self.cutoff)
            self.data.append(arch_dict)

        self.query += self.k

        if self.verbose == 2:
            print('query {}, Neural predictor train error: {}'.format(self.query, train_error))
        if self.verbose:
            top_5_loss = sorted([d['val_loss'] for d in self.data])[:min(5, len(self.data))]
            print('query {}, top 5 losses {}'.format(self.query, top_5_loss))

        return train_error

    #
    def reset_model(self):
        for NN in self.ensemble:
            fed_learn.get_rid_of_the_models(NN)  #
