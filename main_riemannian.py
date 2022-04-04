#!/usr/bin/env python3

'''
Model for Riemannian feature calculation and classification for EEG data
'''

import pandas as pd

import numpy as np
import time
import sys


from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import KFold, train_test_split
from sklearn.calibration import CalibratedClassifierCV

# import self defined functions
from riemannian_multiscale import riemannian_multiscale
from filters import load_filterbank
from get_data import get_data

import Fancy_aggregations as fz
__author__ = "Michael Hersche and Tino Rellstab"
__email__ = "herschmi@ethz.ch,tinor@ethz.ch"

class dummy_plug:
        def _init(self, x=None):
            self.x = x

def fast_montecarlo_optimization(function_alpha, x0=[0.5], minimizer_kwargs=None, niter=200, smart=True):
    '''
    Just randomly samples the function. More functionality might come in the future if necessary.
    '''
    

    iter_actual = 0

    #epsilon = 0.05
    eval_per_iter = 5
    best_fitness = 1
    resultado = dummy_plug()
    if hasattr(x0, '__len__'):
        size_2 = len(x0)
    else:
        size_2 = 1
    while(iter_actual < niter):
        subjects = np.random.random_sample((eval_per_iter, size_2))
        fitness = [function_alpha(x) for x in subjects]
        ordered = np.sort(fitness)
        arg_ordered = np.argsort(fitness)
        iter_actual += 1

        if ordered[1] < best_fitness:
            best_fitness = ordered[1]
            resultado.x = subjects[arg_ordered[1], :]
            resultado.fun = best_fitness
            if best_fitness == 0.0:
                return resultado

    return resultado


def alpha_learn(X, y, cost):
    def compute_accuracy(yhat, y):
        return np.mean(np.equal(yhat, y))

    def optimize_function(X, y, cost_convex, alpha):
        def alpha_cost(real, yhat, axis): return cost_convex(
            real, yhat, axis, alpha)

        agg_logits = fz.penalties.penalty_aggregation(
            X, [np.mean, np.median, np.min, np.max], axis=0, keepdims=False, cost=alpha_cost)
        yhat = np.argmax(agg_logits, axis=1)

        return 1 - compute_accuracy(yhat, y)

    def function_alpha(a): return optimize_function(X, y, cost, a)
    x0 = [0.5]

    res = fast_montecarlo_optimization(
        function_alpha, x0=x0, niter=100)
    alpha_value = res

    if isinstance(alpha_value, dummy_plug):
        alpha_value = alpha_value.x

    if hasattr(alpha_value, 'len'):
        alpha_value = alpha_value[0]

    # , alpha_value
    return lambda real, yhatf, axis: cost(real, yhatf, axis, alpha_value)


class Riemannian_Model:

    def __init__(self, cost, agrupate):
        self.agrupate = agrupate
        self.crossvalidation = True
        self.data_path = '/home/fcojavier.fernandez/Github/Graz/'
        self.svm_kernel = 'linear'  # 'sigmoid'#'linear' # 'sigmoid', 'rbf',
        self.svm_c = 0.1  # for linear 0.1 (inverse),
        self.NO_splits = 20  # number of folds in cross validation
        self.fs = 250.  # sampling frequency
        self.NO_channels = 15  # number of EEG channels
        self.NO_subjects = 9
        # Total number of CSP feature per band and timewindow
        self.NO_riem = int(self.NO_channels*(self.NO_channels+1)/2)
        self.bw = np.array([2, 4, 8, 16, 32])  # bandwidth of filtered signals
        self.ftype = 'butter'  # 'fir', 'butter'
        self.forder = 2  # 4
        self.filter_bank = load_filterbank(
            self.bw, self.fs, order=self.forder, max_freq=40, ftype=self.ftype)  # get filterbank coeffs
        time_windows_flt = np.array([[2.5, 4.5],
                                     [4, 6],
                                     [2.5, 6],
                                     [2.5, 3.5],
                                     [3, 4],
                                     [4, 5]])*self.fs
        self.time_windows = time_windows_flt.astype(int)
        # restrict time windows and frequency bands
        # use only largest timewindow
        self.time_windows = self.time_windows[2:3]
        # self.f_bands_nom = self.f_bands_nom[18:27] # use only 4Hz-32Hz bands
        self.rho = 0.1
        self.NO_bands = self.filter_bank.shape[0]
        self.NO_time_windows = self.time_windows.shape[0]
        self.NO_features = self.NO_riem*self.NO_bands*self.NO_time_windows
        # {"Riemann","Riemann_Euclid","Whitened_Euclid","No_Adaptation"}
        self.riem_opt = "Riemann"
        # time measurements
        self.train_time = 0
        self.train_trials = 0
        self.eval_time = 0
        self.eval_trials = 0
        self.vectorized = False
        self.cost = cost

        if self.cost is None:
            self.agg = lambda x, axis=0, keepdims=False: fz.penalties.penalty_aggregation(x, axis=axis, keepdims=keepdims,
                                                                                      agg_functions=[np.mean, np.max, np.median, np.min], cost=fz.penalties.cost_functions[self.cost])
        else:
            self.agg = np.mean

    def run_riemannian(self):

        ################################ Training ############################################################################
        start_train = time.time()

        # 1. calculate features and mean covariance for training
        riemann = riemannian_multiscale(self.filter_bank, self.time_windows,
                                        riem_opt=self.riem_opt, rho=self.rho, vectorized=self.vectorized)
        train_feat = riemann.fit(self.train_data)

        if self.vectorized:
            # 2. Train SVM Model
            if self.svm_kernel == 'linear':
                clf = LinearSVC(C=self.svm_c, intercept_scaling=1, loss='hinge', max_iter=1000,
                                multi_class='ovr', penalty='l2', random_state=1, tol=0.00001)
            else:
                clf = SVC(C=self.svm_c, kernel=self.svm_kernel, degree=10, gamma='auto', coef0=0.0,
                          tol=0.001, cache_size=10000, max_iter=-1, decision_function_shape='ovr')

            clf.fit(train_feat, self.train_label)

            end_train = time.time()
            self.train_time += end_train-start_train
            self.train_trials += len(self.train_label)
        else:
            clfs = []
            acumulate_groups_size = self.agrupate
            total_feats = train_feat.shape[2]
            size_group = int(total_feats / acumulate_groups_size)

            for freq in np.arange(acumulate_groups_size):
                # 2. Train SVM Model
                if self.svm_kernel == 'linear':
                    clf = CalibratedClassifierCV(LinearSVC(C=self.svm_c, intercept_scaling=1, loss='hinge',
                                                 max_iter=1000, multi_class='ovr', penalty='l2', random_state=1, tol=0.00001))
                else:
                    clf = SVC(C=self.svm_c, kernel=self.svm_kernel, degree=10, gamma='auto', coef0=0.0,
                              tol=0.001, cache_size=10000, max_iter=-1, decision_function_shape='ovr', probability=True)

                if freq != acumulate_groups_size-1:
                    train_feat_freq = train_feat[:, 0,
                                                 freq*size_group:(freq+1)*size_group, :]
                else:
                    train_feat_freq = train_feat[:, 0, freq*size_group:, :]

                clf.fit(train_feat_freq.reshape(train_feat_freq.shape[0], -1), self.train_label-1)

                if len(clfs) == 0:
                        logits_train = clf.predict_proba(train_feat_freq.reshape
                                                         (train_feat_freq.shape[0], -1))
                        logits_train = logits_train.reshape((1, *logits_train.shape))
                else:
                        aux = clf.predict_proba(train_feat_freq.reshape
                                                (train_feat_freq.shape[0], -1))
                        aux = aux.reshape((1, *aux.shape))
                        logits_train = np.concatenate((logits_train, aux))

                clfs.append(clf)

            if self.cost is not None:
                new_cost = alpha_learn(logits_train, self.train_label-1, fz.penalties.cost_functions[self.cost])

                self.agg = lambda x, axis=0, keepdims=False: fz.penalties.penalty_aggregation(x, axis=axis, keepdims=keepdims,
                                                                                      agg_functions=[np.mean, np.max, np.median, np.min], 
																					  cost=new_cost)


            end_train = time.time()
            self.train_time += end_train-start_train
            self.train_trials += len(self.train_label)
        ################################# Evaluation ###################################################
        start_eval = time.time()
        eval_feat = riemann.features(self.eval_data)

        if self.vectorized:
            success_rate = clf.score(eval_feat, self.eval_label)
        else:
            acumulate_groups_size = self.agrupate
            total_feats = eval_feat.shape[2]
            size_group = int(total_feats / acumulate_groups_size)

            full_logits = np.zeros(
                (acumulate_groups_size, eval_feat.shape[0], len(np.unique(self.train_label))))
            for freq in np.arange(acumulate_groups_size):
                if freq != acumulate_groups_size-1:
                    eval_feat_freq = eval_feat[:, 0, freq *
                                               size_group:(freq+1)*size_group, :]
                else:
                    eval_feat_freq = eval_feat[:, 0, freq*size_group:, :]

                freq_logits = clfs[freq].predict_proba(
                    eval_feat_freq.reshape(eval_feat_freq.shape[0], -1))
                full_logits[freq, :, :] = freq_logits

            success_rate = np.mean(np.equal(np.argmax(
                self.agg(full_logits, axis=0), axis=1), self.eval_label-1))

        # print(success_rate)
        end_eval = time.time()

        #print("Time for one Evaluation " + str((end_eval-start_eval)/len(self.eval_label)) )

        self.eval_time += end_eval-start_eval
        self.eval_trials += len(self.eval_label)

        return success_rate

    def load_data(self):
        if self.crossvalidation:
            data, label = get_data(self.subject, True, self.data_path)

            #kf = KFold(n_splits=self.NO_splits)
            #data = np.swapaxes(data, 0, 2)
            X_train, X_test, y_train, y_test = train_test_split(
                data, label, test_size=0.50, random_state=self.split)
            #X_train = np.swapaxes(X_train, 0, 2)
            #X_test = np.swapaxes(X_test, 0, 2)

            self.train_data = X_train
            self.train_label = y_train
            self.eval_data = X_test
            self.eval_label = y_test

        else:
            self.train_data, self.train_label = get_data(
                self.subject, True, self.data_path)
            self.eval_data, self.eval_label = get_data(
                self.subject, False, self.data_path)


def main(output, cost, groups):
    model = Riemannian_Model(cost=cost, agrupate=groups)

    print("Number of used features: " + str(model.NO_features))

    print(model.riem_opt)

    # success rate sum over all subjects
    success_tot_sum = 0

    if model.crossvalidation:
        print("Cross validation run")
    else:
        print("Test data set")

    start = time.time()
    accuracies = []
    # Go through all subjects
    for model.subject in range(1, model.NO_subjects+1):

        print("Subject" + str(model.subject)+":")

        if model.crossvalidation:
            success_sub_sum = 0

            for model.split in range(model.NO_splits):
                model.load_data()
                s_acc = model.run_riemannian()
                print(s_acc)
                accuracies.append(s_acc)
                success_sub_sum += s_acc

            # average over all splits
            success_rate = success_sub_sum/model.NO_splits

        else:
            # load Eval data
            model.load_data()
            success_rate = model.run_riemannian()

        # print(success_rate)
        success_tot_sum += success_rate

    # Average success rate over all subjects
    print("Average success rate: " + str(success_tot_sum/model.NO_subjects))

    print("Training average time: " + str(model.train_time/model.NO_subjects))
    print("Evaluation average time: " + str(model.eval_time/model.NO_subjects))

    end = time.time()

    print("Time elapsed [s] " + str(end - start))
    with open(output, 'w') as f:
        f.write(str(success_tot_sum/model.NO_subjects))

	pd.DataFrame(np.array(accuracies)).to_csv('csp_accuracies_' + cost + '_' + groups + '.csv')


if __name__ == '__main__':
    import sys
    output_file = sys.argv[1]
    cost = int(sys.argv[2])
    group = int(sys.argv[3])

    main(output_file, cost, group)

