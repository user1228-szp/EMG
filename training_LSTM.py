#!/usr/bin/env python
"""
LSTM Example en Python (adaptado de Google Colab a entorno local)

Uso:
    python lstm_example.py --example [toy|csv|valid]
"""

from packaging import version
import tensorflow as tf
assert version.parse(tf.__version__) >= version.parse("2.8.0")

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np
import math
import argparse
from sklearn.model_selection import train_test_split

# Funciones de activación y sus derivadas
def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def sigmoid_derivative(values):
    return values * (1 - values)

def tanh_derivative(values):
    return 1. - values ** 2

# Generador de arreglos aleatorios en un rango dado
def rand_arr(a, b, *args):
    np.random.seed(0)
    return np.random.rand(*args) * (b - a) + a

# Clase que almacena los parámetros de la LSTM
class LstmParam:
    def __init__(self, mem_cell_ct, x_dim):
        self.mem_cell_ct = mem_cell_ct
        self.x_dim = x_dim
        concat_len = x_dim + mem_cell_ct
        self.wg = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        self.wi = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        self.wf = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        self.wo = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        self.bg = rand_arr(-0.1, 0.1, mem_cell_ct)
        self.bi = rand_arr(-0.1, 0.1, mem_cell_ct)
        self.bf = rand_arr(-0.1, 0.1, mem_cell_ct)
        self.bo = rand_arr(-0.1, 0.1, mem_cell_ct)
        self.wg_diff = np.zeros((mem_cell_ct, concat_len))
        self.wi_diff = np.zeros((mem_cell_ct, concat_len))
        self.wf_diff = np.zeros((mem_cell_ct, concat_len))
        self.wo_diff = np.zeros((mem_cell_ct, concat_len))
        self.bg_diff = np.zeros(mem_cell_ct)
        self.bi_diff = np.zeros(mem_cell_ct)
        self.bf_diff = np.zeros(mem_cell_ct)
        self.bo_diff = np.zeros(mem_cell_ct)

    def apply_diff(self, lr=1):
        self.wg -= lr * self.wg_diff
        self.wi -= lr * self.wi_diff
        self.wf -= lr * self.wf_diff
        self.wo -= lr * self.wo_diff
        self.bg -= lr * self.bg_diff
        self.bi -= lr * self.bi_diff
        self.bf -= lr * self.bf_diff
        self.bo -= lr * self.bo_diff
        self.wg_diff = np.zeros_like(self.wg)
        self.wi_diff = np.zeros_like(self.wi)
        self.wf_diff = np.zeros_like(self.wf)
        self.wo_diff = np.zeros_like(self.wo)
        self.bg_diff = np.zeros_like(self.bg)
        self.bi_diff = np.zeros_like(self.bi)
        self.bf_diff = np.zeros_like(self.bf)
        self.bo_diff = np.zeros_like(self.bo)

# Clase que almacena el estado de la LSTM
class LstmState:
    def __init__(self, mem_cell_ct, x_dim):
        self.g = np.zeros(mem_cell_ct)
        self.i = np.zeros(mem_cell_ct)
        self.f = np.zeros(mem_cell_ct)
        self.o = np.zeros(mem_cell_ct)
        self.s = np.zeros(mem_cell_ct)
        self.h = np.zeros(mem_cell_ct)
        self.bottom_diff_h = np.zeros_like(self.h)
        self.bottom_diff_s = np.zeros_like(self.s)

# Clase para un nodo LSTM (un paso de tiempo)
class LstmNode:
    def __init__(self, lstm_param, lstm_state):
        self.state = lstm_state
        self.param = lstm_param
        self.xc = None

    def bottom_data_is(self, x, s_prev=None, h_prev=None):
        if s_prev is None: s_prev = np.zeros_like(self.state.s)
        if h_prev is None: h_prev = np.zeros_like(self.state.h)
        self.s_prev = s_prev
        self.h_prev = h_prev
        xc = np.hstack((x, h_prev))
        self.state.g = np.tanh(np.dot(self.param.wg, xc) + self.param.bg)
        self.state.i = sigmoid(np.dot(self.param.wi, xc) + self.param.bi)
        self.state.f = sigmoid(np.dot(self.param.wf, xc) + self.param.bf)
        self.state.o = sigmoid(np.dot(self.param.wo, xc) + self.param.bo)
        self.state.s = self.state.g * self.state.i + s_prev * self.state.f
        self.state.h = self.state.s * self.state.o
        self.xc = xc

    def top_diff_is(self, top_diff_h, top_diff_s):
        ds = self.state.o * top_diff_h + top_diff_s
        do = self.state.s * top_diff_h
        di = self.state.g * ds
        dg = self.state.i * ds
        df = self.s_prev * ds
        di_input = sigmoid_derivative(self.state.i) * di
        df_input = sigmoid_derivative(self.state.f) * df
        do_input = sigmoid_derivative(self.state.o) * do
        dg_input = tanh_derivative(self.state.g) * dg
        self.param.wi_diff += np.outer(di_input, self.xc)
        self.param.wf_diff += np.outer(df_input, self.xc)
        self.param.wo_diff += np.outer(do_input, self.xc)
        self.param.wg_diff += np.outer(dg_input, self.xc)
        self.param.bi_diff += di_input
        self.param.bf_diff += df_input
        self.param.bo_diff += do_input
        self.param.bg_diff += dg_input
        dxc = (np.dot(self.param.wi.T, di_input) +
               np.dot(self.param.wf.T, df_input) +
               np.dot(self.param.wo.T, do_input) +
               np.dot(self.param.wg.T, dg_input))
        self.state.bottom_diff_s = ds * self.state.f
        self.state.bottom_diff_h = dxc[self.param.x_dim:]

# Red LSTM que agrupa múltiples nodos
class LstmNetwork:
    def __init__(self, lstm_param):
        self.lstm_param = lstm_param
        self.lstm_node_list = []
        self.x_list = []

    def y_list_is(self, y_list, loss_layer):
        assert len(y_list) == len(self.x_list)
        idx = len(self.x_list) - 1
        loss = loss_layer.loss(self.lstm_node_list[idx].state.h, y_list[idx])
        diff_h = loss_layer.bottom_diff(self.lstm_node_list[idx].state.h, y_list[idx])
        diff_s = np.zeros(self.lstm_param.mem_cell_ct)
        self.lstm_node_list[idx].top_diff_is(diff_h, diff_s)
        idx -= 1
        while idx >= 0:
            loss += loss_layer.loss(self.lstm_node_list[idx].state.h, y_list[idx])
            diff_h = loss_layer.bottom_diff(self.lstm_node_list[idx].state.h, y_list[idx])
            diff_h += self.lstm_node_list[idx+1].state.bottom_diff_h
            diff_s = self.lstm_node_list[idx+1].state.bottom_diff_s
            self.lstm_node_list[idx].top_diff_is(diff_h, diff_s)
            idx -= 1
        return loss

    def x_list_clear(self):
        self.x_list = []

    def x_list_add(self, x):
        self.x_list.append(x)
        if len(self.x_list) > len(self.lstm_node_list):
            lstm_state = LstmState(self.lstm_param.mem_cell_ct, self.lstm_param.x_dim)
            self.lstm_node_list.append(LstmNode(self.lstm_param, lstm_state))
        idx = len(self.x_list) - 1
        if idx == 0:
            self.lstm_node_list[idx].bottom_data_is(x)
        else:
            s_prev = self.lstm_node_list[idx-1].state.s
            h_prev = self.lstm_node_list[idx-1].state.h
            self.lstm_node_list[idx].bottom_data_is(x, s_prev, h_prev)

# Capa de pérdida simple para el entrenamiento
class ToyLossLayer:
    @classmethod
    def loss(cls, pred, label):
        return (pred[0] - label) ** 2

    @classmethod
    def bottom_diff(cls, pred, label):
        diff = np.zeros_like(pred)
        diff[0] = 2 * (pred[0] - label)
        return diff

# Ejemplo 1: Entrenamiento con datos sintéticos
def example_toy():
    np.random.seed(0)
    mem_cell_ct = 100
    x_dim = 50
    lstm_param = LstmParam(mem_cell_ct, x_dim)
    lstm_net = LstmNetwork(lstm_param)
    y_list = [-0.5, 0.2, 0.1, -0.5]
    input_val_arr = [np.random.random(x_dim) for _ in y_list]
    for cur_iter in range(100):
        print("iter", f"{cur_iter:2d}", end=": ")
        for ind in range(len(y_list)):
            lstm_net.x_list_add(input_val_arr[ind])
        y_pred_str = ", ".join([f"{lstm_net.lstm_node_list[ind].state.h[0]: 2.5f}" for ind in range(len(y_list))])
        print("y_pred = [" + y_pred_str + "]", end=", ")
        loss = lstm_net.y_list_is(y_list, ToyLossLayer)
        print("loss:", f"{loss:.3e}")
        lstm_param.apply_diff(lr=0.1)
        lstm_net.x_list_clear()

# Ejemplo 2: Entrenamiento cargando datos desde un CSV
def example_with_csv():
    df = pd.read_csv('data.csv')  # Asegúrate de tener 'data.csv'
    x_dim = len(df.columns) - 1
    y_list = df.iloc[:, -1].values
    x_data = df.iloc[:, :-1].values
    mem_cell_ct = 100
    lstm_param = LstmParam(mem_cell_ct, x_dim)
    lstm_net = LstmNetwork(lstm_param)
    for cur_iter in range(100):
        print("iter", f"{cur_iter:2d}", end=": ")
        for ind in range(len(y_list)):
            lstm_net.x_list_add(x_data[ind])
        y_pred_str = ", ".join([f"{lstm_net.lstm_node_list[ind].state.h[0]: 2.5f}" for ind in range(len(y_list))])
        print("y_pred = [" + y_pred_str + "]", end=", ")
        loss = lstm_net.y_list_is(y_list, ToyLossLayer)
        print("loss:", f"{loss:.3e}")
        lstm_param.apply_diff(lr=0.1)
        lstm_net.x_list_clear()

# Ejemplo 3: Entrenamiento con división en entrenamiento y validación
def example_with_validation():
    df = pd.read_csv('data.csv')  # Asegúrate de tener 'data.csv'
    x_data = df.iloc[:, :-1].values
    y_data = df.iloc[:, -1].values
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    mem_cell_ct = 100
    x_dim = x_train.shape[1]
    lstm_param = LstmParam(mem_cell_ct, x_dim)
    lstm_net = LstmNetwork(lstm_param)
    for cur_iter in range(100):
        print("iter", f"{cur_iter:2d}", end=": ")
        for ind in range(len(y_train)):
            lstm_net.x_list_add(x_train[ind])
        train_loss = lstm_net.y_list_is(y_train, ToyLossLayer)
        print("Train loss:", f"{train_loss:.3e}", end=", ")
        lstm_param.apply_diff(lr=0.1)
        lstm_net.x_list_clear()
        for ind in range(len(y_val)):
            lstm_net.x_list_add(x_val[ind])
        val_loss = lstm_net.y_list_is(y_val, ToyLossLayer)
        print("Validation loss:", f"{val_loss:.3e}")
        lstm_net.x_list_clear()

# Función principal para seleccionar el ejemplo a ejecutar
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", choices=["toy", "csv", "valid"], default="toy", help="Ejemplo a ejecutar")
    args = parser.parse_args()
    if args.example == "toy":
        example_toy()
    elif args.example == "csv":
        example_with_csv()
    elif args.example == "valid":
        example_with_validation()

if __name__ == '__main__':
    main()
