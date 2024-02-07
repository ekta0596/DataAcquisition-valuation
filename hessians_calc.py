import pandas as pd 
from classifier import LogisticRegression
import torch
import tqdm
import numpy as np
from utils import *

def hessian_one_point(model, x, y, loss_func):
    x, y = torch.FloatTensor(x), torch.FloatTensor([y])
    loss = loss_func(model, x, y)
    params = [ p for p in model.parameters() if p.requires_grad ]
    first_grads = convert_grad_to_tensor(grad(loss, params, retain_graph=True, create_graph=True))
    hv = np.zeros((len(first_grads), len(first_grads)))
    for i in range(len(first_grads)):
        hv[i, :] = convert_grad_to_ndarray(grad(first_grads[i], params, create_graph=True)).ravel()
    return hv

def get_hessian_all_points(model, x_train, y_train, loss_func):
    hessian_all_points = []
    tbar = tqdm.tqdm(total=len(x_train))
    for i in range(len(x_train)):
        hessian_all_points.append(hessian_one_point(model, x_train[i], y_train[i], loss_func)/len(x_train))
        tbar.update(1)
    hessian_all_points = np.array(hessian_all_points)
    return hessian_all_points

# Compute multiplication of inverse hessian matrix and vector v
def get_hinv_v(hessian_all_points, v, hinv=None, recursive=False):
    if recursive:
        raise NotImplementedError
    else:
        if hinv is None:
            hinv = np.linalg.pinv(np.sum(hessian_all_points, axis=0))
        hinv_v = np.matmul(hinv, v)

    return hinv_v, hinv

def logistic_loss_torch(model, x, y_true):
    y_pred = model(torch.FloatTensor(x))
    return binary_cross_entropy(y_pred, torch.FloatTensor([y_true]))


