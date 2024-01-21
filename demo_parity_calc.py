from utils import *

def del_predictive_parity_del_theta(model, X_test_orig, X_test, y_test):
    y_pred = model.predict_proba(X_test)
    num_params = len(convert_grad_to_ndarray(list(model.parameters())))
    del_f_protected = np.zeros((num_params, 1))
    del_f_privileged = np.zeros((num_params, 1))
    
    protected_idx = X_test_orig[X_test_orig['age']==0].index
    numProtected = len(protected_idx)
    privileged_idx = X_test_orig[X_test_orig['age']==1].index
    numPrivileged = len(privileged_idx)

    u_dash_protected = np.zeros((num_params,))
    v_protected = 0
    v_dash_protected = np.zeros((num_params,))
    u_protected = 0
    for i in range(len(protected_idx)):
        del_f_i = del_f_del_theta_i(model, X_test[protected_idx[i]])
        del_f_i_arr = convert_grad_to_ndarray(del_f_i)
        v_protected += y_pred[protected_idx[i]]
        v_dash_protected = np.add(v_dash_protected, del_f_i_arr)
        if y_test[protected_idx[i]] == 1:
            u_dash_protected = np.add(u_dash_protected, del_f_i_arr)
            u_protected += y_pred[protected_idx[i]]
    del_f_protected = (u_dash_protected * v_protected - u_protected * v_dash_protected)/(v_protected * v_protected)
    
    u_dash_privileged = np.zeros((num_params,))
    v_privileged = 0
    v_dash_privileged = np.zeros((num_params,))
    u_privileged = 0
    for i in range(len(privileged_idx)):
        del_f_i = del_f_del_theta_i(model, X_test[privileged_idx[i]])
        del_f_i_arr = convert_grad_to_ndarray(del_f_i)
        v_privileged += y_pred[privileged_idx[i]]
        v_dash_privileged = np.add(v_dash_privileged, del_f_i_arr)
        if y_test[privileged_idx[i]] == 1:
            u_dash_privileged = np.add(u_dash_privileged, del_f_i_arr)
            u_privileged += y_pred[privileged_idx[i]]
    del_f_privileged = (u_dash_privileged * v_privileged - u_privileged * v_dash_privileged)/(v_privileged * v_privileged)

    v = np.subtract(del_f_protected, del_f_privileged)
    return v


def del_L_del_theta_i(model, x, y_true, loss_func, retain_graph=False):
    loss = loss_func(model, x, y_true)
    w = [p for p in model.parameters() if p.requires_grad]
    return grad(loss, w, create_graph=True, retain_graph=retain_graph)


def get_del_L_del_theta(model, X_train, y_train, loss_func):
    del_L_del_theta = []
    for i in range(int(len(X_train))):
        gradient = convert_grad_to_ndarray(del_L_del_theta_i(model, X_train[i], int(y_train[i]), loss_func))
        while np.sum(np.isnan(gradient))>0:
            gradient = convert_grad_to_ndarray(del_L_del_theta_i(model, X_train[i], int(y_train[i]), loss_func))
        del_L_del_theta.append(gradient)
    del_L_del_theta = np.array(del_L_del_theta)
    return del_L_del_theta