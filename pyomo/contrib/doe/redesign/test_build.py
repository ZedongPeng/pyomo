from experiment_class_example import *
from pyomo.contrib.doe import *
from pyomo.contrib.doe import *

from simple_reaction_example import *

import numpy as np
import logging

doe_obj = [0, 0, 0, 0,]
obj = ['trace', 'det', 'det']
file_name = ['trash1.json', 'trash2.json', 'trash3.json']

for ind, fd in enumerate(['central', 'backward', 'forward']):
    experiment = FullReactorExperiment(data_ex, 10, 3)
    doe_obj[ind] = DesignOfExperiments_(
        experiment, 
        fd_formula='central',
        step=1e-3,
        objective_option=ObjectiveLib(obj[ind]),
        scale_constant_value=1,
        scale_nominal_param_value=(True and (ind != 2)),
        prior_FIM=None,
        jac_initial=None,
        fim_initial=None,
        L_initial=None,
        L_LB=1e-7,
        solver=None,
        tee=False,
        args=None,
        _Cholesky_option=True,
        _only_compute_fim_lower=True,
        logger_level=logging.INFO,
    )
    doe_obj[ind].run_doe(results_file=file_name[ind])


ind = 3

doe_obj[ind] = DesignOfExperiments_(
        experiment, 
        fd_formula='central',
        step=1e-3,
        objective_option=ObjectiveLib.det,
        scale_constant_value=1,
        scale_nominal_param_value=(True and (ind != 2)),
        prior_FIM=None,
        jac_initial=None,
        fim_initial=None,
        L_initial=None,
        L_LB=1e-7,
        solver=None,
        tee=False,
        args=None,
        _Cholesky_option=True,
        _only_compute_fim_lower=True,
        logger_level=logging.INFO,
    )
doe_obj[ind].model.set_blocks = pyo.Set(initialize=[0, 1, 2])
doe_obj[ind].model.block_instances = pyo.Block(doe_obj[ind].model.set_blocks)
doe_obj[ind].create_doe_model(mod=doe_obj[ind].model.block_instances[0])
doe_obj[ind].create_doe_model(mod=doe_obj[ind].model.block_instances[1])
doe_obj[ind].create_doe_model(mod=doe_obj[ind].model.block_instances[2])
# doe_obj[ind].run_doe()

print('Multi-block build complete')

# Old interface comparison
def create_model(m=None, ):
    experiment = FullReactorExperiment(data_ex, 10, 3)
    m = experiment.get_labeled_model().clone()
    return m

### Define inputs
# Control time set [h]
t_control = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
# Define parameter nominal value
parameter_dict = {"A1": 84.79, "A2": 371.72, "E1": 7.78, "E2": 15.05}

# measurement object
measurements = MeasurementVariables()
measurements.add_variables(
    "CA",  # name of measurement
    indices={0: t_control},  # indices of measurement
    time_index_position=0,
    variance=1e-2,
)  # position of time index

measurements.add_variables(
    "CB",  # name of measurement
    indices={0: t_control},  # indices of measurement
    time_index_position=0,
    variance=1e-2,
)  # position of time index

measurements.add_variables(
    "CC",  # name of measurement
    indices={0: t_control},  # indices of measurement
    time_index_position=0,
    variance=1e-2,
)  # position of time index

# design object
exp_design = DesignVariables()

# add CAO as design variable
exp_design.add_variables(
    "CA",  # name of design variable
    indices={0: [0]},  # indices of design variable
    time_index_position=0,  # position of time index
    values=[5],  # nominal value of design variable
    lower_bounds=1,  # lower bound of design variable
    upper_bounds=5,  # upper bound of design variable
)

# add T as design variable
exp_design.add_variables(
    "T",  # name of design variable
    indices={0: t_control},  # indices of design variable
    time_index_position=0,  # position of time index
    values=[
        500,
        300,
        300,
        300,
        300,
        300,
        300,
        300,
        300,
    ],  # nominal value of design variable
    lower_bounds=300,  # lower bound of design variable
    upper_bounds=700,  # upper bound of design variable
)

design_names = exp_design.variable_names
exp1 = [5, 500, 300, 300, 300, 300, 300, 300, 300, 300]
exp1_design_dict = dict(zip(design_names, exp1))
exp_design.update_values(exp1_design_dict)

# add a prior information (scaled FIM with T=500 and T=300 experiments)
# prior = np.asarray(
    # [
        # [28.67892806, 5.41249739, -81.73674601, -24.02377324],
        # [5.41249739, 26.40935036, -12.41816477, -139.23992532],
        # [-81.73674601, -12.41816477, 240.46276004, 58.76422806],
        # [-24.02377324, -139.23992532, 58.76422806, 767.25584508],
    # ]
# )

prior = None

doe_object = DesignOfExperiments(
    parameter_dict,  # dictionary of parameters
    exp_design,  # design variables
    measurements,  # measurement variables
    create_model,  # function to create model
    only_compute_fim_lower=True,
)

square_result, optimize_result = doe_object.stochastic_program(
    if_optimize=True,  # if optimize
    if_Cholesky=True,  # if use Cholesky decomposition
    scale_nominal_param_value=True,  # if scale nominal parameter value
    objective_option="det",  # objective option
)

doe_object2 = DesignOfExperiments(
    parameter_dict,  # dictionary of parameters
    exp_design,  # design variables
    measurements,  # measurement variables
    create_model,  # function to create model
    only_compute_fim_lower=True,
)

square_result2, optimize_result2 = doe_object2.stochastic_program(
    if_optimize=True,  # if optimize
    if_Cholesky=True,  # if use Cholesky decomposition
    scale_nominal_param_value=False,  # if scale nominal parameter value
    objective_option="det",  # objective option
)

# Testing the kaug runs
doe_object3 = DesignOfExperiments(
    parameter_dict,  # dictionary of parameters
    exp_design,  # design variables
    measurements,  # measurement variables
    create_model,  # function to create model
    only_compute_fim_lower=True,
)

res = doe_object3.compute_FIM(scale_nominal_param_value=True)
res.result_analysis()

design_ranges = {
    'CA[0]': [1, 5, 3], 
    'T[0]': [300, 700, 3],
}
doe_obj[0].compute_FIM_full_factorial(design_ranges=design_ranges, method='kaug')

doe_obj[0].compute_FIM(method='kaug')

doe_obj[0].compute_FIM(method='sequential')

print(res.FIM)
print(doe_obj[0].kaug_FIM)


# Optimal values
print("Optimal values for determinant optimized experimental design:")
print("New formulation, scaled: {}".format(pyo.value(doe_obj[1].model.Obj)))
print("New formulation, unscaled: {}".format(pyo.value(doe_obj[2].model.Obj)))
print("Old formulation, scaled: {}".format(pyo.value(optimize_result.model.Obj)))
print("Old formulation, unscaled: {}".format(pyo.value(optimize_result2.model.Obj)))

# Old values
FIM_vals_old = [pyo.value(optimize_result.model.fim[i, j]) for i in optimize_result.model.regression_parameters for j in optimize_result.model.regression_parameters]
L_vals_old = [pyo.value(optimize_result.model.L_ele[i, j]) for i in optimize_result.model.regression_parameters for j in optimize_result.model.regression_parameters]
Q_vals_old = [pyo.value(optimize_result.model.sensitivity_jacobian[i, j]) for i in optimize_result.model.regression_parameters for j in optimize_result.model.measured_variables]
sigma_inv_old = [1 / v for k,v in doe_object.measurement_vars.variance.items()]

FIM_vals_old_np = np.array(FIM_vals_old).reshape((4, 4))

for i in range(4):
    for j in range(4):
        if j > i:
            FIM_vals_old_np[j, i] = FIM_vals_old_np[i, j]

L_vals_old_np = np.array(L_vals_old).reshape((4, 4))
Q_vals_old_np = np.array(Q_vals_old).reshape((4, 27))

sigma_inv_old_np = np.zeros((27, 27))
for i in range(27):
    sigma_inv_old_np[i, i] = sigma_inv_old[i]

# New values
FIM_vals_new = [pyo.value(doe_obj[1].model.fim[i, j]) for i in doe_obj[1].model.parameter_names for j in doe_obj[1].model.parameter_names]
L_vals_new = [pyo.value(doe_obj[1].model.L_ele[i, j]) for i in doe_obj[1].model.parameter_names for j in doe_obj[1].model.parameter_names]
Q_vals_new = [pyo.value(doe_obj[1].model.sensitivity_jacobian[i, j]) for i in doe_obj[1].model.output_names for j in doe_obj[1].model.parameter_names]
sigma_inv_new = [1 / v for k,v in doe_obj[1].model.scenario_blocks[0].measurement_error.items()]
param_vals = np.array([[v for k, v in doe_obj[1].model.scenario_blocks[0].unknown_parameters.items()], ])

FIM_vals_new_np = np.array(FIM_vals_new).reshape((4, 4))

for i in range(4):
    for j in range(4):
        if j < i:
            FIM_vals_new_np[j, i] = FIM_vals_new_np[i, j]

L_vals_new_np = np.array(L_vals_new).reshape((4, 4))
Q_vals_new_np = np.array(Q_vals_new).reshape((27, 4))

sigma_inv_new_np = np.zeros((27, 27))
for i in range(27):
    sigma_inv_new_np[i, i] = sigma_inv_new[i]

rescaled_FIM = rescale_FIM(FIM=FIM_vals_new_np, param_vals=param_vals)

# Comparing values from compute FIM
print("Results from using compute FIM (first old, then new)")
print(res.FIM)
print(doe_obj[0].kaug_FIM)
print(doe_obj[0].seq_FIM)
print(np.log10(np.linalg.det(doe_obj[0].kaug_FIM)))
print(np.log10(np.linalg.det(doe_obj[0].seq_FIM)))
A = doe_obj[0].kaug_jac
B = doe_obj[0].seq_jac
print(np.sum((A - B) ** 2))

measurement_vals_model = []
meas_from_model = []
mod = doe_obj[0].model
for p in mod.parameter_names:
    fd_step_mult = 1
    param_ind = mod.parameter_names.data().index(p)
    
    # Different FD schemes lead to different scenarios for the computation
    if doe_obj[0].fd_formula == FiniteDifferenceStep.central:
        s1 = param_ind * 2
        s2 = param_ind * 2 + 1
        fd_step_mult = 2
    elif doe_obj[0].fd_formula == FiniteDifferenceStep.forward:
        s1 = param_ind + 1
        s2 = 0
    elif doe_obj[0].fd_formula == FiniteDifferenceStep.backward:
        s1 = 0
        s2 = param_ind + 1

    var_up = [pyo.value(k) for k, v in mod.scenario_blocks[s1].experiment_outputs.items()]
    var_lo = [pyo.value(k) for k, v in mod.scenario_blocks[s2].experiment_outputs.items()]
    
    meas_from_model.append(var_up)
    meas_from_model.append(var_lo)


# Optimal values
print("Optimal values for determinant optimized experimental design:")
print("New formulation, scaled: {}".format(pyo.value(doe_obj[1].model.Obj)))
print("New formulation, unscaled: {}".format(pyo.value(doe_obj[2].model.Obj)))
print("New formulation, rescaled: {}".format(np.log10(np.linalg.det(rescaled_FIM))))
print("Old formulation, scaled: {}".format(pyo.value(optimize_result.model.Obj)))
print("Old formulation, unscaled: {}".format(pyo.value(optimize_result2.model.Obj)))

# Draw figures
sens_vars = ['CA[0]', 'T[0]']
des_vars_fixed = {'T[' + str((i + 1) / 8) + ']': 300 for i in range(7)}
des_vars_fixed['T[1]'] = 300
doe_obj[0].draw_factorial_figure(title_text='', xlabel_text='', ylabel_text='', sensitivity_design_variables=sens_vars, fixed_design_variables=des_vars_fixed,)