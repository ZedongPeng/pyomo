#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Cut generation."""
from math import copysign
from pyomo.core import minimize, value, RangeSet, Var, ConstraintList, Objective, Set
from pyomo.core.base.var import _GeneralVarData
import pyomo.core.expr as EXPR
from pyomo.contrib.gdpopt.util import time_code
from pyomo.contrib.mcpp.pyomo_mcpp import McCormick as mc, MCPP_Error
import os
import cplex
import time


def add_oa_cuts(
    target_model,
    dual_values,
    jacobians,
    objective_sense,
    mip_constraint_polynomial_degree,
    mip_iter,
    config,
    timing,
    cb_opt=None,
    linearize_active=True,
    linearize_violated=True,
):
    """Adds OA cuts.

    Generates and adds OA cuts (linearizes nonlinear constraints).
    For nonconvex problems, turn on 'config.add_slack'.
    Slack variables will always be used for nonlinear equality constraints.

    Parameters
    ----------
    target_model : Pyomo model
        The relaxed linear model.
    dual_values : list
        The value of the duals for each constraint.
    jacobians : ComponentMap
        Map nonlinear_constraint --> Map(variable --> jacobian of constraint w.r.t. variable).
    objective_sense : Int
        Objective sense of model.
    mip_constraint_polynomial_degree : Set
        The polynomial degrees of constraints that are regarded as linear.
    mip_iter : Int
        MIP iteration counter.
    config : ConfigBlock
        The specific configurations for MindtPy.
    cb_opt : SolverFactory, optional
        Gurobi_persistent solver, by default None.
    linearize_active : bool, optional
        Whether to linearize the active nonlinear constraints, by default True.
    linearize_violated : bool, optional
        Whether to linearize the violated nonlinear constraints, by default True.
    """
    with time_code(timing, 'OA cut generation'):
        for index, constr in enumerate(target_model.MindtPy_utils.constraint_list):
            # TODO: here the index is correlated to the duals, try if this can be fixed when temp duals are removed.
            if constr.body.polynomial_degree() in mip_constraint_polynomial_degree:
                continue

            constr_vars = list(EXPR.identify_variables(constr.body))
            jacs = jacobians

            # Equality constraint (makes the problem nonconvex)
            if (
                constr.has_ub()
                and constr.has_lb()
                and value(constr.lower) == value(constr.upper)
                and config.equality_relaxation
            ):
                sign_adjust = -1 if objective_sense == minimize else 1
                rhs = constr.lower
                if config.add_slack:
                    slack_var = target_model.MindtPy_utils.cuts.slack_vars.add()
                target_model.MindtPy_utils.cuts.oa_cuts.add(
                    expr=copysign(1, sign_adjust * dual_values[index])
                    * (
                        sum(
                            value(jacs[constr][var]) * (var - value(var))
                            for var in EXPR.identify_variables(constr.body)
                        )
                        + value(constr.body)
                        - rhs
                    )
                    - (slack_var if config.add_slack else 0)
                    <= 0
                )
                if (
                    config.single_tree
                    and config.mip_solver == 'gurobi_persistent'
                    and mip_iter > 0
                    and cb_opt is not None
                ):
                    cb_opt.cbLazy(
                        target_model.MindtPy_utils.cuts.oa_cuts[
                            len(target_model.MindtPy_utils.cuts.oa_cuts)
                        ]
                    )

            else:  # Inequality constraint (possibly two-sided)
                if (
                    constr.has_ub()
                    and (
                        linearize_active
                        and abs(constr.uslack()) < config.zero_tolerance
                    )
                    or (linearize_violated and constr.uslack() < 0)
                    or (config.linearize_inactive and constr.uslack() > 0)
                ) or (
                    'MindtPy_utils.objective_constr' in constr.name and constr.has_ub()
                ):
                    # always add the linearization for the epigraph of the objective
                    if config.add_slack:
                        slack_var = target_model.MindtPy_utils.cuts.slack_vars.add()

                    target_model.MindtPy_utils.cuts.oa_cuts.add(
                        expr=(
                            sum(
                                value(jacs[constr][var]) * (var - var.value)
                                for var in constr_vars
                            )
                            + value(constr.body)
                            - (slack_var if config.add_slack else 0)
                            <= value(constr.upper)
                        )
                    )
                    if (
                        config.single_tree
                        and config.mip_solver == 'gurobi_persistent'
                        and mip_iter > 0
                        and cb_opt is not None
                    ):
                        cb_opt.cbLazy(
                            target_model.MindtPy_utils.cuts.oa_cuts[
                                len(target_model.MindtPy_utils.cuts.oa_cuts)
                            ]
                        )

                if (
                    constr.has_lb()
                    and (
                        linearize_active
                        and abs(constr.lslack()) < config.zero_tolerance
                    )
                    or (linearize_violated and constr.lslack() < 0)
                    or (config.linearize_inactive and constr.lslack() > 0)
                ) or (
                    'MindtPy_utils.objective_constr' in constr.name and constr.has_lb()
                ):
                    if config.add_slack:
                        slack_var = target_model.MindtPy_utils.cuts.slack_vars.add()

                    target_model.MindtPy_utils.cuts.oa_cuts.add(
                        expr=(
                            sum(
                                value(jacs[constr][var]) * (var - var.value)
                                for var in constr_vars
                            )
                            + value(constr.body)
                            + (slack_var if config.add_slack else 0)
                            >= value(constr.lower)
                        )
                    )
                    if (
                        config.single_tree
                        and config.mip_solver == 'gurobi_persistent'
                        and mip_iter > 0
                        and cb_opt is not None
                    ):
                        cb_opt.cbLazy(
                            target_model.MindtPy_utils.cuts.oa_cuts[
                                len(target_model.MindtPy_utils.cuts.oa_cuts)
                            ]
                        )


def add_oa_cuts_for_grey_box(
    target_model, jacobians_model, config, objective_sense, mip_iter, cb_opt=None
):
    sign_adjust = -1 if objective_sense == minimize else 1
    if config.add_slack:
        slack_var = target_model.MindtPy_utils.cuts.slack_vars.add()
    for target_model_grey_box, jacobian_model_grey_box in zip(
        target_model.MindtPy_utils.grey_box_list,
        jacobians_model.MindtPy_utils.grey_box_list,
    ):
        jacobian_matrix = (
            jacobian_model_grey_box.get_external_model()
            .evaluate_jacobian_outputs()
            .toarray()
        )
        for index, output in enumerate(target_model_grey_box.outputs.values()):
            dual_value = jacobians_model.dual[jacobian_model_grey_box][
                output.name.replace("outputs", "output_constraints")
            ]
            target_model.MindtPy_utils.cuts.oa_cuts.add(
                expr=copysign(1, sign_adjust * dual_value)
                * (
                    sum(
                        jacobian_matrix[index][var_index] * (var - value(var))
                        for var_index, var in enumerate(
                            target_model_grey_box.inputs.values()
                        )
                    )
                )
                - (output - value(output))
                - (slack_var if config.add_slack else 0)
                <= 0
            )
    # TODO: gurobi_persistent currently does not support greybox model.
    # https://github.com/Pyomo/pyomo/issues/3000
    # if (
    #     config.single_tree
    #     and config.mip_solver == 'gurobi_persistent'
    #     and mip_iter > 0
    #     and cb_opt is not None
    # ):
    #     cb_opt.cbLazy(
    #         target_model.MindtPy_utils.cuts.oa_cuts[
    #             len(target_model.MindtPy_utils.cuts.oa_cuts)
    #         ]
    #     )


def add_ecp_cuts(
    target_model,
    jacobians,
    config,
    timing,
    linearize_active=True,
    linearize_violated=True,
):
    """Linearizes nonlinear constraints. Adds the cuts for the ECP method.

    Parameters
    ----------
    target_model : Pyomo model
        The relaxed linear model.
    jacobians : ComponentMap
        Map nonlinear_constraint --> Map(variable --> jacobian of constraint w.r.t. variable)
    config : ConfigBlock
        The specific configurations for MindtPy.
    timing : Timing
        Timing.
    linearize_active : bool, optional
        Whether to linearize the active nonlinear constraints, by default True.
    linearize_violated : bool, optional
        Whether to linearize the violated nonlinear constraints, by default True.
    """
    with time_code(timing, 'ECP cut generation'):
        for constr in target_model.MindtPy_utils.nonlinear_constraint_list:
            constr_vars = list(EXPR.identify_variables(constr.body))
            jacs = jacobians

            if constr.has_lb() and constr.has_ub():
                config.logger.warning(
                    'constraint {} has both a lower '
                    'and upper bound.'
                    '\n'.format(constr)
                )
                continue
            if constr.has_ub():
                try:
                    upper_slack = constr.uslack()
                except (ValueError, OverflowError) as e:
                    config.logger.error(
                        str(e) + '\nConstraint {} has caused either a '
                        'ValueError or OverflowError.'
                        '\n'.format(constr)
                    )
                    continue
                if (
                    (linearize_active and abs(upper_slack) < config.ecp_tolerance)
                    or (linearize_violated and upper_slack < 0)
                    or (config.linearize_inactive and upper_slack > 0)
                ):
                    if config.add_slack:
                        slack_var = target_model.MindtPy_utils.cuts.slack_vars.add()

                    target_model.MindtPy_utils.cuts.ecp_cuts.add(
                        expr=(
                            sum(
                                value(jacs[constr][var]) * (var - var.value)
                                for var in constr_vars
                            )
                            - (slack_var if config.add_slack else 0)
                            <= upper_slack
                        )
                    )

            if constr.has_lb():
                try:
                    lower_slack = constr.lslack()
                except (ValueError, OverflowError) as e:
                    config.logger.error(
                        str(e) + '\nConstraint {} has caused either a '
                        'ValueError or OverflowError.'
                        '\n'.format(constr)
                    )
                    continue
                if (
                    (linearize_active and abs(lower_slack) < config.ecp_tolerance)
                    or (linearize_violated and lower_slack < 0)
                    or (config.linearize_inactive and lower_slack > 0)
                ):
                    if config.add_slack:
                        slack_var = target_model.MindtPy_utils.cuts.slack_vars.add()

                    target_model.MindtPy_utils.cuts.ecp_cuts.add(
                        expr=(
                            sum(
                                value(jacs[constr][var]) * (var - var.value)
                                for var in constr_vars
                            )
                            + (slack_var if config.add_slack else 0)
                            >= -lower_slack
                        )
                    )


def add_no_good_cuts(target_model, var_values, config, timing, mip_iter=0, cb_opt=None):
    """Adds no-good cuts.

    This adds an no-good cuts to the no_good_cuts ConstraintList, which is not activated by default.
    However, it may be activated as needed in certain situations or for certain values of option flags.


    Parameters
    ----------
    target_model : Block
        The model to add no-good cuts to.
    var_values : list
        Variable values of the current solution, used to generate the cut.
    config : ConfigBlock
        The specific configurations for MindtPy.
    timing : Timing
        Timing.
    mip_iter : Int, optional
        MIP iteration counter.
    cb_opt : SolverFactory, optional
        Gurobi_persistent solver, by default None.

    Raises
    ------
    ValueError
        The value of binary variable is not 0 or 1.
    """
    if not config.add_no_good_cuts:
        return
    with time_code(timing, 'no_good cut generation'):
        config.logger.debug('Adding no-good cuts')

        m = target_model
        MindtPy = m.MindtPy_utils
        int_tol = config.integer_tolerance

        binary_vars = [v for v in MindtPy.variable_list if v.is_binary()]

        # copy variable values over
        for var, val in zip(MindtPy.variable_list, var_values):
            if not var.is_binary():
                continue
            var.set_value(val, skip_validation=True)

        # check to make sure that binary variables are all 0 or 1
        for v in binary_vars:
            if value(abs(v - 1)) > int_tol and value(abs(v)) > int_tol:
                raise ValueError(
                    'Binary {} = {} is not 0 or 1'.format(v.name, value(v))
                )

        if not binary_vars:  # if no binary variables, skip
            return

        int_cut = (
            sum(1 - v for v in binary_vars if value(abs(v - 1)) <= int_tol)
            + sum(v for v in binary_vars if value(abs(v)) <= int_tol)
            >= 1
        )

        MindtPy.cuts.no_good_cuts.add(expr=int_cut)
        if (
            config.single_tree
            and config.mip_solver == 'gurobi_persistent'
            and mip_iter > 0
            and cb_opt is not None
        ):
            cb_opt.cbLazy(
                target_model.MindtPy_utils.cuts.no_good_cuts[
                    len(target_model.MindtPy_utils.cuts.no_good_cuts)
                ]
            )


def add_affine_cuts(target_model, config, timing):
    """Adds affine cuts using MCPP.

    Parameters
    ----------
    config : ConfigBlock
        The specific configurations for MindtPy.
    timing : Timing
        Timing.
    """
    with time_code(timing, 'Affine cut generation'):
        m = target_model
        config.logger.debug('Adding affine cuts')
        counter = 0

        for constr in m.MindtPy_utils.nonlinear_constraint_list:
            vars_in_constr = list(EXPR.identify_variables(constr.body))
            if any(var.value is None for var in vars_in_constr):
                continue  # a variable has no values

            # mcpp stuff
            try:
                mc_eqn = mc(constr.body)
            except MCPP_Error as e:
                config.logger.error(
                    '\nSkipping constraint %s due to MCPP error %s'
                    % (constr.name, str(e))
                )
                continue  # skip to the next constraint

            ccSlope = mc_eqn.subcc()
            cvSlope = mc_eqn.subcv()
            ccStart = mc_eqn.concave()
            cvStart = mc_eqn.convex()

            # check if the value of ccSlope and cvSlope is not Nan or inf. If so, we skip this.
            concave_cut_valid = True
            convex_cut_valid = True
            for var in vars_in_constr:
                if not var.fixed:
                    if ccSlope[var] == float('nan') or ccSlope[var] == float('inf'):
                        concave_cut_valid = False
                    if cvSlope[var] == float('nan') or cvSlope[var] == float('inf'):
                        convex_cut_valid = False
            # check if the value of ccSlope and cvSlope all equals zero. if so, we skip this.
            if not any(list(ccSlope.values())):
                concave_cut_valid = False
            if not any(list(cvSlope.values())):
                convex_cut_valid = False
            if ccStart == float('nan') or ccStart == float('inf'):
                concave_cut_valid = False
            if cvStart == float('nan') or cvStart == float('inf'):
                convex_cut_valid = False
            if not (concave_cut_valid or convex_cut_valid):
                continue

            ub_int = (
                min(value(constr.upper), mc_eqn.upper())
                if constr.has_ub()
                else mc_eqn.upper()
            )
            lb_int = (
                max(value(constr.lower), mc_eqn.lower())
                if constr.has_lb()
                else mc_eqn.lower()
            )

            aff_cuts = m.MindtPy_utils.cuts.aff_cuts
            if concave_cut_valid:
                concave_cut = (
                    sum(
                        ccSlope[var] * (var - var.value)
                        for var in vars_in_constr
                        if not var.fixed
                    )
                    + ccStart
                    >= lb_int
                )
                aff_cuts.add(expr=concave_cut)
                counter += 1
            if convex_cut_valid:
                convex_cut = (
                    sum(
                        cvSlope[var] * (var - var.value)
                        for var in vars_in_constr
                        if not var.fixed
                    )
                    + cvStart
                    <= ub_int
                )
                aff_cuts.add(expr=convex_cut)
                counter += 1

        config.logger.debug('Added %s affine cuts' % counter)

def add_baron_cuts(model):
    special_baron_path = "/local/scratch/a/peng372/opt/baron4ieg"
    timea = time.time()
    output_filename, symbol_map = model.baronwrite(
        "root_relaxation_baron.bar", format="bar")
    var_ids = symbol_map.byObject
    # dolocal: Local search option for upper bounding. 0: no local search is done during upper bounding. 1: BARON automatically decides when to apply local search based on analyzing the results of previous local searches
    os.system("sed -i '1 a dolocal:0; ' root_relaxation_baron.bar")
    # NumLoc: Number of local searches done in preprocessing.  If NumLoc is set to −1, local searches in preprocessing will be done until proof of globality or MaxTime is reached. If NumLoc is set to −2, BARON decides the number of local searches in preprocessing based on problem and NLP solver characteristics.
    os.system("sed -i '1 a numloc:0; ' root_relaxation_baron.bar")
    os.system("sed -i '1 a maxtime:10000; ' root_relaxation_baron.bar")
    # Option to control log output. 0: all log output is suppressed. 1: print log output.
    os.system("sed -i '1 a prlevel:0; ' root_relaxation_baron.bar")
    os.system("sed -i '1 a ppdo:0; ' root_relaxation_baron.bar")
    os.system("sed -i '1 a pscdo:0; ' root_relaxation_baron.bar")
    # os.system('''sed -i '1 a CplexLibName: "/opt/ibm/ILOG/CPLEX_Studio129/cplex/bin/x86-64_linux/libcplex1290.so"; ' root_relaxation_baron.bar''')
    os.system('''sed -i '1 a CplexLibName: "/package/cplex/22.1/cplex/bin/x86-64_linux/libcplex2210.so"; ' root_relaxation_baron.bar''')
    os.system(special_baron_path + " root_relaxation_baron.bar")
    cplex_model = cplex.Cplex("relax.lp")
    timeb = time.time()
    print("lp file generation time", timeb - timea)
    # create additional variables in the pyomo model
    var_names = cplex_model.variables.get_names()
    num_bar_vars = sum("bar_var" in var for var in var_names)
    num_bar_vars_list = []
    # sometimes the index of bar_var is not continuous
    # Therefore, we cannot use RangeSet
    for var in var_names:
        if "bar_var" in var:
            num_bar_vars_list.append(int(var.split("bar_var")[1]))
    # model.bar_set = RangeSet(num_bar_vars)
    model.bar_set = Set(initialize=num_bar_vars_list)
    model.bar_var = Var(model.bar_set)

    # create a map from cplex var id to pyomo var name
    varid_to_var = {}
    bar_var_indices = []
    cplex_var_names = cplex_model.variables.get_names()
    for vid in var_ids:
        name = symbol_map.byObject[vid]
        var_data = symbol_map.bySymbol[name]
        if isinstance(symbol_map.bySymbol[name], _GeneralVarData):
            # sometimes some variables only appear in baron file not in cplex
            if name in cplex_var_names:
                varid_cplex = cplex_model.variables.get_indices(name)
                varid_to_var[varid_cplex] = var_data
    for i in range(len(cplex_var_names)):
        varname = cplex_var_names[i]
        if "bar_var" in varname:
            varid_pyomo = int(varname.split("bar_var")[1])
            varid_to_var[i] = model.bar_var[varid_pyomo]
            bar_var_indices.append(i)

    # update variable bounds in pyomo
    var_lb = cplex_model.variables.get_lower_bounds()
    var_ub = cplex_model.variables.get_upper_bounds()
    for i in range(len(var_lb)):
        if i in varid_to_var:
            var = varid_to_var[i]
            var.setlb(var_lb[i])
            var.setub(var_ub[i])
    # #To create a list that contain information of the linear constraints in the original pyomo model
    # for c in model.component_objects(Constraint):

    # add constraints that have bar_var
    model.baroncuts = ConstraintList()
    nconstraints = cplex_model.linear_constraints.get_num()
    for c in range(nconstraints):
        row = cplex_model.linear_constraints.get_rows(c)
        rhs = cplex_model.linear_constraints.get_rhs(c)
        sense = cplex_model.linear_constraints.get_senses(c)
        if sum(varid in bar_var_indices for varid in row.ind) > 0:
            expr = sum(row.val[i] * varid_to_var[row.ind[i]]
                       for i in range(len(row.ind)))
            if sense == 'G':
                model.baroncuts.add(expr >= rhs)
            if sense == 'L':
                model.baroncuts.add(expr <= rhs)
            if sense == 'E':
                model.baroncuts.add(expr == rhs)
    # change objective
    # move nonlinear objective function to constraint
    # next(model.component_data_objects(Objective, active=True)).deactivate()
    # model.obj.deactivate()
    # coeff = cplex_model.objective.get_linear()
    # if cplex_model.objective.get_sense() == 1:
    #     model.baron_obj = Objective(expr=sum(varid_to_var[i] * coeff[i] for i in range(
    #         cplex_model.variables.get_num()) if i in varid_to_var.keys()), sense=minimize)
    # else:
    #     model.baron_obj = Objective(expr=sum(varid_to_var[i] * coeff[i] for i in range(
    #         cplex_model.variables.get_num()) if i in varid_to_var.keys()), sense=maximize)

    timec = time.time()
    print("time to add the cuts to pyomo model", timec-timeb)
