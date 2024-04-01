#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import datetime
import io
import math
import os

from pyomo.common.config import ConfigValue
from pyomo.common.dependencies import attempt_import
from pyomo.common.shutdown import python_is_shutting_down
from pyomo.common.tee import capture_output, TeeStream
from pyomo.common.timing import HierarchicalTimer

from pyomo.contrib.solver.base import SolverBase
from pyomo.contrib.solver.config import BranchAndBoundConfig
from pyomo.contrib.solver.results import Results, SolutionStatus, TerminationCondition
from pyomo.contrib.solver.solution import SolutionLoaderBase

from pyomo.core.staleflag import StaleFlagManager

from pyomo.repn.plugins.standard_form import LinearStandardFormCompiler

gurobipy, gurobipy_available = attempt_import('gurobipy')


class GurobiConfig(BranchAndBoundConfig):
    def __init__(
        self,
        description=None,
        doc=None,
        implicit=False,
        implicit_domain=None,
        visibility=0,
    ):
        super(GurobiConfig, self).__init__(
            description=description,
            doc=doc,
            implicit=implicit,
            implicit_domain=implicit_domain,
            visibility=visibility,
        )
        self.use_mipstart: bool = self.declare(
            'use_mipstart',
            ConfigValue(
                default=False,
                domain=bool,
                description="If True, the values of the integer variables will be passed to Gurobi.",
            ),
        )


class GurobiDirectSolutionLoader(SolutionLoaderBase):
    def __init__(self, grb_model, grb_vars, pyo_vars):
        self._grb_model = grb_model
        self._grb_vars = grb_vars
        self._pyo_vars = pyo_vars
        GurobiDirect._num_instances += 1

    def __del__(self):
        if not python_is_shutting_down():
            GurobiDirect._num_instances -= 1
            if GurobiDirect._num_instances == 0:
                GurobiDirect.release_license()

    def load_vars(self, vars_to_load=None, solution_number=0):
        assert vars_to_load is None
        assert solution_number == 0
        for p_var, g_var in zip(self._pyo_vars, self._grb_vars.x.tolist()):
            p_var.set_value(g_var, skip_validation=True)

    def get_primals(self, vars_to_load=None):
        assert vars_to_load is None
        assert solution_number == 0
        return ComponentMap(zip(self._pyo_vars, self._grb_vars.x.tolist()))


class GurobiDirect(SolverBase):
    CONFIG = GurobiConfig()

    _available = None
    _num_instances = 0
    _tc_map = None

    def __init__(self, **kwds):
        super().__init__(**kwds)
        GurobiDirect._num_instances += 1

    def available(self):
        if not gurobipy_available:  # this triggers the deferred import
            return self.Availability.NotFound
        elif self._available == self.Availability.BadVersion:
            return self.Availability.BadVersion
        else:
            return self._check_license()

    def _check_license(self):
        avail = False
        try:
            # Gurobipy writes out license file information when creating
            # the environment
            with capture_output(capture_fd=True):
                m = gurobipy.Model()
            avail = True
        except gurobipy.GurobiError:
            avail = False

        if avail:
            if self._available is None:
                self._available = GurobiDirect._check_full_license(m)
            return self._available
        else:
            return self.Availability.BadLicense

    @classmethod
    def _check_full_license(cls, model=None):
        if model is None:
            model = gurobipy.Model()
        model.setParam('OutputFlag', 0)
        try:
            model.addVars(range(2001))
            model.optimize()
            return cls.Availability.FullLicense
        except gurobipy.GurobiError:
            return cls.Availability.LimitedLicense

    def __del__(self):
        if not python_is_shutting_down():
            GurobiDirect._num_instances -= 1
            if GurobiDirect._num_instances == 0:
                self.release_license()

    @staticmethod
    def release_license():
        if gurobipy_available:
            with capture_output(capture_fd=True):
                gurobipy.disposeDefaultEnv()

    def version(self):
        version = (
            gurobipy.GRB.VERSION_MAJOR,
            gurobipy.GRB.VERSION_MINOR,
            gurobipy.GRB.VERSION_TECHNICAL,
        )
        return version

    def solve(self, model, **kwds) -> Results:
        start_timestamp = datetime.datetime.now(datetime.timezone.utc)
        self._config = config = self.config(value=kwds, preserve_implicit=True)
        StaleFlagManager.mark_all_as_stale()
        if config.timer is None:
            config.timer = HierarchicalTimer()
        timer = config.timer

        timer.start('compile_model')
        repn = LinearStandardFormCompiler().write(model, mixed_form=True)
        timer.stop('compile_model')

        timer.start('prepare_matrices')
        inf = float('inf')
        ninf = -inf
        lb = []
        ub = []
        for v in repn.columns:
            _l, _u = v.bounds
            if _l is None:
                _l = ninf
            if _u is None:
                _u = inf
            lb.append(_l)
            ub.append(_u)
        vtype = [
            (
                gurobipy.GRB.CONTINUOUS
                if v.is_continuous()
                else (
                    gurobipy.GRB.BINARY
                    if v.is_binary()
                    else gurobipy.GRB.INTEGER if v.is_integer() else '?'
                )
            )
            for v in repn.columns
        ]
        sense_type = '>=<'
        sense = [sense_type[r[1] + 1] for r in repn.rows]
        timer.stop('prepare_matrices')

        ostreams = [io.StringIO()] + config.tee

        try:
            orig_cwd = os.getcwd()
            if self._config.working_dir:
                os.chdir(self._config.working_dir)
            with TeeStream(*ostreams) as t, capture_output(t.STDOUT, capture_fd=False):
                gurobi_model = gurobipy.Model()

                timer.start('transfer_model')
                x = gurobi_model.addMVar(
                    len(repn.columns),
                    lb=lb,
                    ub=ub,
                    obj=repn.c.todense()[0],
                    vtype=vtype,
                )
                A = gurobi_model.addMConstr(repn.A, x, sense, repn.rhs)
                # gurobi_model.update()
                timer.stop('transfer_model')

                options = config.solver_options

                gurobi_model.setParam('LogToConsole', 1)

                if config.threads is not None:
                    gurobi_model.setParam('Threads', config.threads)
                if config.time_limit is not None:
                    gurobi_model.setParam('TimeLimit', config.time_limit)
                if config.rel_gap is not None:
                    gurobi_model.setParam('MIPGap', config.rel_gap)
                if config.abs_gap is not None:
                    gurobi_model.setParam('MIPGapAbs', config.abs_gap)

                if config.use_mipstart:
                    raise MouseTrap("MIPSTART not yet supported")

                for key, option in options.items():
                    gurobi_model.setParam(key, option)

                timer.start('optimize')
                gurobi_model.optimize()
                timer.stop('optimize')
        finally:
            os.chdir(orig_cwd)

        res = self._postsolve(
            timer, GurobiDirectSolutionLoader(gurobi_model, x, repn.columns)
        )
        res.solver_configuration = config
        res.solver_name = 'Gurobi'
        res.solver_version = self.version()
        res.solver_log = ostreams[0].getvalue()

        end_timestamp = datetime.datetime.now(datetime.timezone.utc)
        res.timing_info.start_timestamp = start_timestamp
        res.timing_info.wall_time = (end_timestamp - start_timestamp).total_seconds()
        res.timing_info.timer = timer
        return res

    def _postsolve(self, timer: HierarchicalTimer, loader):
        config = self._config

        gprob = loader._grb_model
        status = gprob.Status

        results = Results()
        results.solution_loader = loader
        results.timing_info.gurobi_time = gprob.Runtime

        if gprob.SolCount > 0:
            if status == gurobipy.GRB.OPTIMAL:
                results.solution_status = SolutionStatus.optimal
            else:
                results.solution_status = SolutionStatus.feasible
        else:
            results.solution_status = SolutionStatus.noSolution

        results.termination_condition = self._get_tc_map().get(
            status, TerminationCondition.unknown
        )

        if (
            results.termination_condition
            != TerminationCondition.convergenceCriteriaSatisfied
            and config.raise_exception_on_nonoptimal_result
        ):
            raise RuntimeError(
                'Solver did not find the optimal solution. Set '
                'opt.config.raise_exception_on_nonoptimal_result=False '
                'to bypass this error.'
            )

        results.incumbent_objective = None
        results.objective_bound = None
        try:
            results.incumbent_objective = gprob.ObjVal
        except (gurobipy.GurobiError, AttributeError):
            results.incumbent_objective = None
        try:
            results.objective_bound = gprob.ObjBound
        except (gurobipy.GurobiError, AttributeError):
            if self._objective.sense == minimize:
                results.objective_bound = -math.inf
            else:
                results.objective_bound = math.inf

        if results.incumbent_objective is not None and not math.isfinite(
            results.incumbent_objective
        ):
            results.incumbent_objective = None

        results.iteration_count = gprob.getAttr('IterCount')

        timer.start('load solution')
        if config.load_solutions:
            if gprob.SolCount > 0:
                results.solution_loader.load_vars()
            else:
                raise RuntimeError(
                    'A feasible solution was not found, so no solution can be loaded.'
                    'Please set opt.config.load_solutions=False and check '
                    'results.solution_status and '
                    'results.incumbent_objective before loading a solution.'
                )
        timer.stop('load solution')

        return results

    def _get_tc_map(self):
        if GurobiDirect._tc_map is None:
            grb = gurobipy.GRB
            tc = TerminationCondition
            GurobiDirect._tc_map = {
                grb.LOADED: tc.unknown,  # problem is loaded, but no solution
                grb.OPTIMAL:  tc.convergenceCriteriaSatisfied,
                grb.INFEASIBLE: tc.provenInfeasible,
                grb.INF_OR_UNBD: tc.infeasibleOrUnbounded,
                grb.UNBOUNDED: tc.unbounded,
                grb.CUTOFF: tc.objectiveLimit,
                grb.ITERATION_LIMIT: tc.iterationLimit,
                grb.NODE_LIMIT: tc.iterationLimit,
                grb.TIME_LIMIT: tc.maxTimeLimit,
                grb.SOLUTION_LIMIT: tc.unknown,
                grb.INTERRUPTED: tc.interrupted,
                grb.NUMERIC: tc.unknown,
                grb.SUBOPTIMAL: tc.unknown,
                grb.USER_OBJ_LIMIT: tc.objectiveLimit,
            }
        return GurobiDirect._tc_map
