import numpy as np
from ortools.linear_solver import pywraplp
from scipy.optimize import linear_sum_assignment


class Matcher:
    def __init__(self, algorithm=0):
        self.algorithm = algorithm
        if algorithm == "HA":
            # Hungarian algorithm
            self.solver = None
        elif algorithm == "MCF":
            # min-cost flow
            self.solver = pywraplp.Solver.CreateSolver("SCIP")
        else:
            raise NotImplementedError

    def match(
        self,
        aff_matrix,
        aff_thresh=0,
        det_scores=None,
        trk_scores=None,
        entry_scores=None,
        exit_scores=None,
        post_valid_mask=None,
        unused=None,
    ):
        num_dets, num_trks = aff_matrix.shape

        if self.algorithm == "HA":
            row_ind, col_ind = linear_sum_assignment(aff_matrix, maximize=True)
            valid_mask = aff_matrix[row_ind, col_ind] >= aff_thresh
            if post_valid_mask is not None:
                valid_mask &= post_valid_mask[row_ind, col_ind]
            row_ind = row_ind[valid_mask]
            col_ind = col_ind[valid_mask]
            match_matrix = np.zeros((num_dets, num_trks), dtype=bool)
            match_matrix[row_ind, col_ind] = True
            entry_vec = np.logical_not(np.any(match_matrix, axis=1))
            exit_vec = np.logical_not(np.any(match_matrix, axis=0))

            false_det_vec = np.zeros(num_dets, dtype=bool)
            false_trk_vec = np.zeros(num_trks, dtype=bool)

        elif self.algorithm == "MCF":
            assert len(entry_scores) == num_dets
            assert len(exit_scores) == num_trks

            self.solver.Clear()

            # Variables
            y_det_cls = [self.solver.BoolVar(f"y_det_cls_{i}") for i in range(num_dets)]
            y_trk_cls = [self.solver.BoolVar(f"y_trk_cls_{i}") for i in range(num_trks)]
            y_entry = [self.solver.BoolVar(f"y_entry_{i}") for i in range(num_dets)]
            y_exit = [self.solver.BoolVar(f"y_exit_{i}") for i in range(num_trks)]
            y_link = [
                [self.solver.BoolVar(f"y_exit_{i}_{j}") for j in range(num_trks)]
                for i in range(num_dets)
            ]

            # Constraints
            #   det = link + entry
            for i in range(num_dets):
                self.solver.Add(
                    self.solver.Sum(
                        [-y_det_cls[i]]
                        + [y_link[i][j] for j in range(num_trks)]
                        + [y_entry[i]]
                    )
                    == 0
                )
            #   trk = link + exit
            for j in range(num_trks):
                self.solver.Add(
                    self.solver.Sum(
                        [-y_trk_cls[j]]
                        + [y_link[i][j] for i in range(num_dets)]
                        + [y_exit[j]]
                    )
                    == 0
                )

            # Objective
            w_det_cls = [(det_scores[i] - 1) * y_det_cls[i] for i in range(num_dets)]
            w_trk_cls = [(trk_scores[i] - 1) * y_trk_cls[i] for i in range(num_trks)]
            w_entry = [entry_scores[i] * y_entry[i] for i in range(num_dets)]
            w_exit = [exit_scores[i] * y_exit[i] for i in range(num_trks)]
            w_link = [
                [aff_matrix[i, j] * y_link[i][j] for j in range(num_trks)]
                for i in range(num_dets)
            ]

            self.solver.Maximize(
                self.solver.Sum(
                    w_det_cls + w_trk_cls + w_entry + w_exit + sum(w_link, [])
                )
            )

            self.solver.Solve()

            match_matrix = np.array(
                [
                    [y_link[i][j].solution_value() for j in range(num_trks)]
                    for i in range(num_dets)
                ],
                dtype=bool,
            )
            entry_vec = np.array([y.solution_value() for y in y_entry], dtype=bool)
            exit_vec = np.array([y.solution_value() for y in y_exit], dtype=bool)

            if aff_thresh > 0:
                invalid_mask = np.logical_or(aff_matrix < aff_thresh, ~post_valid_mask)
                match_matrix[invalid_mask] = False

            false_det_vec = np.logical_not(
                np.logical_or(entry_vec, np.any(match_matrix, axis=1))
            )
            false_trk_vec = np.logical_not(
                np.logical_or(exit_vec, np.any(match_matrix, axis=0))
            )

        return match_matrix, entry_vec, exit_vec, false_det_vec, false_trk_vec
