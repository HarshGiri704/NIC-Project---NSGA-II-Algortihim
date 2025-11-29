import os
import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.termination import get_termination
from pymoo.optimize import minimize


# ========== CONFIGURATION ==========

# Folder where your .txt and .f files live
FOLDER = "/Users/harshgiri/Documents/NIC GRP PROJECT"

PROBLEM_NAME = "pla33810-n338090"

TXT_FILE = os.path.join(FOLDER, f"{PROBLEM_NAME}.txt")
OUT_FILE = os.path.join(FOLDER, f"NSGA-II_{PROBLEM_NAME}.f")


# ========== DATA LOADING ==========

def load_points(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                t, p = float(parts[0]), float(parts[1])
                rows.append([t, p])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"No numeric data found in {path}")
    return np.array(rows)


X_data = load_points(TXT_FILE)          # shape (N, 2): [time, negative_profit]

# normalize to [0,1] to help the optimizer
x_min = X_data.min(axis=0)
x_max = X_data.max(axis=0)
X_range = x_max - x_min + 1e-9   # avoid division by zero


# ========== DEFINE PYMOO PROBLEM ==========

class TwoObjProblem(Problem):
    """
    Decision variables: 2 numbers in [0,1]
    Objectives: scaled back to your original (time, negative_profit) values.
    """
    def __init__(self):
        super().__init__(
            n_var=2,
            n_obj=2,
            n_constr=0,
            xl=np.array([0.0, 0.0]),
            xu=np.array([1.0, 1.0])
        )

    def _evaluate(self, X, out, *args, **kwargs):
        # map normalized decision variables back to real time/profit scale
        X_real = X * X_range + x_min     # shape (pop_size, 2)
        # NSGA-II will try to minimise both objectives,
        # which in your case are exactly time and negative profit.
        out["F"] = X_real


problem = TwoObjProblem()


# ========== NSGA-II ALGORITHM SETUP ==========

algorithm = NSGA2(
    pop_size=100,              # can raise later if it runs fast
    eliminate_duplicates=True
)

termination = get_termination("n_gen", 50)   # 50 generations


# ========== RUN OPTIMIZATION ==========

if __name__ == "__main__":
    print(f"Running pymoo NSGA-II for problem: {PROBLEM_NAME}")
    print(f"Data file: {TXT_FILE}")

    res = minimize(
        problem,
        algorithm,
        termination,
        seed=42,
        verbose=True
    )

    # res.F is the final non-dominated set in objective space
    F = res.F   # shape (M, 2) -> [time, negative_profit]

    # save as .f file (same format as other participants)
    np.savetxt(OUT_FILE, F, fmt="%.6f")
    print(f"NSGA-II Pareto front (pymoo) saved to {OUT_FILE}")
