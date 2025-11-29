import os
import numpy as np
from normalization import normalize
from hv import Hypervolume

# ---- config ----
folder = "/Users/harshgiri/Documents/NIC GRP PROJECT"

participants = ["ALLAOUI", "HPI"]    # add "NSGA-II" here later if you want
problems_to_run = ["pla33810-n33809"]   # or a list of all problems


# ---- load .f files (same style as in NSGA-II.py) ----
def load_participant_data(folder, problems, participants):
    data = {problem: {} for problem in problems}
    for problem in problems:
        for participant in participants:
            fname = f"{participant}_{problem}.f"
            alt_fname = f"{participant}_{problem.replace('_', '-')}.f"
            if os.path.isfile(os.path.join(folder, fname)):
                path = os.path.join(folder, fname)
            elif os.path.isfile(os.path.join(folder, alt_fname)):
                path = os.path.join(folder, alt_fname)
            else:
                continue
            try:
                F = np.loadtxt(path)
                if F.ndim > 1 and F.shape[1] > 2:
                    F = F[:, :2]
                elif F.ndim == 1:
                    F = F.reshape(-1, 2)
                data[problem][participant] = F
            except Exception as e:
                print(f"Warning: Could not read {path} ({e})")
    return data


if __name__ == "__main__":
    data = load_participant_data(folder, problems_to_run, participants)

    # use normalized space; reference point slightly worse than [1,1]
    ref_point = np.array([1.1, 1.1])
    hv_calc = Hypervolume(ref_point)

    for problem in problems_to_run:
        if not data[problem]:
            continue

        # 1) build ideal / nadir from ALL algorithms for this problem
        fronts = [F for F in data[problem].values()]
        all_points = np.vstack(fronts)
        ideal = all_points.min(axis=0)   # [min_time, min_profit]
        nadir = all_points.max(axis=0)   # [max_time, max_profit]

        print(f"\nProblem: {problem}")
        print("Participant\tHypervolume (normalized)")

        # 2) compute normalized HV for each participant
        for participant, F in data[problem].items():
            F_norm = normalize(F, ideal, nadir)
            hv_value = hv_calc.calc(F_norm)
            print(f"{participant}\t{hv_value:.6f}")
