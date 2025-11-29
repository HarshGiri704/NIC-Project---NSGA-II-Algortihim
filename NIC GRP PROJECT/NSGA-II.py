import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random

# Reproducibility
random.seed(42)
np.random.seed(42)

'''# Participants and problem
participants = ["ALLAOUI", "HPI"]
problems_to_run = ["pla33810-n33809"]
folder = "/Users/harshgiri/Documents/NIC GRP PROJECT"'''

# Participants and problem
participants = ["ALLAOUI", "HPI", "NSGA-II"]
problems_to_run = ["pla33810-n169045", "pla33810-n338090"]
folder = "/Users/harshgiri/Documents/NIC GRP PROJECT"


# ---------- NSGA-II core ----------

def fast_non_dominated_sort(population):
    S = [[] for _ in range(len(population))]
    front = [[]]
    n = [0 for _ in range(len(population))]
    rank = [0 for _ in range(len(population))]
    for p in range(len(population)):
        S[p] = []
        n[p] = 0
        for q in range(len(population)):
            if dominates(population[p], population[q]):
                S[p].append(q)
            elif dominates(population[q], population[p]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            front[0].append(p)
    i = 0
    while front[i]:
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)
        i += 1
        front.append(Q)
    return front[:-1]

def dominates(ind1, ind2):
    return (ind1[0] <= ind2[0] and ind1[1] <= ind2[1]) and \
           (ind1[0] < ind2[0] or ind1[1] < ind2[1])

def crowding_distance(pop):
    obj = np.array(pop)
    l = len(pop)
    distances = np.zeros(l)
    for m in range(obj.shape[1]):
        idx = np.argsort(obj[:, m])
        distances[idx[0]] = distances[idx[-1]] = float("inf")
        fmax, fmin = obj[idx[-1], m], obj[idx[0], m]
        for i in range(1, l - 1):
            distances[idx[i]] += (obj[idx[i + 1], m] - obj[idx[i - 1], m]) / (fmax - fmin + 1e-9)
    return distances

def tournament_selection(pop, ranks, distances):
    size = len(pop)
    idx1, idx2 = random.randrange(size), random.randrange(size)
    if ranks[idx1] < ranks[idx2]:
        return pop[idx1]
    elif ranks[idx2] < ranks[idx1]:
        return pop[idx2]
    else:
        return pop[idx1] if distances[idx1] > distances[idx2] else pop[idx2]

def arithmetic_crossover(parent1, parent2):
    alpha = np.random.uniform(0.4, 0.6)
    child1 = alpha * np.array(parent1) + (1 - alpha) * np.array(parent2)
    child2 = (1 - alpha) * np.array(parent1) + alpha * np.array(parent2)
    return child1, child2

def mutate(individual, mutation_rate=0.05, mutation_scale=0.002):
    mutant = np.array(individual)
    for i in range(len(mutant)):
        if random.random() < mutation_rate:
            mutant[i] += np.random.normal(0, mutation_scale)
    return mutant

def nsga2_solve(population, generations=5, pop_size=20):
    fronts = fast_non_dominated_sort(population)
    pareto_indices = fronts[0]
    pareto_points = population[pareto_indices]

    if len(pareto_points) >= pop_size:
        pop = [pareto_points[random.randint(0, len(pareto_points) - 1)]
               for _ in range(pop_size)]
    else:
        pop = list(pareto_points)
        while len(pop) < pop_size:
            pop.append(population[random.randint(0, len(population) - 1)])

    for gen in range(generations):
        print(f"    Generation {gen+1}/{generations}")
        ranks = [None] * len(pop)
        fronts = fast_non_dominated_sort(pop)
        for i, front in enumerate(fronts):
            for idx in front:
                ranks[idx] = i
        distances = crowding_distance(pop)

        offspring = []
        while len(offspring) < pop_size:
            p1 = tournament_selection(pop, ranks, distances)
            p2 = tournament_selection(pop, ranks, distances)
            c1, c2 = arithmetic_crossover(p1, p2)
            offspring.append(mutate(c1))
            if len(offspring) < pop_size:
                offspring.append(mutate(c2))

        pop += offspring
        full_fronts = fast_non_dominated_sort(pop)
        new_pop = []
        for front in full_fronts:
            front_individuals = [pop[idx] for idx in front]
            if len(new_pop) + len(front_individuals) <= pop_size:
                new_pop += front_individuals
            else:
                cd = crowding_distance(front_individuals)
                sorted_idx = np.argsort(-cd)
                for idx in sorted_idx[:pop_size - len(new_pop)]:
                    new_pop.append(front_individuals[idx])
                break
        pop = new_pop

    final_fronts = fast_non_dominated_sort(pop)
    raw_front = np.array([pop[idx] for idx in final_fronts[0]])
    unique_points = []
    for pt in raw_front:
        if not any(np.allclose(pt, up, atol=1e-5) for up in unique_points):
            unique_points.append(pt)
    return np.array(unique_points)

# ---------- Step 1: run NSGA-II and save ----------

def run_nsga2_and_save(folder, problems):
    for problem in problems:
        print(f"\nRunning NSGA-II for problem: {problem}")
        txt_file = os.path.join(folder, f"{problem}.txt")
        if not os.path.isfile(txt_file):
            print(f"  Data file not found: {txt_file}")
            continue

        numeric_rows = []
        with open(txt_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                try:
                    t, p = float(parts[0]), float(parts[1])
                    numeric_rows.append([t, p])
                except ValueError:
                    continue

        if not numeric_rows:
            print(f"  No numeric data in {txt_file}")
            continue

        population = np.array(numeric_rows)
        # IMPORTANT: keep small values here
        nd_sol = nsga2_solve(population, generations=5, pop_size=20)
        out_name = os.path.join(folder, f"NSGA-II_{problem}.f")
        np.savetxt(out_name, nd_sol, fmt="%.4f")
        print(f"  NSGA-II Pareto front saved to {out_name}")

# ---------- Step 2: load and plot remain unchanged ----------

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

def plot_all_participants(data, problems, participants):
    cmap = cm.get_cmap("tab20", len(participants))
    for problem in problems:
        if not data[problem]:
            print(f"No data to plot for {problem}")
            continue
        plt.figure(figsize=(8, 6))
        for i, participant in enumerate(participants):
            if participant in data[problem]:
                F = data[problem][participant]
                plt.scatter(
                    F[:, 0], F[:, 1],
                    label=participant,
                    s=22,
                    edgecolors=cmap(i),
                    facecolors="none",
                    alpha=0.70,
                    linewidths=1.2
                )
        plt.xlabel("time")
        plt.ylabel("negative profit")
        plt.title(f"{problem} - All Participants")
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=9)
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.xscale("log")
        plt.yscale("log")
        out_png = f"{problem}_multi_participant.png"
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        print(f"  Plot saved to {out_png}")
        plt.show()

'''if __name__ == "__main__":
    run_nsga2_and_save(folder, problems_to_run)
    data = load_participant_data(folder, problems_to_run, participants)
    plot_all_participants(data, problems_to_run, participants)'''
 
if __name__ == "__main__":
    data = load_participant_data(folder, problems_to_run, participants)
    plot_all_participants(data, problems_to_run, participants)
    
