"""
Simple Delivery Allocation (Load Balancing)
AI Course Project — Comparative & Hybrid AI (Search vs. Genetic Algorithm)

Problem:
  A company has 6 delivery packages and 2 vehicles.
  Distribute packages between vehicles so workload is balanced.

Approach:
  1. Build a 2D grid with walls, a Depot, and 6 Delivery Points.
  2. BFS: find the shortest path cost from Depot to each Delivery Point.
  3. GA:  evolve a binary chromosome [0/1 per package] to minimise
          |dist_vehicle1 - dist_vehicle2|.
  4. Visualise results with matplotlib.

Mutation types implemented (from lecture):
  - Random Mutation   : flip a randomly chosen single gene
  - Swap Mutation     : swap two randomly chosen genes (Inorder Mutation)
  - Gaussian Mutation : add Gaussian noise then threshold back to binary

Termination conditions implemented (from lecture):
  - Max Iterations : stop after NUM_GENERATIONS generations
  - Convergence    : stop if best fitness does not improve for PATIENCE gens
  - Target Found   : stop immediately if fitness reaches TARGET_FITNESS
"""

import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import deque

# =============================================================================
# SECTION 1 — ENVIRONMENT SETUP
# =============================================================================

# Grid: 0 = walkable cell, 1 = wall (obstacle)
GRID = np.array([
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1],
    [0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0],
    [0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0],
    [0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
], dtype=int)

ROWS, COLS = GRID.shape

DEPOT = (0, 0)
DELIVERY_POINTS = [
    (2,  4),   # P1
    (5,  13),  # P2
    (7,  11),  # P3
    (10, 3),   # P4
    (12, 5),   # P5
    (14, 12),  # P6
]
NUM_PACKAGES = len(DELIVERY_POINTS)

# =============================================================================
# SECTION 2 — BFS SEARCH
#
# WHY BFS AND NOT DFS?
# ─────────────────────────────────────────────────────────────────────────────
# BFS (Breadth-First Search):
#   - Uses a FIFO queue: expands all nodes at distance d before d+1.
#   - Guaranteed to find the SHORTEST path (optimal) in unweighted grids.
#   - Complete: will always find a solution if one exists.
#   - Time complexity: O(rows × cols). Space: O(rows × cols) for the queue.
#
# DFS (Depth-First Search):
#   - Uses a LIFO stack: dives deep before backtracking.
#   - NOT optimal: may find a path of length 50 when length 10 exists.
#   - NOT complete in infinite/cyclic spaces (can loop forever).
#   - Better for: exhaustive path exploration, detecting cycles.
#
# For this project we need the EXACT shortest distance from Depot to each
# delivery point, so BFS is the correct and only appropriate choice.
# =============================================================================

def bfs(grid, start, goal):
    """
    Breadth-First Search on a 2D grid.

    Parameters:
        grid  : 2D numpy array (0 = walkable, 1 = wall)
        start : (row, col) starting cell
        goal  : (row, col) target cell

    Returns:
        (cost, path) where cost = number of steps, path = list of (row,col).
        Returns (inf, []) if goal is unreachable.
    """
    queue   = deque([(start, [start])])
    visited = {start}

    while queue:
        (r, c), path = queue.popleft()   # FIFO: closest nodes first

        if (r, c) == goal:
            return len(path) - 1, path   # steps = cells visited - 1 (start is free)

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:   # 4-connected
            nr, nc = r + dr, c + dc
            if (0 <= nr < ROWS and 0 <= nc < COLS
                    and grid[nr, nc] == 0
                    and (nr, nc) not in visited):
                visited.add((nr, nc))
                queue.append(((nr, nc), path + [(nr, nc)]))

    return float('inf'), []


def compute_all_distances(grid, depot, delivery_points):
    """Run BFS from depot to each delivery point. Returns (distances, paths)."""
    distances, paths = [], []
    for point in delivery_points:
        cost, path = bfs(grid, depot, point)
        distances.append(cost)
        paths.append(path)
    return distances, paths


# =============================================================================
# SECTION 3 — GENETIC ALGORITHM
# =============================================================================

# ── Parameters ────────────────────────────────────────────────────────────────
POPULATION_SIZE = 60
MUTATION_RATE   = 0.30    # probability that a chromosome undergoes mutation

# ── Termination parameters (all three from the lecture) ───────────────────────
NUM_GENERATIONS = 400     # Termination 1 — Max Iterations (hard upper limit)
PATIENCE        = 80      # Termination 2 — Convergence: gens without improvement
CONVERGENCE_TOL = 1e-6    # minimum fitness delta counted as "improvement"
TARGET_FITNESS  = 1.0     # Termination 3 — Target Found: perfect balance

# ── Selection parameter ───────────────────────────────────────────────────────
TOURNAMENT_SIZE = 5

# =============================================================================
# FITNESS FUNCTION
#
# WHY THIS FORMULA?
# ─────────────────────────────────────────────────────────────────────────────
# Objective : minimise the imbalance |dist_V1 - dist_V2|.
# GAs maximise, so we invert: fitness = 1 / (1 + |diff|)
#   - 2|diff| = 0  →  fitness = 1.0  (perfect balance, maximum score)
#   - |diff| → ∞  →  fitness → 0    (extreme imbalance, minimum score)
#   - Always positive, always in (0, 1] — simple to compare, simple to select.
# Alternative (fitness = -|diff|) is valid but produces negative scores, which
# complicates roulette-wheel selection. The 1/(1+x) transform is preferred.
# =============================================================================

def fitness(chromosome, distances):
    """
    Compute fitness of a chromosome.

    chromosome : list of 0s and 1s (0 = Vehicle 1, 1 = Vehicle 2)
    distances  : list of BFS costs for each package

    Returns float in (0, 1].
    """
    dist_v1 = sum(distances[i] for i in range(NUM_PACKAGES) if chromosome[i] == 0)
    dist_v2 = sum(distances[i] for i in range(NUM_PACKAGES) if chromosome[i] == 1)
    return 1.0 / (1.0 + abs(dist_v1 - dist_v2))


# =============================================================================
# SELECTION — Tournament Selection
#
# WHY TOURNAMENT AND NOT ROULETTE WHEEL?
# ─────────────────────────────────────────────────────────────────────────────
# Roulette Wheel: probability of selection = fitness_i / sum(all_fitness).
#   Problem: if one chromosome has fitness 0.9 and all others 0.01, it
#   dominates the wheel and diversity collapses (premature convergence).
#   Also sensitive to fitness scale — adding a constant to all scores changes
#   selection probabilities.
#
# Tournament (k=5): pick 5 random chromosomes, select the fittest.
#   Advantages:
#     - Selection pressure controlled by k (not by raw fitness values).
#     - Immune to scale/translation of fitness function.
#     - For our 64-chromosome space, k=5 gives strong-enough pressure to
#       converge while keeping enough diversity to avoid local optima.
# =============================================================================

def tournament_selection(population, distances):
    """Pick TOURNAMENT_SIZE random candidates; return the fittest."""
    candidates = random.sample(population, TOURNAMENT_SIZE)
    return max(candidates, key=lambda c: fitness(c, distances))


# =============================================================================
# CROSSOVER — One-Point Crossover
#
# WHY ONE-POINT AND NOT TWO-POINT OR UNIFORM?
# ─────────────────────────────────────────────────────────────────────────────
# Our chromosome has only 6 genes (2^6 = 64 possible solutions).
# One-point crossover splits the chromosome at a random point and swaps
# the right halves. It preserves contiguous gene blocks ("building blocks")
# which is important when nearby genes might interact.
#
# Two-point crossover: better for longer chromosomes (20+ genes) where
# disrupting building blocks is acceptable in exchange for more mixing.
# Uniform crossover: every gene chosen independently — destroys all building
# blocks, too destructive for a 6-gene chromosome.
#
# Conclusion: one-point is the correct balance of simplicity and effectiveness
# for a 6-bit binary chromosome.
# =============================================================================

def one_point_crossover(parent1, parent2):
    """
    One-point crossover at a random cut point cp in [1, 5].
    offspring1 = parent1[:cp] + parent2[cp:]
    offspring2 = parent2[:cp] + parent1[cp:]
    """
    cp = random.randint(1, NUM_PACKAGES - 1)
    return (parent1[:cp] + parent2[cp:],
            parent2[:cp] + parent1[cp:])


# =============================================================================
# MUTATION — Three Types (all from the lecture)
#
# ROLE OF MUTATION:
#   Crossover recombines existing genes but cannot introduce gene values that
#   were absent from the initial population. Mutation is the escape mechanism:
#   it flips, swaps, or perturbs genes to explore entirely new solutions and
#   prevent premature convergence to a local optimum.
# =============================================================================

def random_mutation(chromosome):
    """
    Random Mutation (Lecture, Slide 4).

    Selects ONE random gene index and flips its value (0→1 or 1→0).
    This is the simplest mutation and the most natural for binary chromosomes.

    WHY CHOSEN AS PRIMARY for this project:
      Each gene has exactly two valid values (0 or 1). Replacing with a
      "random value within allowed limits" means flipping. One flip = one
      package reassigned between vehicles, which is the right granularity
      for a 6-gene binary chromosome.

    Lecture pseudocode:
      index     = random integer from 0 to len-1
      new_value = random value in allowed limits  →  1 - chromosome[index]
      chromosome[index] = new_value
    """
    idx     = random.randint(0, NUM_PACKAGES - 1)
    mutated = chromosome[:]
    mutated[idx] = 1 - mutated[idx]   # binary flip
    return mutated


def swap_mutation(chromosome):
    """
    Inorder Mutation / Swap Mutation (Lecture, Slide 5).

    Selects TWO random gene indices and swaps their values.
    Designed for permutation chromosomes (e.g. TSP route order).

    Applied here for diversity: if the two selected genes differ (one 0,
    one 1), the result is equivalent to reassigning one package from each
    vehicle, producing a different valid allocation without randomising the
    whole chromosome. If both genes are equal, the swap is neutral (no change),
    which is acceptable — it simply wastes one mutation event.

    Lecture pseudocode:
      i = random index; j = random index
      swap chromosome[i] and chromosome[j]
    """
    i = random.randint(0, NUM_PACKAGES - 1)
    j = random.randint(0, NUM_PACKAGES - 1)
    while j == i:                         
        j = random.randint(0, NUM_PACKAGES - 1)
    mutated    = chromosome[:]
    mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated


def gaussian_mutation(chromosome, sigma=1.5):
    """
    Gaussian Mutation (Lecture, Slide 6).

    Designed for real-valued chromosomes: adds Gaussian noise (mean=0,
    std=sigma) to a selected gene, causing a smooth small adjustment.

    Adapted for binary chromosomes:
      1. Pick a random gene index.
      2. Add Gaussian noise to the gene's float value.
      3. Apply sigmoid to map the result into (0, 1).
      4. Round to nearest integer → 0 or 1.

    Effect: genes far from 0.5 (i.e. clearly 0 or clearly 1) are harder
    to flip; genes near 0.5 after the sigmoid are more likely to flip.
    With sigma=1.5, the noise is strong enough to regularly cross the 0.5
    threshold, making this effectively a probabilistic bit-flip.

    Lecture pseudocode (adapted):
      index             = random integer
      noise             = Gaussian(mean=0, std=sigma)
      raw               = chromosome[index] + noise
      chromosome[index] = round(sigmoid(raw))   → 0 or 1
    """
    idx     = random.randint(0, NUM_PACKAGES - 1)
    mutated = chromosome[:]
    noise   = random.gauss(0, sigma)
    raw     = mutated[idx] + noise
    sig     = 1.0 / (1.0 + math.exp(-raw))
    mutated[idx] = 1 if sig >= 0.5 else 0
    return mutated


def apply_mutation(chromosome):
    """
    Choose one mutation type uniformly at random and apply it.

    All three types from the lecture are covered each generation:
    random mutation (primary for binary), swap mutation (structural),
    and Gaussian mutation (smooth probabilistic flip).
    """
    choice = random.choice(['random', 'swap', 'gaussian'])
    if choice == 'random':
        return random_mutation(chromosome)
    elif choice == 'swap':
        return swap_mutation(chromosome)
    else:
        return gaussian_mutation(chromosome)


# =============================================================================
# TERMINATION — Three Conditions (Lecture, Slide 8)
#
# The GA must know WHEN TO STOP. Running forever wastes compute; stopping too
# early gives a suboptimal solution. Three conditions from the lecture:
#
# 1. Max Iterations  (always active as a safety net)
#      Stop after a fixed number of generations regardless of quality.
#      Guarantees the program always terminates. Controlled by NUM_GENERATIONS.
#
# 2. Convergence     (smart early stopping)
#      Stop when the best fitness has not improved for PATIENCE consecutive
#      generations. This detects when the population has stabilised and further
#      evolution is unlikely to help. Saves computation.
#
# 3. Target Found    (ideal-solution detection)
#      Stop immediately when a chromosome with fitness >= TARGET_FITNESS is
#      found. For this project TARGET=1.0 (perfect balance). If the optimal
#      solution exists and is found early, there is no reason to continue.
# =============================================================================

def check_termination(generation, best_fitness, no_improve_count):
    """
    Evaluate all three termination conditions.

    Returns (should_stop: bool, reason: str).
    Max Iterations is checked by the for-loop itself; this function handles
    Convergence and Target Found.
    """
    # Termination 3 — Target Found
    if best_fitness >= TARGET_FITNESS:
        return True, f"Target Found: perfect fitness {TARGET_FITNESS:.4f} reached at generation {generation + 1}"

    # Termination 2 — Convergence
    if no_improve_count >= PATIENCE:
        return True, (f"Convergence: no improvement for {PATIENCE} consecutive "
                      f"generations (stopped at generation {generation + 1})")

    return False, ""


# =============================================================================
# MAIN GA LOOP
# =============================================================================

def run_genetic_algorithm(distances):
    """
    Full Genetic Algorithm with all three termination conditions.

    Per-generation workflow:
      1. Evaluate fitness for every chromosome.
      2. Update global best; track convergence counter.
      3. Check Termination 2 (Convergence) and 3 (Target Found).
      4. Build next generation:
           a. Elitism — copy best chromosome unchanged.
           b. Tournament selection of two parents.
           c. One-point crossover → two offspring.
           d. Mutation (random/swap/gaussian) with probability MUTATION_RATE.
      5. Repeat until a termination condition fires.
      6. Termination 1 (Max Iterations) is the for-loop bound.

    Returns:
        best_chromosome    : list[int]  — best allocation found
        best_fitness       : float
        history            : list[float] — best fitness per generation
        termination_reason : str        — which condition fired and when
    """
    population           = [random_chromosome() for _ in range(POPULATION_SIZE)]
    history              = []
    best_chromosome      = None
    best_fitness_overall = -1.0
    no_improve_count     = 0
    term_reason          = f"Max Iterations: all {NUM_GENERATIONS} generations completed"

    for generation in range(NUM_GENERATIONS):   # Termination 1 — Max Iterations

        # Evaluate
        scores      = [fitness(c, distances) for c in population]
        best_idx    = scores.index(max(scores))
        gen_best    = scores[best_idx]
        history.append(gen_best)

        # Update global best & convergence counter
        if gen_best > best_fitness_overall + CONVERGENCE_TOL:
            best_fitness_overall = gen_best
            best_chromosome      = population[best_idx][:]
            no_improve_count     = 0
        else:
            no_improve_count += 1

        # Check Termination 2 & 3
        stop, reason = check_termination(generation, best_fitness_overall,
                                         no_improve_count)
        if stop:
            term_reason = reason
            break

        # Build next generation
        next_pop = [population[best_idx][:]]   # Elitism

        while len(next_pop) < POPULATION_SIZE:
            p1 = tournament_selection(population, distances)
            p2 = tournament_selection(population, distances)
            c1, c2 = one_point_crossover(p1, p2)

            if random.random() < MUTATION_RATE:
                c1 = apply_mutation(c1)
            if random.random() < MUTATION_RATE:
                c2 = apply_mutation(c2)

            next_pop.append(c1)
            if len(next_pop) < POPULATION_SIZE:
                next_pop.append(c2)

        population = next_pop

    return best_chromosome, best_fitness_overall, history, term_reason


def random_chromosome():
    return [random.randint(0, 1) for _ in range(NUM_PACKAGES)]


# =============================================================================
# SECTION 4 — VISUALISATION
#
# WHAT THE FIGURE SHOWS AND WHY IT IS USEFUL:
#
# LEFT PANEL — Delivery Allocation Map
#   A bird's-eye view of the 15×15 grid.
#   Dark cells = walls (obstacles BFS must navigate around).
#   Blue square (D) = Depot — where both vehicles start.
#   Green circles (V1) = packages assigned to Vehicle 1 by the GA.
#   Orange circles (V2) = packages assigned to Vehicle 2 by the GA.
#   Each circle shows the package label (P1–P6) and its BFS distance number.
#   Bottom-left box: final imbalance (steps difference) and fitness score.
#
#   INSIGHT FROM THIS PANEL:
#     - Visually confirms no two packages with very different distances ended
#       up on the same vehicle (which would cause imbalance).
#     - The wall pattern shows WHY straight-line distance would be wrong —
#       BFS correctly navigates around obstacles.
#
# RIGHT PANEL — GA Fitness Convergence Curve
#   X-axis = generation number. Y-axis = best fitness in that generation.
#   Purple area = fitness over time. Red dashed line = generation where best
#   solution was first found. Bottom text = termination reason.
#
#   INSIGHT FROM THIS PANEL:
#     - Steep early rise: GA quickly finds rough balance.
#     - Plateau: GA has converged — further evolution is not improving things.
#     - The termination annotation tells you WHICH condition fired and WHEN.
#     - Together the two panels prove: the GA improved purposefully (not randomly)
#       and found a near-perfect allocation.
# =============================================================================

def plot_results(grid, depot, delivery_points, best_chromosome,
                 distances, history, termination_reason):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("Simple Delivery Allocation — Load Balancing via GA",
                 fontsize=15, fontweight='bold', y=0.98)

    # ── Left: map ─────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_title("Delivery Allocation Map", fontsize=13, fontweight='bold')
    ax.imshow(np.where(grid == 1, 0.2, 0.95), cmap='gray',
              vmin=0, vmax=1, origin='upper')

    for x in range(COLS + 1):
        ax.axvline(x - 0.5, color='#cccccc', linewidth=0.4)
    for y in range(ROWS + 1):
        ax.axhline(y - 0.5, color='#cccccc', linewidth=0.4)

    for r in range(ROWS):
        for c in range(COLS):
            if grid[r, c] == 1:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                             facecolor='#3d3d3d', edgecolor='#222', linewidth=0.3))

    dr, dc = depot
    ax.plot(dc, dr, 's', color='#1a1aff', markersize=18, zorder=5)
    ax.text(dc, dr, 'D', color='white', ha='center', va='center',
            fontsize=9, fontweight='bold', zorder=6)

    v1_total = v2_total = 0
    for i, (pr, pc) in enumerate(delivery_points):
        dist = distances[i]
        if best_chromosome[i] == 0:
            col, ecol, v1_total = '#2ecc71', '#1a7a44', v1_total + dist
        else:
            col, ecol, v2_total = '#e67e22', '#8b4a00', v2_total + dist
        ax.plot(pc, pr, 'o', color=col, markersize=18, zorder=5,
                markeredgecolor=ecol, markeredgewidth=1)
        ax.text(pc, pr, f'P{i+1}', color='white', ha='center', va='center',
                fontsize=8, fontweight='bold', zorder=6)
        ax.text(pc + 0.5, pr - 0.5, str(dist), color=ecol,
                fontsize=7, fontweight='bold', zorder=7)

    ax.legend(handles=[
        mpatches.Patch(color='#1a1aff', label='Depot (start)'),
        mpatches.Patch(color='#2ecc71', label=f'Vehicle 1  ({v1_total} steps)'),
        mpatches.Patch(color='#e67e22', label=f'Vehicle 2  ({v2_total} steps)'),
        mpatches.Patch(color='#3d3d3d', label='Wall'),
    ], loc='upper right', fontsize=8.5)

    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(ROWS - 0.5, -0.5)
    ax.set_xticks(range(COLS)); ax.set_yticks(range(ROWS))
    ax.set_xticklabels(range(COLS), fontsize=7)
    ax.set_yticklabels(range(ROWS), fontsize=7)
    ax.set_xlabel("Column", fontsize=10); ax.set_ylabel("Row", fontsize=10)
    imbalance = abs(v1_total - v2_total)
    ax.text(0.01, 0.01,
            f'Imbalance: {imbalance} steps\nFitness: {1/(1+imbalance):.4f}',
            transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#fffbe6',
                      edgecolor='#ccaa00', alpha=0.9))

    # ── Right: fitness curve ───────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_title("GA Fitness Convergence", fontsize=13, fontweight='bold')
    gens = list(range(1, len(history) + 1))
    ax2.plot(gens, history, color='#9b59b6', linewidth=1.5, alpha=0.8)
    ax2.fill_between(gens, history, alpha=0.15, color='#9b59b6')
    bg = history.index(max(history)) + 1
    ax2.axvline(bg, color='#e74c3c', linestyle='--', linewidth=1, alpha=0.7)
    ax2.scatter([bg], [max(history)], color='#e74c3c', zorder=5, s=60)
    ax2.text(bg + 2, max(history) - 0.02,
             f'Best: {max(history):.4f}\n@ gen {bg}', fontsize=9, color='#e74c3c')
    ax2.text(0.02, 0.05, f'Stopped: {termination_reason}',
             transform=ax2.transAxes, fontsize=7.5, color='#555',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0', alpha=0.8))
    ax2.set_xlabel("Generation", fontsize=11)
    ax2.set_ylabel("Best Fitness  [ 1 / (1 + |V1 − V2|) ]", fontsize=11)
    ax2.set_xlim(0, max(len(history) + 10, 50))
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_facecolor('#fafafa')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'delivery_allocation_results.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Figure saved: {out}")
    plt.show()


# =============================================================================
# SECTION 5 — MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("  DELIVERY ALLOCATION — LOAD BALANCING")
    print("  BFS + Genetic Algorithm (Mutation & Termination)")
    print("=" * 65)

    # BFS
    print("\n[1] BFS: computing shortest distances...\n")
    distances, _ = compute_all_distances(GRID, DEPOT, DELIVERY_POINTS)
    print("  Package   Point       BFS Distance")
    print("  " + "-" * 38)
    for i, (pt, d) in enumerate(zip(DELIVERY_POINTS, distances)):
        print(f"  P{i+1}        {str(pt):<12} {d} steps")
    print(f"\n  Total combined: {sum(distances)} steps")

    # GA
    print("\n[2] Genetic Algorithm...\n")
    print(f"  Population      : {POPULATION_SIZE}")
    print(f"  [Term 1] Max gen: {NUM_GENERATIONS}")
    print(f"  [Term 2] Patience: {PATIENCE} gens without improvement")
    print(f"  [Term 3] Target : fitness >= {TARGET_FITNESS}")
    print(f"  Mutation rate   : {MUTATION_RATE}  (random / swap / gaussian)")
    print(f"  Selection       : Tournament k={TOURNAMENT_SIZE}")
    print(f"  Crossover       : One-point\n")

    best, bf, history, reason = run_genetic_algorithm(distances)

    # Results
    print("[3] Results\n")
    print(f"  Termination      : {reason}")
    print(f"  Generations run  : {len(history)}")
    print(f"  Best chromosome  : {best}")
    print(f"  Best fitness     : {bf:.6f}")

    v1 = [i for i in range(NUM_PACKAGES) if best[i] == 0]
    v2 = [i for i in range(NUM_PACKAGES) if best[i] == 1]
    d1 = sum(distances[i] for i in v1)
    d2 = sum(distances[i] for i in v2)
    print(f"\n  Vehicle 1 → {[f'P{i+1}' for i in v1]}  total: {d1} steps")
    print(f"  Vehicle 2 → {[f'P{i+1}' for i in v2]}  total: {d2} steps")
    print(f"  Imbalance        : {abs(d1-d2)} steps")

    # Fitness trace
    print("\n[4] Manual fitness trace")
    print(f"  dist_V1 = {' + '.join(str(distances[i]) for i in v1)} = {d1}")
    print(f"  dist_V2 = {' + '.join(str(distances[i]) for i in v2)} = {d2}")
    print(f"  fitness = 1/(1+|{d1}-{d2}|) = 1/(1+{abs(d1-d2)}) = {bf:.6f}")

    # Mutation demo
    print("\n[5] Mutation type demonstration on best chromosome")
    print(f"  Original          : {best}")
    print(f"  random_mutation   : {random_mutation(best)}")
    print(f"  swap_mutation     : {swap_mutation(best)}")
    print(f"  gaussian_mutation : {gaussian_mutation(best)}")

    # Comparison
    print("\n[6] BFS vs GA / Naive vs GA")
    nv1 = sum(distances[:3]); nv2 = sum(distances[3:])
    impr = (1 - abs(d1-d2)/max(abs(nv1-nv2),1)) * 100
    print(f"  Naive (P1-P3|P4-P6): V1={nv1}, V2={nv2}, imbalance={abs(nv1-nv2)}")
    print(f"  GA solution        : V1={d1},  V2={d2},  imbalance={abs(d1-d2)}")
    print(f"  Improvement        : {impr:.0f}%")

    # Visualise
    print("\n[7] Generating visualisation...")
    plot_results(GRID, DEPOT, DELIVERY_POINTS, best, distances, history, reason)
    print("\nDone.")


if __name__ == "__main__":
    main()
