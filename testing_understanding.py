def random_mutation(chromosome):

    idx     = random.randint(0, NUM_PACKAGES - 1)
    mutated = chromosome[:]
    mutated[idx] = 1 - mutated[idx]   # binary flip
    return mutated


def swap_mutation(chromosome):

    i = random.randint(0, NUM_PACKAGES - 1)
    j = random.randint(0, NUM_PACKAGES - 1)
    while j == i:                         
        j = random.randint(0, NUM_PACKAGES - 1)
    mutated    = chromosome[:]
    mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated


def gaussian_mutation(chromosome, sigma=1.5):

    idx     = random.randint(0, NUM_PACKAGES - 1)
    mutated = chromosome[:]
    noise   = random.gauss(0, sigma)
    raw     = mutated[idx] + noise
    sig     = 1.0 / (1.0 + math.exp(-raw))
    mutated[idx] = 1 if sig >= 0.5 else 0
    return mutated


def apply_mutation(chromosome):

    choice = random.choice(['random', 'swap', 'gaussian'])
    if choice == 'random':
        return random_mutation(chromosome)
    elif choice == 'swap':
        return swap_mutation(chromosome)
    else:
        return gaussian_mutation(chromosome)