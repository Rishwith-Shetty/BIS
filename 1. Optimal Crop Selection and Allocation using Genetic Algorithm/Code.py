import random

# Crop information
crops = {
    "Rice":   {"profit": 50000, "water": 1200},
    "Wheat":  {"profit": 35000, "water": 450},
    "Maize":  {"profit": 30000, "water": 600},
    "Pulses": {"profit": 25000, "water": 350}
}

LAND = 10       # Total land in hectares
WATER = 5000    # Available water in cubic metres

POP_SIZE = 20
GENERATIONS = 100
MUTATION_RATE = 0.1

crop_names = list(crops.keys())


# Generate random chromosome
def create_chromosome():
    return [random.uniform(0, LAND) for _ in crop_names]


# Check and repair constraints
def repair(chromosome):
    land_used = sum(chromosome)

    water_used = sum(
        chromosome[i] * crops[crop_names[i]]["water"]
        for i in range(len(crop_names))
    )

    scale = min(
        1,
        LAND / land_used if land_used > 0 else 1,
        WATER / water_used if water_used > 0 else 1
    )

    return [x * scale for x in chromosome]


# Calculate fitness (total profit)
def fitness(chromosome):
    profit = sum(
        chromosome[i] * crops[crop_names[i]]["profit"]
        for i in range(len(crop_names))
    )
    return profit


# Selection
def selection(population):
    candidates = random.sample(population, 3)
    return max(candidates, key=fitness)


# Crossover
def crossover(parent1, parent2):
    child = []

    for i in range(len(parent1)):
        if random.random() < 0.5:
            child.append(parent1[i])
        else:
            child.append(parent2[i])

    return repair(child)


# Mutation
def mutation(chromosome):
    for i in range(len(chromosome)):
        if random.random() < MUTATION_RATE:
            chromosome[i] += random.uniform(-2, 2)
            chromosome[i] = max(0, chromosome[i])

    return repair(chromosome)


# Genetic Algorithm
def genetic_algorithm():

    population = [
        repair(create_chromosome())
        for _ in range(POP_SIZE)
    ]

    for generation in range(GENERATIONS):

        population.sort(key=fitness, reverse=True)

        new_population = [population[0]]

        while len(new_population) < POP_SIZE:

            parent1 = selection(population)
            parent2 = selection(population)

            child = crossover(parent1, parent2)
            child = mutation(child)

            new_population.append(child)

        population = new_population

    best = max(population, key=fitness)

    return best


# Main Program
best = genetic_algorithm()

print("OPTIMAL CROP ALLOCATION")
print("-----------------------")

total_profit = 0
total_land = 0
total_water = 0

for i in range(len(crop_names)):

    crop = crop_names[i]
    area = best[i]

    profit = area * crops[crop]["profit"]
    water = area * crops[crop]["water"]

    print(crop, ":", round(area, 2), "hectares")

    total_profit += profit
    total_land += area
    total_water += water

print("\nTotal Land Used:", round(total_land, 2), "hectares")
print("Total Water Used:", round(total_water, 2), "m3")
print("Maximum Profit: Rs.", round(total_profit, 2))
