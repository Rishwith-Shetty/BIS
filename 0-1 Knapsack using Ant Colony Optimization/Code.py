
import random

# Item weights and profits
weights = [2, 3, 4, 5, 9]
profits = [3, 4, 5, 8, 10]

capacity = 10

n = len(weights)

# ACO parameters
ants = 10
iterations = 50
alpha = 1
beta = 2
evaporation = 0.5

# Initialize pheromone
pheromone = [1.0] * n

# Heuristic information
heuristic = [
    profits[i] / weights[i]
    for i in range(n)
]

best_solution = None
best_profit = 0


# Calculate solution profit
def calculate_profit(solution):
    return sum(
        solution[i] * profits[i]
        for i in range(n)
    )


# Calculate solution weight
def calculate_weight(solution):
    return sum(
        solution[i] * weights[i]
        for i in range(n)
    )


# ACO Algorithm
for iteration in range(iterations):

    solutions = []

    for ant in range(ants):

        solution = [0] * n
        total_weight = 0

        # Randomize item selection order
        item_order = list(range(n))
        random.shuffle(item_order)

        for i in item_order:

            if total_weight + weights[i] <= capacity:

                # Calculate selection probability
                desirability = (
                    (pheromone[i] ** alpha)
                    * (heuristic[i] ** beta)
                )

                probability = desirability / (
                    1 + desirability
                )

                if random.random() < probability:
                    solution[i] = 1
                    total_weight += weights[i]

        profit = calculate_profit(solution)

        solutions.append(solution)

        if profit > best_profit:
            best_profit = profit
            best_solution = solution.copy()

    # Pheromone evaporation
    pheromone = [
        p * (1 - evaporation)
        for p in pheromone
    ]

    # Pheromone update
    for solution in solutions:

        profit = calculate_profit(solution)

        for i in range(n):
            if solution[i] == 1:
                pheromone[i] += profit / max(best_profit, 1)


# Display results
print("0/1 KNAPSACK USING ACO")
print("----------------------")

print("Selected Items:")

for i in range(n):
    if best_solution[i] == 1:
        print(
            "Item", i + 1,
            "Weight:", weights[i],
            "Profit:", profits[i]
        )

print("\nTotal Weight:", calculate_weight(best_solution))
print("Total Profit:", best_profit)
print("Knapsack Capacity:", capacity)
