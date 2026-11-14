
import numpy as np
import matplotlib.pyplot as plt

# Electricity price for 24 hours (Rs/kWh)
prices = np.array([
    8, 8, 7, 7, 7, 8, 10, 12,
    14, 15, 16, 16, 15, 14, 13, 14,
    16, 18, 20, 19, 16, 13, 10, 9
])

# EV charging requirements
battery_demand = 30       # Required energy in kWh
max_power = 7             # Maximum charging power in kW

# EV availability: connected from 10 PM to 7 AM
availability = np.zeros(24)
availability[0:7] = 1
availability[22:24] = 1

# Cuckoo Search parameters
n_nests = 25
pa = 0.25
iterations = 100

# Function to repair solutions and satisfy constraints
def repair(solution):
    solution = np.clip(solution, 0, max_power)
    solution = solution * availability

    # Ensure required energy is supplied
    if solution.sum() < battery_demand:
        for i in range(24):
            if availability[i] == 1:
                extra = min(
                    max_power - solution[i],
                    battery_demand - solution.sum()
                )
                solution[i] += extra

                if solution.sum() >= battery_demand:
                    break

    # Avoid charging more than required
    if solution.sum() > battery_demand:
        solution *= battery_demand / solution.sum()

    return solution

# Fitness function: minimize charging cost
def fitness(solution):
    return np.sum(solution * prices)

# Initialize nests
nests = np.array([
    repair(np.random.uniform(0, max_power, 24))
    for _ in range(n_nests)
])

best_cost_history = []

# Cuckoo Search Algorithm
for iteration in range(iterations):

    # Generate new solutions using Levy-like random steps
    for i in range(n_nests):

        step = np.random.normal(0, 0.9, 24)

        new_solution = repair(nests[i] + step)

        # Select a random nest
        j = np.random.randint(n_nests)

        # Replace if new solution is better
        if fitness(new_solution) < fitness(nests[j]):
            nests[j] = new_solution

    # Abandon worst nests
    costs = np.array([fitness(nest) for nest in nests])

    n_abandon = int(pa * n_nests)
    worst_indices = np.argsort(costs)[-n_abandon:]

    for j in worst_indices:
        nests[j] = repair(
            np.random.uniform(0, max_power, 24)
        )

    # Record best solution
    costs = np.array([fitness(nest) for nest in nests])
    best_cost_history.append(np.min(costs))

# Find optimal charging schedule
best_index = np.argmin([
    fitness(nest) for nest in nests
])

best_solution = nests[best_index]
minimum_cost = fitness(best_solution)

# Display results
print("OPTIMAL EV CHARGING SCHEDULE")
print("--------------------------------")
print("Total Energy Required:", battery_demand, "kWh")
print("Total Energy Charged:", round(best_solution.sum(), 2), "kWh")
print("Minimum Charging Cost: Rs.", round(minimum_cost, 2))

print("\nHourly Charging Schedule:")

for hour in range(24):
    if best_solution[hour] > 0.01:
        print(
            f"{hour:02d}:00 - {hour+1:02d}:00 : "
            f"{best_solution[hour]:.2f} kWh"
        )

# Plot charging schedule
plt.figure(figsize=(12, 5))
plt.bar(range(24), best_solution)
plt.xlabel("Hour of Day")
plt.ylabel("Charging Energy (kWh)")
plt.title("Optimized EV Charging Schedule")
plt.xticks(range(24))
plt.grid(axis='y')
plt.show()

# Plot convergence
plt.figure(figsize=(10, 4))
plt.plot(best_cost_history)
plt.xlabel("Iteration")
plt.ylabel("Charging Cost (Rs.)")
plt.title("Cuckoo Search Convergence")
plt.grid()
plt.show()
