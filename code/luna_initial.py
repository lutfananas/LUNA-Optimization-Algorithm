import numpy as np

# ===== Parameters =====
num_locations = 10    # jumlah lokasi potensial
num_demand = 50       # jumlah node permintaan
pop_size = 20         # populasi awal
max_iter = 10         # iterasi minimal untuk demo

# ===== Synthetic Data =====
np.random.seed(42)
locations = np.random.rand(num_locations, 2) * 10   # koordinat lokasi
demand_nodes = np.random.rand(num_demand, 2) * 10   # koordinat demand

# ===== Initialize Population (Continuous Vector) =====
# Segment A: Prioritas lokasi (0-1), Segment B: Jumlah charger (0-1)
pop = np.random.rand(pop_size, num_locations*2)

# ===== Decoder Sederhana =====
def decode(individual):
    # Segment A -> pilih lokasi terbaik
    seg_A = individual[:num_locations]
    selected_locations = np.argsort(-seg_A)[:5]  # pilih 5 lokasi teratas
    # Segment B -> jumlah charger proporsional
    seg_B = individual[num_locations:]
    chargers = np.round(seg_B[selected_locations]*10).astype(int)
    return selected_locations, chargers

# ===== Objective Function Dummy =====
def objective(selected, chargers):
    cost = np.sum(chargers)*1000
    waiting_time = np.random.rand() * 10
    return cost + waiting_time

# ===== Partial LUNA Iteration (Safe) =====
best_obj = float('inf')
best_sol = None

for it in range(max_iter):
    for ind in pop:
        selected, chargers = decode(ind)
        obj = objective(selected, chargers)
        if obj < best_obj:
            best_obj = obj
            best_sol = (selected, chargers)
    # Partial update: shuffle populasi (tidak menampilkan semua operator)
    pop = pop + 0.05*np.random.randn(*pop.shape)

# ===== Output =====
print("=== LUNA Safe Prototype ===")
print("Best Objective (demo):", best_obj)
print("Selected Locations (demo):", best_sol[0])
print("Chargers per Location (demo):", best_sol[1])
