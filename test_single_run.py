from models.GenerationSimulation import GenerationSimulation
from utils.global_config import load_global_config
import json

load_global_config('src/calibration/calibration_results/config.json')

with open('src/calibration/calibration_results/config.json', 'r') as f:
    config = json.load(f)

# Используем оптимизированные параметры из вашего последнего запуска
mutation_freqs = {
    'M694V': 0.06,      # было 0.04
    'V726A': 0.02,      # было 0.012
    'M680I': 0.01,      # было 0.006
    'R761H': 0.005,     # было 0.002
    'N': 0.905          # 1 - сумма
}

penetrance = {
    'M694V_homozygous': 0.95,      # было 0.88
    'compound_heterozygous': 0.90,  # было 0.82
    'other_homozygous': 0.90,       # было 0.82
    'heterozygous': 0.06,           # было 0.04
    'default': 0.0
}

sim = GenerationSimulation(
    initial_population_size=1_000_000,
    max_age_limit=49,
    mutation_frequencies=mutation_freqs,
    penetrance_config=penetrance,
    simulation_years=200,
    base_year=1862,
    verbose=True
)

# Явно создаем список годов
years_list = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

print(f"DEBUG: base_year = {sim.base_year}")
print(f"DEBUG: simulation_years = {sim.simulation_years}")
print(f"DEBUG: max_year = {max(years_list)}")
print(f"DEBUG: years_list = {years_list}")

results = sim.run_simulation_with_history(years_list)

print("\n📊 РЕЗУЛЬТАТЫ ПО ГОДАМ:")
print("-" * 50)
for year, data in results.items():
    print(f"{year}: prevalence = {data['prevalence']:.6f}, population = {data['population']}")

print("\n📈 АНАЛИЗ ДИНАМИКИ:")
prev_values = []
years_sorted = sorted(results.keys())
for year in years_sorted:
    prev = results[year]['prevalence']
    prev_values.append(prev)
    print(f"{year}: {prev:.6f}")

if len(prev_values) > 1:
    growth = (prev_values[-1] - prev_values[0]) / prev_values[0] * 100
    print(f"\n📊 Рост prevalence с {years_sorted[0]} по {years_sorted[-1]}: {growth:.1f}%")