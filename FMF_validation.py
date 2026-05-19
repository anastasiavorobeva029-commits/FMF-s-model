import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import json
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
import random
import uuid
from tqdm import tqdm
from scipy import stats
from joblib import Parallel, delayed
import sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial, lru_cache
import warnings
import hashlib
import heapq

with open('config.json', 'r') as f:
    CONFIG = json.load(f)

SIM_PARAMS = CONFIG['simulation_params']
MUTATION_FREQS = CONFIG['mutation_frequencies']
PENETRANCE = CONFIG['penetrance']
SEVERITY_WEIGHTS = CONFIG['severity_weights']
ONSET_AGE_DIST = CONFIG['onset_age_dist']

# Глобальный кэш для результатов симуляции
SIMULATION_CACHE = {}
CACHE_HITS = 0
CACHE_MISSES = 0
MAX_CACHE_SIZE = 5000


class Agent:
    def __init__(self, gender: str, age: int,
                 generation: int,
                 max_age_limit: int = 49,
                 ethnicity: str = 'Armenian',
                 father_id: Optional[str] = None,
                 mother_id: Optional[str] = None):

        self.id = str(uuid.uuid4())[:8]
        self.gender = gender
        self.age = age
        self.max_age_limit = max_age_limit if max_age_limit is not None else SIM_PARAMS['max_age_limit']
        self.generation = generation
        self.ethnicity = ethnicity
        self.alive = True

        # Медицинский статус
        self.clinical_status = 'asymptomatic'
        self.disease_severity = None
        self.age_of_onset = None
        self.on_colchicine = False

        # Семейные связи
        self.father_id = father_id
        self.mother_id = mother_id
        self.partner_id = None
        self.children_ids = []

        # Скрининг
        self.is_screened = False
        self.received_counseling = False

        # Генетика
        self.mefv_allele_1 = None
        self.mefv_allele_2 = None
        self.genotype_status = None
        self.mutation_type = None
        self.penetrance = 0.0
        self.will_develop_symptoms = False

    @staticmethod
    @lru_cache(maxsize=1024)
    def _cached_genotype_status(allele_1: str, allele_2: str):
        """Кэшированное определение статуса генотипа"""
        alleles = [allele_1, allele_2]
        mutant_count = sum(1 for allele in alleles if allele != "N")

        if mutant_count == 0:
            return "healthy", None
        elif mutant_count == 1:
            return "carrier", "heterozygous"
        else:
            if len(set(alleles)) == 1:
                allele_type = alleles[0]
                if allele_type == "M694V":
                    return "affected", "M694V_homozygous"
                else:
                    return "affected", "other_homozygous"
            else:
                return "affected", "compound_heterozygous"

    def set_genotype(self, allele_1: str, allele_2: str):
        valid_values = ['N', 'M694V', "V726A", "M680I", "R761H"]
        if allele_1 not in valid_values or allele_2 not in valid_values:
            raise ValueError(f"Invalid allele values. Valid values: {valid_values}")

        self.mefv_allele_1 = allele_1
        self.mefv_allele_2 = allele_2

        # Используем кэшированный метод
        self.genotype_status, self.mutation_type = self._cached_genotype_status(allele_1, allele_2)
        self._determine_lifetime_risk()

    def _determine_lifetime_risk(self):
        if self.genotype_status not in ['affected', 'carrier']:
            self.will_develop_symptoms = False
            self.age_of_onset = None
            return

        self.penetrance = PENETRANCE.get(self.mutation_type, PENETRANCE['default'])
        self.will_develop_symptoms = random.random() < self.penetrance

        if self.will_develop_symptoms:
            self.age_of_onset = self._generate_age_of_onset()
        else:
            self.age_of_onset = None
            self.clinical_status = 'never_symptomatic'

    def _generate_age_of_onset(self):
        if self.mutation_type == "M694V_homozygous":
            age_groups = ONSET_AGE_DIST['M694V_homozygous']
        elif self.mutation_type in ["compound_heterozygous", "other_homozygous"]:
            age_groups = ONSET_AGE_DIST['compound_or_other_homo']
        elif self.mutation_type == "heterozygous":
            age_groups = ONSET_AGE_DIST['heterozygous']
        else:
            return None

        group_probs = [g[2] for g in age_groups]
        total = sum(group_probs)
        group_probs = [p / total for p in group_probs]

        selected_group = random.choices(age_groups, weights=group_probs, k=1)[0]
        min_age, max_age, _ = selected_group
        return random.randint(min_age, max_age)

    def age_year(self):
        if not self.alive:
            return

        self.age += 1
        if self.age >= self.max_age_limit:
            self.alive = False
            return

        if (self.will_develop_symptoms and
                self.clinical_status == 'asymptomatic' and
                self.age_of_onset is not None and
                self.age >= self.age_of_onset):

            if self.mutation_type == "heterozygous":
                if random.random() < 0.7:
                    self.clinical_status = 'symptomatic'
                    self._determine_disease_severity()
            else:
                self.clinical_status = 'symptomatic'
                self._determine_disease_severity()

        if (self.clinical_status == 'symptomatic' and
                not self.on_colchicine):

            base_prob = 0.02

            if self.age <= 15:
                base_prob *= 2

            if self.disease_severity == 'severe':
                base_prob *= 3.0
            elif self.disease_severity == 'moderate':
                base_prob *= 2.0

            base_prob = min(base_prob, 1.0)

            if random.random() < base_prob:
                self.on_colchicine = True

    def _determine_disease_severity(self):
        severity_weights = SEVERITY_WEIGHTS.get(self.mutation_type, {"mild": 1.0, "moderate": 0.0, "severe": 0.0})
        labels = list(severity_weights.keys())
        weights = list(severity_weights.values())
        self.disease_severity = random.choices(labels, weights=weights, k=1)[0]

    def set_partner(self, partner: 'Agent'):
        self.partner_id = partner.id
        partner.partner_id = self.id

    def can_have_children(self):
        return (self.alive and
                18 <= self.age <= 45 and
                self.partner_id is not None)


class GenerationSimulation:
    def __init__(self,
                 initial_population_size: int = 10000,
                 max_age_limit: int = 49,
                 mutation_frequencies: Dict[str, float] = None,
                 simulation_years: int = 150,
                 validation_year: int = 2012,
                 base_year: int = 1862,
                 verbose: bool = False,
                 use_cache: bool = True):

        self.initial_population_size = initial_population_size if (initial_population_size
                                                                   is not None) else SIM_PARAMS[
            'initial_population_size']
        self.max_age_limit = max_age_limit if (max_age_limit
                                               is not None) else SIM_PARAMS['max_age_limit']
        self.simulation_years = simulation_years if (simulation_years
                                                     is not None) else SIM_PARAMS['simulation_years']
        self.validation_year = validation_year
        self.base_year = base_year if base_year is not None else SIM_PARAMS['base_year']

        self.mutation_frequencies = mutation_frequencies if (mutation_frequencies
                                                             is not None) else MUTATION_FREQS.copy()

        self.agents: Dict[str, Agent] = {}
        self.year = 0
        self.population_history = []
        self.children_born = 0

        # Валидационные данные
        self.validation_point_reached = False
        self.validation_data = None

        self.verbose = verbose
        self.use_cache = use_cache

        # Индексы для быстрого доступа
        self._male_indices = []
        self._female_indices = []
        self._alive_indices = []
        self._partnered_indices = set()

        # Предвычисленные вероятности
        self._birth_probs = self._precompute_birth_probs()

    def _precompute_birth_probs(self):
        """Предвычисление вероятностей рождения"""
        probs = {}
        for age in range(18, 46):
            if age < 20:
                probs[age] = 0.07
            elif age < 25:
                probs[age] = 0.09
            elif age < 30:
                probs[age] = 0.10
            elif age < 35:
                probs[age] = 0.08
            elif age < 40:
                probs[age] = 0.05
            else:
                probs[age] = 0.02
        return probs

    def _get_cache_key(self):
        """Создает ключ для кэширования"""
        # Используем только важные параметры для ключа
        key_parts = [
            self.initial_population_size,
            self.max_age_limit,
            self.simulation_years,
            self.validation_year,
            self.base_year,
            tuple(sorted(self.mutation_frequencies.items()))
        ]
        return hashlib.md5(str(key_parts).encode()).hexdigest()

    def initialize_founders(self):
        """Оптимизированная инициализация основателей"""
        ages = [18, 22, 25, 28, 30, 32, 35, 38, 40, 42]
        age_weights = [0.05, 0.10, 0.15, 0.15, 0.15, 0.10, 0.10, 0.08, 0.07, 0.05]

        # Генерируем все возраста сразу
        selected_ages = random.choices(ages, weights=age_weights, k=self.initial_population_size)
        genders = random.choices(['male', 'female'], k=self.initial_population_size)

        # Предвычисляем аллели для всех
        alleles = list(self.mutation_frequencies.keys())
        allele_weights = list(self.mutation_frequencies.values())

        # Очищаем индексы перед инициализацией
        self._male_indices = []
        self._female_indices = []
        self._alive_indices = []
        self._partnered_indices = set()

        for i in range(self.initial_population_size):
            agent = Agent(
                gender=genders[i],
                age=selected_ages[i],
                generation=0,
                max_age_limit=self.max_age_limit,
                father_id=None,
                mother_id=None
            )

            allele1 = random.choices(alleles, weights=allele_weights)[0]
            allele2 = random.choices(alleles, weights=allele_weights)[0]
            agent.set_genotype(allele1, allele2)

            self.agents[agent.id] = agent

            # Обновляем индексы
            self._alive_indices.append(agent.id)
            if agent.gender == 'male':
                self._male_indices.append(agent.id)
            else:
                self._female_indices.append(agent.id)

        self._form_initial_partnerships()

    def _get_random_allele(self) -> str:
        alleles = list(self.mutation_frequencies.keys())
        weights = list(self.mutation_frequencies.values())
        return random.choices(alleles, weights=weights)[0]

    def _form_initial_partnerships(self):
        """Оптимизированное формирование начальных пар"""
        males = [a for a in self.agents.values()
                 if a.gender == 'male' and a.partner_id is None and a.age >= 18]
        females = [a for a in self.agents.values()
                   if a.gender == 'female' and a.partner_id is None and a.age >= 18]

        males.sort(key=lambda x: x.age)
        females.sort(key=lambda x: x.age)

        i, j = 0, 0

        while i < len(males) and j < len(females):
            male = males[i]
            female = females[j]

            age_diff = male.age - female.age

            if abs(age_diff) <= 10:
                male.set_partner(female)
                self._partnered_indices.add(male.id)
                self._partnered_indices.add(female.id)
                i += 1
                j += 1
            elif age_diff > 10:
                j += 1
            else:
                i += 1

    def run_validation_simulation(self, target_year: int = 2012) -> dict:
        """Запуск с проверкой кэша"""
        if self.use_cache:
            cache_key = self._get_cache_key()
            cached_result = SIMULATION_CACHE.get(cache_key)
            if cached_result is not None:
                global CACHE_HITS
                CACHE_HITS += 1
                return cached_result
            global CACHE_MISSES
            CACHE_MISSES += 1

        self.initialize_founders()

        for year in range(1, self.simulation_years + 1):
            self.year = year
            self._run_single_year_fast()

            current_calendar_year = self.base_year + year

            if current_calendar_year == target_year:
                self.validation_point_reached = True
                self._collect_validation_data()

        results = {
            'validation_year': target_year,
            'final_population': len(self.agents),
            'final_prevalence': self.calculate_simulated_prevalence(),
            'validation_point_reached': self.validation_point_reached,
            'validation_data': self.validation_data,
            'children_born': self.children_born
        }

        if self.use_cache and len(SIMULATION_CACHE) < MAX_CACHE_SIZE:
            SIMULATION_CACHE[cache_key] = results

        return results

    def _collect_validation_data(self):
        """Быстрый сбор валидационных данных"""
        affected_count = 0
        symptomatic_count = 0
        asymptomatic_carriers = 0
        onset_ages = []

        for agent in self.agents.values():
            if not agent.alive:
                continue

            if agent.genotype_status == 'affected':
                affected_count += 1

            if agent.clinical_status == 'symptomatic':
                symptomatic_count += 1
            elif agent.will_develop_symptoms and agent.clinical_status == 'asymptomatic':
                asymptomatic_carriers += 1

            if agent.age_of_onset is not None:
                onset_ages.append(agent.age_of_onset)

        mean_onset_age = np.mean(onset_ages) if onset_ages else 0.0

        self.validation_data = {
            'year': self.base_year + self.year,
            'prevalence': self.calculate_simulated_prevalence(),
            'population': len(self.agents),
            'affected_count': affected_count,
            'symptomatic_count': symptomatic_count,
            'asymptomatic_carriers': asymptomatic_carriers,
            'mean_onset_age': mean_onset_age,
            'penetrance_by_genotype': {
                k: PENETRANCE[k] for k in ('M694V_homozygous', 'compound_heterozygous', 'heterozygous')
            }
        }

    def _run_single_year_fast(self):
        """Оптимизированный годовой цикл"""
        # Сначала собираем умерших, но не удаляем их сразу из agents
        dead_agents = []
        agents_to_remove = []

        for agent_id, agent in list(self.agents.items()):
            if agent.alive:
                agent.age_year()
            if not agent.alive:
                dead_agents.append(agent_id)
                agents_to_remove.append(agent_id)

        # Удаляем умерших из agents и индексов
        for agent_id in agents_to_remove:
            if agent_id in self.agents:
                del self.agents[agent_id]

            # Обновляем все индексы
            if agent_id in self._alive_indices:
                self._alive_indices.remove(agent_id)
            if agent_id in self._male_indices:
                self._male_indices.remove(agent_id)
            if agent_id in self._female_indices:
                self._female_indices.remove(agent_id)
            if agent_id in self._partnered_indices:
                self._partnered_indices.remove(agent_id)

        # Формируем новые пары и рождаем детей
        self._form_new_partnerships_fast()
        self._birth_process_fast()

    def _form_new_partnerships_fast(self):
        """Быстрое формирование новых пар"""
        single_males = []
        single_females = []

        # Используем alive_indices для быстрого доступа, но проверяем существование
        for agent_id in list(self._alive_indices):  # Создаем копию списка для итерации
            if agent_id not in self.agents:
                # Очищаем индекс если агент не существует
                self._alive_indices.remove(agent_id)
                continue

            if agent_id in self._partnered_indices:
                continue

            agent = self.agents[agent_id]
            if agent.gender == 'male' and 18 <= agent.age <= 45:
                single_males.append(agent)
            elif agent.gender == 'female' and 18 <= agent.age <= 40:
                single_females.append(agent)

        # Если мало кандидатов, пропускаем
        if len(single_males) < 2 or len(single_females) < 2:
            return

        # Быстрое перемешивание
        random.shuffle(single_males)
        random.shuffle(single_females)

        # Формируем пары
        n_pairs = min(len(single_males), len(single_females))
        pairs_formed = 0
        max_pairs_per_year = min(50, n_pairs)

        used_males = set()
        used_females = set()

        for i in range(min(n_pairs, max_pairs_per_year)):
            if i >= len(single_males) or i >= len(single_females):
                break

            male = single_males[i]
            female = single_females[i]

            # Проверяем, не используются ли уже эти агенты
            if male.id in used_males or female.id in used_females:
                continue

            if abs(male.age - female.age) <= 15 and random.random() < 0.3:
                male.set_partner(female)
                self._partnered_indices.add(male.id)
                self._partnered_indices.add(female.id)
                used_males.add(male.id)
                used_females.add(female.id)
                pairs_formed += 1

    def _birth_process_fast(self):
        """Оптимизированный процесс рождения"""
        potential_parents = []

        # Быстрый поиск потенциальных родителей
        for agent_id in list(self._partnered_indices):  # Создаем копию
            if agent_id not in self.agents:
                # Очищаем индекс если агент не существует
                self._partnered_indices.remove(agent_id)
                continue

            agent = self.agents[agent_id]
            if (agent.alive and agent.can_have_children() and
                    agent.partner_id in self.agents):
                potential_parents.append(agent)

        # Ограничиваем количество проверок
        max_births = min(len(potential_parents) // 2, 30)
        births = 0
        used_parents = set()

        for parent in potential_parents:
            if births >= max_births:
                break

            # Проверяем, не использовали ли уже этого родителя
            if parent.id in used_parents:
                continue

            partner = self.agents.get(parent.partner_id)
            if not partner or not partner.can_have_children():
                continue

            # Быстрый расчет вероятности
            avg_age = (parent.age + partner.age) / 2
            age_key = int(avg_age)
            if age_key < 18:
                age_key = 18
            elif age_key > 45:
                age_key = 45

            birth_prob = self._birth_probs.get(age_key, 0.02)

            if random.random() < birth_prob:
                child_id = self._create_child_fast(parent, partner)
                if child_id:
                    self.children_born += 1
                    births += 1
                    used_parents.add(parent.id)
                    used_parents.add(partner.id)

    def _create_child_fast(self, parent1: Agent, parent2: Agent) -> Optional[str]:
        """Быстрое создание ребенка, возвращает id ребенка или None"""
        try:
            if parent1.gender == 'male':
                father, mother = parent1, parent2
            else:
                father, mother = parent2, parent1

            # Наследование аллелей
            father_allele = father.mefv_allele_1 if random.random() < 0.5 else father.mefv_allele_2
            mother_allele = mother.mefv_allele_1 if random.random() < 0.5 else mother.mefv_allele_2

            child_gender = 'male' if random.random() < 0.5 else 'female'
            child_generation = max(father.generation, mother.generation) + 1

            child = Agent(
                gender=child_gender,
                age=0,
                generation=child_generation,
                max_age_limit=self.max_age_limit,
                father_id=father.id,
                mother_id=mother.id
            )

            child.set_genotype(father_allele, mother_allele)
            self.agents[child.id] = child

            # Обновляем индексы
            self._alive_indices.append(child.id)
            if child.gender == 'male':
                self._male_indices.append(child.id)
            else:
                self._female_indices.append(child.id)

            return child.id

        except Exception as e:
            if self.verbose:
                print(f"Ошибка при создании ребенка: {e}")
            return None

    def calculate_simulated_prevalence(self) -> float:
        """Быстрый расчет распространенности"""
        if not self._alive_indices:
            return 0.0

        symptomatic_count = 0
        total_alive = 0

        # Создаем копию списка для безопасной итерации
        for agent_id in list(self._alive_indices):
            if agent_id not in self.agents:
                # Очищаем индекс если агент не существует
                self._alive_indices.remove(agent_id)
                continue

            agent = self.agents[agent_id]
            if agent.alive:
                total_alive += 1
                if agent.clinical_status == 'symptomatic':
                    symptomatic_count += 1

        return symptomatic_count / total_alive if total_alive > 0 else 0.0


def load_real_data(file_path: str = 'FMF_data2.csv') -> pd.DataFrame:
    """Загрузка данных с кэшированием"""
    data = pd.read_csv(file_path)
    data['prevalence_0_49'] = data['Registered_0-49'] / data['0-49 population_Total']

    target_columns = [
        'years',
        'Registered_0-49',
        '0-49 population_Total',
        'prevalence_0_49'
    ]

    return data[target_columns].copy()


def _run_single_simulation_worker(year, population_size, sim_params, mutation_freqs, use_cache=True):
    """Оптимизированная функция-воркер"""
    simulation_years = year - sim_params['base_year']

    sim = GenerationSimulation(
        initial_population_size=population_size,
        max_age_limit=sim_params['max_age_limit'],
        simulation_years=simulation_years,
        validation_year=year,
        base_year=sim_params['base_year'],
        mutation_frequencies=mutation_freqs.copy(),
        use_cache=use_cache
    )

    results = sim.run_validation_simulation(target_year=year)
    return results['final_prevalence'] if results.get('validation_point_reached') else None


def run_validation_multiyear(target_years: List[int] = None,
                             n_simulations_per_year: int = 30,
                             population_size: int = None,
                             use_parallel: bool = True) -> Dict:
    """Оптимизированная мультигодовая валидация"""
    if target_years is None:
        target_years = SIM_PARAMS['target_years']
    if population_size is None:
        population_size = SIM_PARAMS['initial_population_size']

    print(f" Запуск валидации (по {n_simulations_per_year} сим/год)")
    print(f"   Режим: {'параллельный' if use_parallel else 'последовательный'}")

    real_data = load_real_data('FMF_data2.csv')
    results_by_year = {}

    for year in target_years:
        print(f"\n{'=' * 50}")
        print(f" Валидация {year} года")
        print(f"{'=' * 50}")

        year_data = real_data[real_data['years'] == year]
        if year_data.empty:
            print(f"⚠️ Данные за {year} год не найдены.")
            continue

        real_prevalence = year_data['prevalence_0_49'].iloc[0]

        if use_parallel:
            # Параллельный режим с батчингом
            n_jobs = mp.cpu_count()
            batch_size = max(1, n_simulations_per_year // n_jobs)

            raw_results = []
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = []
                for _ in range(n_simulations_per_year):
                    future = executor.submit(
                        _run_single_simulation_worker,
                        year, population_size, SIM_PARAMS, MUTATION_FREQS, True
                    )
                    futures.append(future)

                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc=f"Симуляции {year}", ncols=80):
                    try:
                        result = future.result(timeout=300)  # 5 минут таймаут
                        raw_results.append(result)
                    except Exception as e:
                        print(f"  Ошибка: {e}")
                        raw_results.append(None)
        else:
            # Последовательный режим
            raw_results = []
            for _ in tqdm(range(n_simulations_per_year), desc=f"Симуляции {year}", ncols=80):
                result = _run_single_simulation_worker(
                    year, population_size, SIM_PARAMS, MUTATION_FREQS, True
                )
                raw_results.append(result)

        prevalences = [p for p in raw_results if p is not None]

        if not prevalences:
            print(f"❌ Ошибка: нет успешных симуляций для {year}")
            continue

        # Быстрые вычисления статистики
        prevalences_array = np.array(prevalences)
        mean_prev = np.mean(prevalences_array)
        std_prev = np.std(prevalences_array)
        ci_lower, ci_upper = np.percentile(prevalences_array, [2.5, 97.5])
        error = abs(mean_prev - real_prevalence)
        rel_error = (error / real_prevalence) * 100 if real_prevalence > 0 else 0
        is_valid = ci_lower <= real_prevalence <= ci_upper

        results_by_year[year] = {
            'real_prevalence': real_prevalence,
            'simulated_mean': mean_prev,
            'simulated_std': std_prev,
            'ci_95': [ci_lower, ci_upper],
            'absolute_error': error,
            'relative_error': rel_error,
            'n_successful': len(prevalences),
            'all_prevalences': prevalences
        }

        print(f"\nРезультаты для {year} года:")
        print(f"  Реальная: {real_prevalence:.6f}")
        print(f"  Модель:   {mean_prev:.6f} ± {std_prev:.6f}")
        print(f"  95% ДИ:   [{ci_lower:.6f}, {ci_upper:.6f}]")
        print(f"  Ошибка:   {error:.6f} ({rel_error:.1f}%)")
        print(f"  Статус:   {'✅ ВАЛИДНО' if is_valid else '❌ НЕ ВАЛИДНО'}")

    global CACHE_HITS, CACHE_MISSES
    print(f"\nСтатистика кэша: Хиты={CACHE_HITS}, Промахи={CACHE_MISSES}, "
          f"Hit ratio={CACHE_HITS / (CACHE_HITS + CACHE_MISSES) * 100 if CACHE_HITS + CACHE_MISSES > 0 else 0:.1f}%")

    return results_by_year
#
#
# class GeneticOptimizer:
#     """Оптимизированный генетический алгоритм"""
#
#     BOUNDS = {
#         'M694V': (0.020, 0.060),
#         'V726A': (0.005, 0.015),
#         'M680I': (0.001, 0.010),
#         'R761H': (0.0005, 0.004),
#     }
#
#     DETECTION_BOUNDS = {
#         'slope': (0.0, 0.05),
#         'intercept': (0.4, 0.7),
#     }
#
#     def __init__(self,
#                  real_data: pd.DataFrame,
#                  sim_params: dict,
#                  base_freqs: dict,
#                  target_years,
#                  population_size: int = 20,
#                  generations: int = 10,
#                  n_simulations: int = 3,
#                  n_jobs: int = -1,
#                  use_parallel: bool = True,
#                  optimize_detection: bool = True,
#                  verbose: bool = True,
#                  use_cache: bool = True):
#
#         self.real_data = real_data
#         self.sim_params = sim_params
#         self.base_freqs = base_freqs.copy()
#         self.target_years = target_years
#         self.population_size = population_size
#         self.generations = generations
#         self.n_simulations = max(2, n_simulations)
#         self.use_parallel = use_parallel
#         self.n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs
#         self.optimize_detection = optimize_detection
#         self.verbose = verbose
#         self.use_cache = use_cache
#
#         # Предвычисленные целевые значения
#         self.targets = {}
#         self.target_weights = {}
#         for i, year in enumerate(target_years):
#             year_data = real_data[real_data['years'] == year]
#             if len(year_data) > 0:
#                 self.targets[year] = year_data['prevalence_0_49'].iloc[0]
#                 self.target_weights[year] = 1.0 + 0.1 * i
#
#         self.sim_cfg = {
#             'pop_size': sim_params['initial_population_size'],
#             'max_age': sim_params['max_age_limit'],
#             'base_year': sim_params['base_year']
#         }
#
#         self.base_detection_year = min(target_years) if target_years else 2012
#
#         # Кэш для результатов оценки
#         self.fitness_cache = {}
#         self.cache_hits = 0
#         self.cache_misses = 0
#
#         self.history = []
#         self.best_fitness_ever = float('inf')
#
#     def _individual_to_key(self, individual: Dict) -> str:
#         """Быстрое создание ключа для особи"""
#         genes_key = tuple(sorted((k, round(v, 6)) for k, v in individual['genes'].items()))
#         detection_key = tuple(sorted((k, round(v, 6)) for k, v in individual['detection'].items()))
#         return hashlib.md5(str((genes_key, detection_key)).encode()).hexdigest()
#
#     def _normalize_frequencies(self, freqs: Dict[str, float]) -> Dict[str, float]:
#         """Быстрая нормализация частот"""
#         freqs = freqs.copy()
#         mutant_sum = sum(v for k, v in freqs.items() if k != 'N')
#
#         if mutant_sum > 0.15:
#             scale = 0.14 / (mutant_sum if abs(mutant_sum) > 1e-10 else 1.0)
#             for k in freqs:
#                 if k != 'N':
#                     freqs[k] *= scale
#             mutant_sum = sum(v for k, v in freqs.items() if k != 'N')
#
#         freqs['N'] = max(0.001, 1.0 - mutant_sum)
#         return freqs
#
#     def _detection_factor(self, detection_params: Dict, year: int) -> float:
#         """Быстрый расчет фактора диагностики"""
#         slope = detection_params['slope']
#         intercept = detection_params['intercept']
#         t = year - self.base_detection_year
#         return np.clip(slope * t + intercept, 0.3, 1.0)
#
#     def _create_individual(self) -> Dict:
#         """Быстрое создание особи"""
#         genetic_part = self.base_freqs.copy()
#
#         for mutation, (low, high) in self.BOUNDS.items():
#             if mutation in genetic_part:
#                 if np.random.random() < 0.7:
#                     # Логнормальное распределение
#                     log_low = np.log(max(low, 1e-10))
#                     log_high = np.log(high)
#                     genetic_part[mutation] = np.exp(np.random.uniform(log_low, log_high))
#                 else:
#                     genetic_part[mutation] = np.random.uniform(low, high)
#
#         genetic_part = self._normalize_frequencies(genetic_part)
#
#         if self.optimize_detection:
#             diagnostic_part = {
#                 'slope': np.random.uniform(*self.DETECTION_BOUNDS['slope']),
#                 'intercept': np.random.uniform(*self.DETECTION_BOUNDS['intercept'])
#             }
#         else:
#             diagnostic_part = {'slope': 0.02, 'intercept': 0.6}
#
#         return {'genes': genetic_part, 'detection': diagnostic_part}
#
#     def _mutate(self, individual: Dict, generation: int) -> Dict:
#         """Быстрая мутация"""
#         mutated = {
#             'genes': individual['genes'].copy(),
#             'detection': individual['detection'].copy()
#         }
#
#         progress = generation / max(1, self.generations)
#         sigma = 0.2 * (1 - 0.7 * progress)
#         mutation_rate = 0.3 * (1 - 0.5 * progress)
#
#         # Мутация генов
#         for mutation, (low, high) in self.BOUNDS.items():
#             if mutation in mutated['genes'] and np.random.random() < mutation_rate:
#                 current = mutated['genes'][mutation]
#                 if np.random.random() < 0.8:
#                     safe_current = max(current, low * 0.1)
#                     log_current = np.log(safe_current)
#                     log_new = log_current + np.random.normal(0, sigma)
#                     mutated['genes'][mutation] = np.clip(np.exp(log_new), low, high)
#                 else:
#                     mutated['genes'][mutation] = np.random.uniform(low, high)
#
#         mutated['genes'] = self._normalize_frequencies(mutated['genes'])
#
#         # Мутация диагностики
#         if self.optimize_detection and np.random.random() < mutation_rate:
#             if np.random.random() < 0.5:
#                 current = mutated['detection']['slope']
#                 low, high = self.DETECTION_BOUNDS['slope']
#                 mutated['detection']['slope'] = np.clip(
#                     current + np.random.normal(0, sigma * 0.02), low, high
#                 )
#             else:
#                 current = mutated['detection']['intercept']
#                 low, high = self.DETECTION_BOUNDS['intercept']
#                 mutated['detection']['intercept'] = np.clip(
#                     current + np.random.normal(0, sigma * 0.1), low, high
#                 )
#
#         return mutated
#
#     def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
#         """Быстрый кроссовер"""
#         child_genes = {}
#         alpha = 0.3
#
#         for mutation in self.BOUNDS.keys():
#             if mutation in parent1['genes'] and mutation in parent2['genes']:
#                 p1 = parent1['genes'][mutation]
#                 p2 = parent2['genes'][mutation]
#                 d = abs(p1 - p2)
#                 low = min(p1, p2) - alpha * d
#                 high = max(p1, p2) + alpha * d
#
#                 bounds_low, bounds_high = self.BOUNDS[mutation]
#                 low = max(low, bounds_low)
#                 high = min(high, bounds_high)
#
#                 if low > high:
#                     low, high = high, low
#
#                 child_genes[mutation] = np.random.uniform(low, high)
#
#         child_genes = self._normalize_frequencies(child_genes)
#
#         if self.optimize_detection:
#             child_detection = {}
#             for param in ['slope', 'intercept']:
#                 p1 = parent1['detection'][param]
#                 p2 = parent2['detection'][param]
#                 d = abs(p1 - p2)
#                 low = min(p1, p2) - 0.4 * d
#                 high = max(p1, p2) + 0.4 * d
#
#                 bounds_low, bounds_high = self.DETECTION_BOUNDS[param]
#                 low = max(low, bounds_low)
#                 high = min(high, bounds_high)
#
#                 if low > high:
#                     low, high = high, low
#
#                 child_detection[param] = np.random.uniform(low, high)
#         else:
#             child_detection = parent1['detection'].copy()
#
#         return {'genes': child_genes, 'detection': child_detection}
#
#     def _evaluate_individual_batch(self, individuals: List[Dict]) -> List[float]:
#         """Пакетная оценка особей"""
#         fitness_scores = []
#
#         for individual in individuals:
#             # Проверка кэша
#             if self.use_cache:
#                 key = self._individual_to_key(individual)
#                 if key in self.fitness_cache:
#                     self.cache_hits += 1
#                     fitness_scores.append(self.fitness_cache[key])
#                     continue
#                 self.cache_misses += 1
#
#             # Оценка без детального прогресс-бара
#             freqs = individual['genes']
#             detection_params = individual['detection']
#
#             yearly_results = defaultdict(list)
#
#             # Запускаем симуляции
#             for _ in range(self.n_simulations):
#                 for year in self.target_years:
#                     if year not in self.targets:
#                         continue
#
#                     sim = GenerationSimulation(
#                         initial_population_size=self.sim_cfg['pop_size'],
#                         max_age_limit=self.sim_cfg['max_age'],
#                         simulation_years=year - self.sim_cfg['base_year'],
#                         validation_year=year,
#                         base_year=self.sim_cfg['base_year'],
#                         mutation_frequencies=freqs.copy(),
#                         verbose=False,
#                         use_cache=self.use_cache
#                     )
#
#                     results = sim.run_validation_simulation(target_year=year)
#                     if results and results.get('validation_point_reached'):
#                         yearly_results[year].append(results.get('final_prevalence', 0.0))
#
#             # Вычисление fitness
#             fitness = self._compute_fitness(yearly_results, freqs, detection_params)
#
#             if self.use_cache:
#                 self.fitness_cache[key] = fitness
#                 if len(self.fitness_cache) > 1000:
#                     # Очистка кэша при переполнении
#                     keys_to_remove = list(self.fitness_cache.keys())[:200]
#                     for k in keys_to_remove:
#                         del self.fitness_cache[k]
#
#             fitness_scores.append(fitness)
#
#         return fitness_scores
#
#     def _compute_fitness(self, yearly_results: Dict, freqs: Dict, detection_params: Dict) -> float:
#         """Быстрое вычисление fitness"""
#         total_error = 0
#         n_successful_years = 0
#
#         for year in self.target_years:
#             if year not in self.targets or year not in yearly_results or not yearly_results[year]:
#                 continue
#
#             true_prevalence = np.median(yearly_results[year])
#             detection_factor = self._detection_factor(detection_params, year)
#             observed_prevalence = true_prevalence * detection_factor
#             real_value = self.targets[year]
#             weight = self.target_weights.get(year, 1.0)
#
#             if abs(real_value) > 1e-10:
#                 rel_error_squared = ((observed_prevalence - real_value) / real_value) ** 2
#             else:
#                 rel_error_squared = (observed_prevalence - real_value) ** 2
#
#             total_error += weight * rel_error_squared
#             n_successful_years += 1
#
#         if n_successful_years == 0:
#             return 100.0
#
#         # Штрафы
#         mutant_sum = sum(v for k, v in freqs.items() if k != 'N')
#         if mutant_sum > 0.15:
#             total_error += (mutant_sum - 0.15) * 10
#
#         return total_error / n_successful_years
#
#     def _tournament_selection(self, population: List[Dict], fitness_scores: List[float]) -> Dict:
#         """Турнирная селекция"""
#         tournament_size = min(4, len(population))
#         idx = np.random.choice(len(population), tournament_size, replace=False)
#         winner_idx = idx[np.argmin([fitness_scores[i] for i in idx])]
#         return {
#             'genes': population[winner_idx]['genes'].copy(),
#             'detection': population[winner_idx]['detection'].copy()
#         }
#
#     def optimize(self) -> Dict:
#         """Основной метод оптимизации"""
#         print("\nСоздание начальной популяции...")
#
#         # Создаем популяцию
#         population = [self._create_individual() for _ in range(self.population_size)]
#
#         # Основной цикл
#         for generation in range(self.generations):
#             gen_start = time.time()
#
#             # Пакетная оценка
#             fitness_scores = self._evaluate_individual_batch(population)
#
#             # Поиск лучшей особи
#             best_idx = np.argmin(fitness_scores)
#             best_fitness = fitness_scores[best_idx]
#             best_individual = {
#                 'genes': population[best_idx]['genes'].copy(),
#                 'detection': population[best_idx]['detection'].copy()
#             }
#
#             mean_fitness = np.mean(fitness_scores)
#             std_fitness = np.std(fitness_scores)
#
#             # Сохраняем историю
#             self.history.append({
#                 'generation': generation,
#                 'best_fitness': best_fitness,
#                 'best_individual': best_individual.copy(),
#                 'mean_fitness': mean_fitness,
#                 'std_fitness': std_fitness,
#                 'time': time.time() - gen_start
#             })
#
#             # Прогресс
#             m694v = best_individual['genes'].get('M694V', 0)
#             print(f"\nПоколение {generation + 1}/{self.generations}")
#             print(f"  Лучший fitness: {best_fitness:.6f}")
#             print(f"  M694V: {m694v:.4f}")
#             print(f"  Средний fitness: {mean_fitness:.6f} ± {std_fitness:.6f}")
#
#             if self.optimize_detection:
#                 slope = best_individual['detection']['slope']
#                 intercept = best_individual['detection']['intercept']
#                 print(f"  Диагностика: y = {slope:.3f}·t + {intercept:.2f}")
#
#             if generation == self.generations - 1:
#                 break
#
#             # Элитизм
#             elite_size = max(2, self.population_size // 10)
#             sorted_idx = np.argsort(fitness_scores)
#             elite = [population[i] for i in sorted_idx[:elite_size]]
#
#             # Создание нового поколения
#             new_population = elite.copy()
#
#             while len(new_population) < self.population_size:
#                 parent1 = self._tournament_selection(population, fitness_scores)
#                 parent2 = self._tournament_selection(population, fitness_scores)
#
#                 if np.random.random() < 0.7:
#                     child = self._crossover(parent1, parent2)
#                 else:
#                     child = {'genes': parent1['genes'].copy(), 'detection': parent1['detection'].copy()}
#
#                 child = self._mutate(child, generation)
#                 new_population.append(child)
#
#             population = new_population
#
#         # Финальный отчет
#         print(f"\nСтатистика кэша: Хиты={self.cache_hits}, Промахи={self.cache_misses}, "
#               f"Hit ratio={self.cache_hits / (self.cache_hits + self.cache_misses) * 100 if self.cache_hits + self.cache_misses > 0 else 0:.1f}%")
#
#         return {
#             'best_individual': self.history[-1]['best_individual'],
#             'history': self.history,
#             'fitness': self.history[-1]['best_fitness']
#         }
#
#
# def run_genetic_optimization_pipeline(real_data: pd.DataFrame,
#                                       sim_params: dict,
#                                       base_freqs: dict,
#                                       fast_mode: bool = True,
#                                       optimize_detection: bool = True,
#                                       verbose: bool = True):
#     """Оптимизированный пайплайн"""
#
#     print("\n" + "=" * 80)
#     print("ГЕНЕТИЧЕСКАЯ ОПТИМИЗАЦИЯ")
#     print("=" * 80)
#
#     if fast_mode:
#         train_years = [2012, 2014]
#         print(f"\nБыстрый режим: {train_years}")
#         optimizer = GeneticOptimizer(
#             real_data=real_data,
#             sim_params=sim_params,
#             base_freqs=base_freqs,
#             target_years=train_years,
#             population_size=10,
#             generations=5,
#             n_simulations=2,
#             optimize_detection=optimize_detection,
#             verbose=verbose
#         )
#     else:
#         train_years = [2012, 2014, 2016, 2018]
#         print(f"\nПолный режим: {train_years}")
#         optimizer = GeneticOptimizer(
#             real_data=real_data,
#             sim_params=sim_params,
#             base_freqs=base_freqs,
#             target_years=train_years,
#             population_size=15,
#             generations=8,
#             n_simulations=3,
#             optimize_detection=optimize_detection,
#             verbose=verbose
#         )
#
#     opt_results = optimizer.optimize()
#
#     return {
#         'optimization': opt_results,
#         'optimizer': optimizer
#     }
#
#
# if __name__ == "__main__":
#     mp.freeze_support()
#
#     from tabulate import tabulate
#     import warnings
#
#     warnings.filterwarnings('ignore')
#
#     print("\n" + "═" * 70)
#     print(" " * 20 + "🧬 FMF ГЕНЕТИЧЕСКАЯ ОПТИМИЗАЦИЯ")
#     print("═" * 70)
#
#     # Загрузка конфига
#     with open('config.json') as f:
#         cfg = json.load(f)
#
#     sim_params = cfg['simulation_params']
#     base_freqs = cfg['mutation_frequencies']
#
#     # Загрузка данных
#     df = pd.read_csv('FMF_data2.csv')
#     df['prevalence'] = df['Registered_0-49'] / df['0-49 population_Total']
#
#     # Меню
#     print("\n[1] Быстрый тест (5-10 минут)")
#     print("[2] Полный анализ (30-60 минут)")
#     print("[3] Сравнение с/без диагностики")
#
#     mode = input("\nВыберите режим: ").strip()
#
#     # Запускаем выбранный режим
#     if mode == '1':
#         results = run_genetic_optimization_pipeline(df, sim_params, base_freqs, True, True)
#         train_years = [2012, 2014]
#         test_years = [2016, 2018, 2020]
#         optimize_detection = True
#     elif mode == '2':
#         results = run_genetic_optimization_pipeline(df, sim_params, base_freqs, False, True)
#         train_years = [2012, 2014, 2016, 2018]
#         test_years = [2020, 2022, 2024]
#         optimize_detection = True
#     elif mode == '3':
#         print("\n➡️  Без диагностики...")
#         r1 = run_genetic_optimization_pipeline(df, sim_params, base_freqs, True, False, False)
#         print("\n➡️  С диагностикой...")
#         r2 = run_genetic_optimization_pipeline(df, sim_params, base_freqs, True, True, False)
#         results = r2  # для таблицы используем результаты с диагностикой
#         train_years = [2012, 2014]
#         test_years = [2016, 2018, 2020]
#         optimize_detection = True
#     else:
#         results = run_genetic_optimization_pipeline(df, sim_params, base_freqs, True, True)
#         train_years = [2012, 2014]
#         test_years = [2016, 2018, 2020]
#         optimize_detection = True
#
#     # ========== ФОРМИРУЕМ ТАБЛИЦУ РЕЗУЛЬТАТОВ ==========
#     print("\n" + "═" * 100)
#     print(" " * 35 + "📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
#     print("═" * 100)
#
#     # Извлекаем данные из результатов
#     best = results['optimization']['best_individual']
#     opt = results['optimizer']
#
#     print(f"\n📅 Обучающие годы: {train_years}")
#     print(f"📅 Тестовые годы: {test_years}")
#
#     # Собираем данные по годам
#     print("\n🔄 Расчет превалентности по годам...")
#
#     validation_results = {}
#     all_years = sorted(list(set(train_years + test_years)))
#
#     from tqdm import tqdm
#
#     for year in tqdm(all_years, desc="Валидация", unit="год", ncols=80):
#         if year not in df['years'].values:
#             continue
#
#         real_value = df[df['years'] == year]['prevalence'].iloc[0]
#
#         # Запускаем 3 симуляции для усреднения
#         prevalences = []
#         for _ in range(3):
#             sim = GenerationSimulation(
#                 initial_population_size=sim_params['initial_population_size'],
#                 max_age_limit=sim_params['max_age_limit'],
#                 simulation_years=year - sim_params['base_year'],
#                 validation_year=year,
#                 base_year=sim_params['base_year'],
#                 mutation_frequencies=best['genes'].copy(),
#                 verbose=False
#             )
#             res = sim.run_validation_simulation(target_year=year)
#             if res and res.get('validation_point_reached'):
#                 prevalences.append(res.get('final_prevalence', 0.0))
#
#         if prevalences:
#             true_prev = np.mean(prevalences)
#             std_prev = np.std(prevalences)
#
#             # Применяем диагностику если она есть
#             if hasattr(opt, '_detection_factor') and optimize_detection:
#                 det_factor = opt._detection_factor(best['detection'], year)
#                 obs_prev = true_prev * det_factor
#             else:
#                 obs_prev = true_prev
#                 det_factor = 1.0
#
#             error_pct = abs(obs_prev - real_value) / real_value * 100
#
#             validation_results[year] = {
#                 'real': real_value,
#                 'observed': obs_prev,
#                 'biological': true_prev,
#                 'std': std_prev,
#                 'detection': det_factor,
#                 'error': error_pct,
#                 'is_train': year in train_years
#             }
#
#     # ========== ТАБЛИЦА 1: ПРЕВАЛЕНТНОСТЬ ПО ГОДАМ ==========
#     print("\n" + "─" * 100)
#     print("📈 СРАВНЕНИЕ РЕАЛЬНОЙ И МОДЕЛЬНОЙ ПРЕВАЛЕНТНОСТИ")
#     print("─" * 100)
#
#     table1 = []
#     for year in sorted(validation_results.keys()):
#         r = validation_results[year]
#
#         # Определяем тип года
#         year_type = "📚 Train" if r['is_train'] else "🔮 Test"
#
#         # Оценка качества
#         if r['error'] < 10:
#             quality = "✅"
#         elif r['error'] < 20:
#             quality = "⚠️"
#         else:
#             quality = "❌"
#
#         table1.append([
#             year,
#             year_type,
#             f"{r['real']:.6f}",
#             f"{r['observed']:.6f}",
#             f"{r['biological']:.6f}",
#             f"{r['detection']:.1%}",
#             f"{r['error']:.2f}% {quality}"
#         ])
#
#     headers1 = [
#         "Год",
#         "Тип",
#         "Реальная",
#         "Модель\n(с диаг.)",
#         "Истинная\n(биол.)",
#         "Диагностика",
#         "Ошибка"
#     ]
#
#     print(tabulate(table1, headers=headers1, tablefmt="grid", numalign="right", stralign="center"))
#
#     # ========== ТАБЛИЦА 2: СТАТИСТИКА ОШИБОК ==========
#     print("\n" + "─" * 100)
#     print("📊 СТАТИСТИКА ОШИБОК")
#     print("─" * 100)
#
#     train_errors = [r['error'] for r in validation_results.values() if r['is_train']]
#     test_errors = [r['error'] for r in validation_results.values() if not r['is_train']]
#     all_errors = [r['error'] for r in validation_results.values()]
#
#     stats_table = []
#
#     if train_errors:
#         stats_table.append([
#             "Обучающая выборка",
#             f"{len(train_errors)} лет",
#             f"{np.mean(train_errors):.2f}%",
#             f"{np.median(train_errors):.2f}%",
#             f"{np.min(train_errors):.2f}%",
#             f"{np.max(train_errors):.2f}%"
#         ])
#
#     if test_errors:
#         stats_table.append([
#             "Тестовая выборка",
#             f"{len(test_errors)} лет",
#             f"{np.mean(test_errors):.2f}%",
#             f"{np.median(test_errors):.2f}%",
#             f"{np.min(test_errors):.2f}%",
#             f"{np.max(test_errors):.2f}%"
#         ])
#
#     if all_errors:
#         stats_table.append([
#             "Все годы",
#             f"{len(all_errors)} лет",
#             f"{np.mean(all_errors):.2f}%",
#             f"{np.median(all_errors):.2f}%",
#             f"{np.min(all_errors):.2f}%",
#             f"{np.max(all_errors):.2f}%"
#         ])
#
#     headers2 = ["Выборка", "Кол-во", "Среднее", "Медиана", "Минимум", "Максимум"]
#     print(tabulate(stats_table, headers=headers2, tablefmt="grid", numalign="right"))
#
#     # ========== ТАБЛИЦА 3: ОПТИМИЗИРОВАННЫЕ ГЕНЫ ==========
#     print("\n" + "─" * 100)
#     print("🧬 ОПТИМИЗИРОВАННЫЕ ЧАСТОТЫ ГЕНОВ")
#     print("─" * 100)
#
#     gene_table = []
#     for mutation in ['M694V', 'V726A', 'M680I', 'R761H']:
#         opt_val = best['genes'].get(mutation, 0)
#         base_val = base_freqs.get(mutation, 0)
#         change = ((opt_val - base_val) / base_val * 100) if base_val > 0 else float('inf')
#
#         # Проверка границ
#         bounds = GeneticOptimizer.BOUNDS.get(mutation, (0, 1))
#         in_bounds = bounds[0] <= opt_val <= bounds[1]
#         status = "✅" if in_bounds else "⚠️"
#
#         change_str = f"{change:+.1f}%" if base_val > 0 else "новый"
#
#         gene_table.append([
#             mutation,
#             f"{opt_val:.6f}",
#             f"[{bounds[0]:.4f}, {bounds[1]:.4f}]",
#             status,
#             f"{base_val:.6f}",
#             change_str
#         ])
#
#     headers3 = ["Мутация", "Оптимизировано", "Допустимый диапазон", "Норма", "Исходно", "Изменение"]
#     print(tabulate(gene_table, headers=headers3, tablefmt="grid", numalign="right"))
#
#     # ========== ДИАГНОСТИКА ==========
#     if optimize_detection and 'detection' in best:
#         print("\n" + "─" * 100)
#         print("📈 ПАРАМЕТРЫ ДИАГНОСТИКИ")
#         print("─" * 100)
#
#         det = best['detection']
#         base_year = opt.base_detection_year if hasattr(opt, 'base_detection_year') else 2012
#         print(f"\n📐 Функция: Detection(t) = {det['slope']:.3f}·(t - {base_year}) + {det['intercept']:.2f}")
#
#         det_table = []
#         for year in [2012, 2014, 2016, 2018, 2020, 2022, 2024]:
#             if year >= base_year:
#                 factor = det['slope'] * (year - base_year) + det['intercept']
#                 factor = max(0.3, min(1.0, factor))
#                 det_table.append([year, f"{factor:.1%}"])
#
#         print(tabulate(det_table, headers=["Год", "Коэффициент выявляемости"], tablefmt="simple"))
#
#         # ========== ИТОГОВАЯ ОЦЕНКА ==========
#         print("\n" + "═" * 100)
#         print(" " * 40 + "🎯 ИТОГОВАЯ ОЦЕНКА")
#         print("═" * 100)
#
#         if test_errors:
#             avg_test_error = np.mean(test_errors)
#
#             if avg_test_error < 15:
#                 rating = "🏆 ОТЛИЧНО"
#                 desc = "Модель отлично прогнозирует реальные данные"
#             elif avg_test_error < 25:
#                 rating = "👍 ХОРОШО"
#                 desc = "Модель хорошо описывает реальные данные"
#             elif avg_test_error < 35:
#                 rating = "👌 УДОВЛЕТВОРИТЕЛЬНО"
#                 desc = "Модель приемлемо описывает данные, требуется доработка"
#             else:
#                 rating = "👎 ПЛОХО"
#                 desc = "Модель плохо описывает реальные данные"
#
#             print(f"\n  {rating}")
#             print(f"  {desc}")
#             print(f"\n  📊 Средняя ошибка на тестовых данных: {avg_test_error:.2f}%")
#
#         if train_errors:
#             print(f"  📚 Средняя ошибка на обучающих данных: {np.mean(train_errors):.2f}%")
#
#         # Исправляем ключ с 'optimization_time' на 'time'
#         if 'optimization' in results and 'time' in results['optimization']:
#             print(f"  ⏱️  Время оптимизации: {results['optimization']['time']:.1f} сек")
#
#         if 'optimization' in results and 'cache_stats' in results['optimization']:
#             print(f"  💾 Эффективность кэша: {results['optimization']['cache_stats']['hit_ratio']:.1f}%")
#
#         print("\n" + "═" * 100)
#         print("✅ РАБОТА ЗАВЕРШЕНА")
#         print("═" * 100)

class GeneticOptimizer:
    """Оригинальный генетический алгоритм"""

    BOUNDS = {
        'M694V': (0.020, 0.080),
        'V726A': (0.0005, 0.015),
        'M680I': (0.0005, 0.015),
        'R761H': (0.0001, 0.006),
    }

    DETECTION_BOUNDS = {
        'slope': (0.0, 0.05),
        'intercept': (0.3, 0.9),  # Оригинальные границы
    }

    def __init__(self,
                 real_data: pd.DataFrame,
                 sim_params: dict,
                 base_freqs: dict,
                 target_years,
                 population_size: int = 20,
                 generations: int = 10,
                 n_simulations: int = 3,
                 n_jobs: int = -1,
                 use_parallel: bool = True,
                 optimize_detection: bool = True,
                 verbose: bool = True,
                 use_cache: bool = True):

        self.real_data = real_data
        self.sim_params = sim_params
        self.base_freqs = base_freqs.copy()
        self.target_years = target_years
        self.population_size = population_size
        self.generations = generations
        self.n_simulations = max(2, n_simulations)
        self.use_parallel = use_parallel
        self.n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs
        self.optimize_detection = optimize_detection
        self.verbose = verbose
        self.use_cache = use_cache

        # Предвычисленные целевые значения
        self.targets = {}
        self.target_weights = {}
        for i, year in enumerate(target_years):
            year_data = real_data[real_data['years'] == year]
            if len(year_data) > 0:
                self.targets[year] = year_data['prevalence_0_49'].iloc[0]
                self.target_weights[year] = 1.0 + 0.1 * i

        self.sim_cfg = {
            'pop_size': sim_params['initial_population_size'],
            'max_age': sim_params['max_age_limit'],
            'base_year': sim_params['base_year']
        }

        self.base_detection_year = min(target_years) if target_years else 2012

        # Кэш для результатов оценки
        self.fitness_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        self.history = []
        self.best_fitness_ever = float('inf')

    def _individual_to_key(self, individual: Dict) -> str:
        """Быстрое создание ключа для особи"""
        genes_key = tuple(sorted((k, round(v, 6)) for k, v in individual['genes'].items()))
        detection_key = tuple(sorted((k, round(v, 6)) for k, v in individual['detection'].items()))
        return hashlib.md5(str((genes_key, detection_key)).encode()).hexdigest()

    def _normalize_frequencies(self, freqs: Dict[str, float]) -> Dict[str, float]:
        """Быстрая нормализация частот"""
        freqs = freqs.copy()
        mutant_sum = sum(v for k, v in freqs.items() if k != 'N')

        if mutant_sum > 0.15:
            scale = 0.14 / (mutant_sum if abs(mutant_sum) > 1e-10 else 1.0)
            for k in freqs:
                if k != 'N':
                    freqs[k] *= scale
            mutant_sum = sum(v for k, v in freqs.items() if k != 'N')

        freqs['N'] = max(0.001, 1.0 - mutant_sum)
        return freqs

    def _detection_factor(self, detection_params: Dict, year: int) -> float:
        """Быстрый расчет фактора диагностики"""
        slope = detection_params['slope']
        intercept = detection_params['intercept']
        t = year - self.base_detection_year
        return np.clip(slope * t + intercept, 0.3, 1.0)

    def _create_individual(self) -> Dict:
        """Быстрое создание особи"""
        genetic_part = self.base_freqs.copy()

        for mutation, (low, high) in self.BOUNDS.items():
            if mutation in genetic_part:
                if np.random.random() < 0.7:
                    # Логнормальное распределение
                    log_low = np.log(max(low, 1e-10))
                    log_high = np.log(high)
                    genetic_part[mutation] = np.exp(np.random.uniform(log_low, log_high))
                else:
                    genetic_part[mutation] = np.random.uniform(low, high)

        genetic_part = self._normalize_frequencies(genetic_part)

        if self.optimize_detection:
            diagnostic_part = {
                'slope': np.random.uniform(*self.DETECTION_BOUNDS['slope']),
                'intercept': np.random.uniform(*self.DETECTION_BOUNDS['intercept'])
            }
        else:
            diagnostic_part = {'slope': 0.02, 'intercept': 0.6}

        return {'genes': genetic_part, 'detection': diagnostic_part}

    def _mutate(self, individual: Dict, generation: int) -> Dict:
        """Быстрая мутация"""
        mutated = {
            'genes': individual['genes'].copy(),
            'detection': individual['detection'].copy()
        }

        progress = generation / max(1, self.generations)
        sigma = 0.2 * (1 - 0.7 * progress)
        mutation_rate = 0.3 * (1 - 0.5 * progress)

        # Мутация генов
        for mutation, (low, high) in self.BOUNDS.items():
            if mutation in mutated['genes'] and np.random.random() < mutation_rate:
                current = mutated['genes'][mutation]
                if np.random.random() < 0.8:
                    safe_current = max(current, low * 0.1)
                    log_current = np.log(safe_current)
                    log_new = log_current + np.random.normal(0, sigma)
                    mutated['genes'][mutation] = np.clip(np.exp(log_new), low, high)
                else:
                    mutated['genes'][mutation] = np.random.uniform(low, high)

        mutated['genes'] = self._normalize_frequencies(mutated['genes'])

        # Мутация диагностики
        if self.optimize_detection and np.random.random() < mutation_rate:
            if np.random.random() < 0.5:
                current = mutated['detection']['slope']
                low, high = self.DETECTION_BOUNDS['slope']
                mutated['detection']['slope'] = np.clip(
                    current + np.random.normal(0, sigma * 0.02), low, high
                )
            else:
                current = mutated['detection']['intercept']
                low, high = self.DETECTION_BOUNDS['intercept']
                mutated['detection']['intercept'] = np.clip(
                    current + np.random.normal(0, sigma * 0.1), low, high
                )

        return mutated

    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Быстрый кроссовер"""
        child_genes = {}
        alpha = 0.3

        for mutation in self.BOUNDS.keys():
            if mutation in parent1['genes'] and mutation in parent2['genes']:
                p1 = parent1['genes'][mutation]
                p2 = parent2['genes'][mutation]
                d = abs(p1 - p2)
                low = min(p1, p2) - alpha * d
                high = max(p1, p2) + alpha * d

                bounds_low, bounds_high = self.BOUNDS[mutation]
                low = max(low, bounds_low)
                high = min(high, bounds_high)

                if low > high:
                    low, high = high, low

                child_genes[mutation] = np.random.uniform(low, high)

        child_genes = self._normalize_frequencies(child_genes)

        if self.optimize_detection:
            child_detection = {}
            for param in ['slope', 'intercept']:
                p1 = parent1['detection'][param]
                p2 = parent2['detection'][param]
                d = abs(p1 - p2)
                low = min(p1, p2) - 0.4 * d
                high = max(p1, p2) + 0.4 * d

                bounds_low, bounds_high = self.DETECTION_BOUNDS[param]
                low = max(low, bounds_low)
                high = min(high, bounds_high)

                if low > high:
                    low, high = high, low

                child_detection[param] = np.random.uniform(low, high)
        else:
            child_detection = parent1['detection'].copy()

        return {'genes': child_genes, 'detection': child_detection}

    def _evaluate_individual_batch(self, individuals: List[Dict]) -> List[float]:
        """Пакетная оценка особей"""
        fitness_scores = []

        for individual in individuals:
            # Проверка кэша
            if self.use_cache:
                key = self._individual_to_key(individual)
                if key in self.fitness_cache:
                    self.cache_hits += 1
                    fitness_scores.append(self.fitness_cache[key])
                    continue
                self.cache_misses += 1

            # Оценка без детального прогресс-бара
            freqs = individual['genes']
            detection_params = individual['detection']

            yearly_results = defaultdict(list)

            # Запускаем симуляции
            for _ in range(self.n_simulations):
                for year in self.target_years:
                    if year not in self.targets:
                        continue

                    sim = GenerationSimulation(
                        initial_population_size=self.sim_cfg['pop_size'],
                        max_age_limit=self.sim_cfg['max_age'],
                        simulation_years=year - self.sim_cfg['base_year'],
                        validation_year=year,
                        base_year=self.sim_cfg['base_year'],
                        mutation_frequencies=freqs.copy(),
                        verbose=False,
                        use_cache=self.use_cache
                    )

                    results = sim.run_validation_simulation(target_year=year)
                    if results and results.get('validation_point_reached'):
                        yearly_results[year].append(results.get('final_prevalence', 0.0))

            # Вычисление fitness
            fitness = self._compute_fitness(yearly_results, freqs, detection_params)

            if self.use_cache:
                self.fitness_cache[key] = fitness
                if len(self.fitness_cache) > 1000:
                    # Очистка кэша при переполнении
                    keys_to_remove = list(self.fitness_cache.keys())[:200]
                    for k in keys_to_remove:
                        del self.fitness_cache[k]

            fitness_scores.append(fitness)

        return fitness_scores

    def _compute_fitness(self, yearly_results: Dict, freqs: Dict, detection_params: Dict) -> float:
        """Быстрое вычисление fitness"""
        total_error = 0
        n_successful_years = 0

        for year in self.target_years:
            if year not in self.targets or year not in yearly_results or not yearly_results[year]:
                continue

            true_prevalence = np.median(yearly_results[year])
            detection_factor = self._detection_factor(detection_params, year)
            observed_prevalence = true_prevalence * detection_factor
            real_value = self.targets[year]
            weight = self.target_weights.get(year, 1.0)

            if abs(real_value) > 1e-10:
                rel_error_squared = ((observed_prevalence - real_value) / real_value) ** 2
            else:
                rel_error_squared = (observed_prevalence - real_value) ** 2

            total_error += weight * rel_error_squared
            n_successful_years += 1

        if n_successful_years == 0:
            return 100.0

        # Штрафы
        mutant_sum = sum(v for k, v in freqs.items() if k != 'N')
        if mutant_sum > 0.15:
            total_error += (mutant_sum - 0.15) * 10

        return total_error / n_successful_years

    def _tournament_selection(self, population: List[Dict], fitness_scores: List[float]) -> Dict:
        """Турнирная селекция"""
        tournament_size = min(4, len(population))
        idx = np.random.choice(len(population), tournament_size, replace=False)
        winner_idx = idx[np.argmin([fitness_scores[i] for i in idx])]
        return {
            'genes': population[winner_idx]['genes'].copy(),
            'detection': population[winner_idx]['detection'].copy()
        }

    def optimize(self) -> Dict:
        """Основной метод оптимизации"""
        print("\nСоздание начальной популяции...")

        # Создаем популяцию
        population = [self._create_individual() for _ in range(self.population_size)]

        # Основной цикл
        for generation in range(self.generations):
            gen_start = time.time()

            # Пакетная оценка
            fitness_scores = self._evaluate_individual_batch(population)

            # Поиск лучшей особи
            best_idx = np.argmin(fitness_scores)
            best_fitness = fitness_scores[best_idx]
            best_individual = {
                'genes': population[best_idx]['genes'].copy(),
                'detection': population[best_idx]['detection'].copy()
            }

            mean_fitness = np.mean(fitness_scores)
            std_fitness = np.std(fitness_scores)

            # Сохраняем историю
            self.history.append({
                'generation': generation,
                'best_fitness': best_fitness,
                'best_individual': best_individual.copy(),
                'mean_fitness': mean_fitness,
                'std_fitness': std_fitness,
                'time': time.time() - gen_start
            })

            # Прогресс
            m694v = best_individual['genes'].get('M694V', 0)
            print(f"\nПоколение {generation + 1}/{self.generations}")
            print(f"  Лучший fitness: {best_fitness:.6f}")
            print(f"  M694V: {m694v:.4f}")
            print(f"  Средний fitness: {mean_fitness:.6f} ± {std_fitness:.6f}")

            if self.optimize_detection:
                slope = best_individual['detection']['slope']
                intercept = best_individual['detection']['intercept']
                print(f"  Диагностика: y = {slope:.3f}·t + {intercept:.2f}")

            if generation == self.generations - 1:
                break

            # Элитизм
            elite_size = max(2, self.population_size // 10)
            sorted_idx = np.argsort(fitness_scores)
            elite = [population[i] for i in sorted_idx[:elite_size]]

            # Создание нового поколения
            new_population = elite.copy()

            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)

                if np.random.random() < 0.7:
                    child = self._crossover(parent1, parent2)
                else:
                    child = {'genes': parent1['genes'].copy(), 'detection': parent1['detection'].copy()}

                child = self._mutate(child, generation)
                new_population.append(child)

            population = new_population

        # Финальный отчет
        print(f"\nСтатистика кэша: Хиты={self.cache_hits}, Промахи={self.cache_misses}, "
              f"Hit ratio={self.cache_hits / (self.cache_hits + self.cache_misses) * 100 if self.cache_hits + self.cache_misses > 0 else 0:.1f}%")

        return {
            'best_individual': self.history[-1]['best_individual'],
            'history': self.history,
            'fitness': self.history[-1]['best_fitness']
        }


# ========== УЛУЧШЕННАЯ ВЕРСИЯ ==========
class GeneticOptimizerWithMetrics:
    """
    Улучшенный GeneticOptimizer с:
    - Доверительными интервалами
    - Кросс-валидацией по годам
    - Правильным train/test разделением
    - Новыми границами диагностики
    """

    # Новые границы диагностики (intercept пониже для запаса роста)
    DETECTION_BOUNDS = {
        'slope': (0.0, 0.05),
        'intercept': (0.4, 0.7),  # Было (0.3, 0.9), стало (0.4, 0.7)
    }

    def __init__(self,
                 real_data: pd.DataFrame,
                 sim_params: dict,
                 base_freqs: dict,
                 train_years: list = [2012, 2014, 2016, 2018],
                 test_years: list = [2020, 2022, 2024],
                 confidence_level: float = 0.95,
                 n_folds: int = 3,
                 population_size: int = 15,
                 generations: int = 10,
                 n_simulations: int = 3,
                 optimize_detection: bool = True,
                 verbose: bool = True,
                 use_cache: bool = True):

        self.real_data = real_data
        self.sim_params = sim_params
        self.base_freqs = base_freqs.copy()
        self.train_years = train_years
        self.test_years = test_years
        self.confidence_level = confidence_level
        self.n_folds = n_folds
        self.population_size = population_size
        self.generations = generations
        self.n_simulations = n_simulations
        self.optimize_detection = optimize_detection
        self.verbose = verbose
        self.use_cache = use_cache

        # Обновляем границы диагностики в оригинальном классе
        GeneticOptimizer.DETECTION_BOUNDS = self.DETECTION_BOUNDS

        # Инициализируем оригинальный оптимизатор
        self.optimizer = GeneticOptimizer(
            real_data=real_data,
            sim_params=sim_params,
            base_freqs=base_freqs,
            target_years=train_years,  # Обучаем только на train годах
            population_size=population_size,
            generations=generations,
            n_simulations=n_simulations,
            optimize_detection=optimize_detection,
            verbose=verbose,
            use_cache=use_cache
        )

        # Вычисляем доверительные интервалы
        self.confidence_intervals = self._compute_confidence_intervals()

        print(f"\n{'=' * 80}")
        print("🔬 GENETIC OPTIMIZER WITH METRICS")
        print(f"{'=' * 80}")
        print(f"📅 Train годы: {train_years}")
        print(f"🔮 Test годы: {test_years}")
        print(f"📊 Диагностика: intercept {self.DETECTION_BOUNDS['intercept']}")
        print(f"📈 Уровень доверия: {confidence_level:.0%}")
        print(f"🔄 Кросс-валидация: {n_folds} фолдов")
        print(f"{'=' * 80}\n")

    def _compute_confidence_intervals(self) -> dict:
        """Вычисление доверительных интервалов для реальных данных"""
        intervals = {}

        for year in self.real_data['years'].unique():
            year_data = self.real_data[self.real_data['years'] == year]

            if len(year_data) == 0:
                continue

            registered = year_data['Registered_0-49'].iloc[0]
            population = year_data['0-49 population_Total'].iloc[0]
            prevalence = year_data['prevalence_0_49'].iloc[0]

            # Стандартная ошибка для биномиальной пропорции
            standard_error = np.sqrt(prevalence * (1 - prevalence) / population) if population > 0 else 0

            # z-значение для доверительного интервала
            z_score = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)

            # Доверительный интервал
            margin = z_score * standard_error
            ci_lower = max(0, prevalence - margin)
            ci_upper = min(1, prevalence + margin)

            # Коэффициент вариации (качество данных)
            cv = standard_error / prevalence if prevalence > 0 else 1.0
            data_quality = 1.0 / (1.0 + 5.0 * cv)  # Шкала от 0 до 1

            intervals[year] = {
                'prevalence': prevalence,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'standard_error': standard_error,
                'cv': cv,
                'data_quality': data_quality,
                'population': population,
                'registered': registered,
                'margin': margin
            }

            if self.verbose:
                print(f"   Год {year}: прев.={prevalence:.6f} "
                      f"CI=[{ci_lower:.6f}, {ci_upper:.6f}] "
                      f"качество={data_quality:.2f}")

        return intervals

    def _weighted_fitness(self,
                          yearly_results: Dict,
                          individual: Dict) -> float:
        """
        Взвешенная fitness с учетом качества данных
        """
        total_weighted_error = 0
        total_weight = 0

        for year in self.train_years:
            if year not in yearly_results or not yearly_results[year]:
                continue

            # Модельные значения
            true_prevalence = np.median(yearly_results[year])
            detection_factor = self.optimizer._detection_factor(individual['detection'], year)
            observed_prevalence = true_prevalence * detection_factor

            # Реальные данные с доверительным интервалом
            ci_data = self.confidence_intervals.get(year)
            if ci_data is None:
                continue

            real_value = ci_data['prevalence']
            ci_lower = ci_data['ci_lower']
            ci_upper = ci_data['ci_upper']
            data_quality = ci_data['data_quality']

            # Проверяем, попадает ли модель в доверительный интервал
            in_ci = ci_lower <= observed_prevalence <= ci_upper

            # Базовая ошибка (относительная)
            if abs(real_value) > 1e-10:
                rel_error = (observed_prevalence - real_value) / real_value
                rel_error_squared = rel_error ** 2
            else:
                rel_error_squared = (observed_prevalence - real_value) ** 2

            # Штраф за выход за пределы CI
            if not in_ci:
                distance_to_ci = min(
                    abs(observed_prevalence - ci_lower),
                    abs(observed_prevalence - ci_upper)
                )
                ci_penalty = (distance_to_ci / ci_data['standard_error']) ** 2 if ci_data['standard_error'] > 0 else 100
                rel_error_squared += ci_penalty

            # Вес: базовый вес года * качество данных
            weight = self.optimizer.target_weights.get(year, 1.0) * data_quality

            total_weighted_error += weight * rel_error_squared
            total_weight += weight

        if total_weight == 0:
            return 100.0

        # Базовый fitness
        fitness = total_weighted_error / total_weight

        # Штраф за нереалистичные частоты
        mutant_sum = sum(v for k, v in individual['genes'].items() if k != 'N')
        if mutant_sum > 0.15:
            fitness += (mutant_sum - 0.15) * 10

        # Штраф за нереалистичную диагностику
        if individual['detection']['intercept'] > 0.8:
            fitness += (individual['detection']['intercept'] - 0.8) * 50
        if individual['detection']['intercept'] < 0.3:
            fitness += (0.3 - individual['detection']['intercept']) * 50

        return fitness

    def _evaluate_with_ci(self, individual: Dict) -> Dict:
        """
        Оценка особи с учетом доверительных интервалов
        """
        yearly_results = defaultdict(list)

        # Запускаем симуляции
        for _ in range(self.n_simulations):
            for year in self.train_years:
                if year not in self.confidence_intervals:
                    continue

                sim = GenerationSimulation(
                    initial_population_size=self.optimizer.sim_cfg['pop_size'],
                    max_age_limit=self.optimizer.sim_cfg['max_age'],
                    simulation_years=year - self.optimizer.sim_cfg['base_year'],
                    validation_year=year,
                    base_year=self.optimizer.sim_cfg['base_year'],
                    mutation_frequencies=individual['genes'].copy(),
                    verbose=False,
                    use_cache=self.use_cache
                )

                results = sim.run_validation_simulation(target_year=year)
                if results and results.get('validation_point_reached'):
                    yearly_results[year].append(results.get('final_prevalence', 0.0))

        # Вычисляем fitness
        fitness = self._weighted_fitness(yearly_results, individual)

        # Собираем метрики по годам
        year_metrics = {}
        for year in self.train_years:
            if year in yearly_results and yearly_results[year]:
                model_prev = np.median(yearly_results[year])
                det_factor = self.optimizer._detection_factor(individual['detection'], year)
                observed = model_prev * det_factor

                ci_data = self.confidence_intervals.get(year, {})
                real = ci_data.get('prevalence', 0)
                in_ci = ci_data.get('ci_lower', 0) <= observed <= ci_data.get('ci_upper', 1)

                error_pct = abs(observed - real) / real * 100 if real > 0 else 0

                year_metrics[year] = {
                    'real': real,
                    'observed': observed,
                    'biological': model_prev,
                    'detection': det_factor,
                    'error_pct': error_pct,
                    'in_ci': in_ci
                }

        return {
            'fitness': fitness,
            'year_metrics': year_metrics,
            'individual': individual
        }

    def _cross_validate(self, individual: Dict) -> dict:
        """
        Кросс-валидация особи по годам
        """
        # Создаем фолды для кросс-валидации
        years = sorted(self.train_years)
        fold_size = len(years) // self.n_folds
        folds = []

        for i in range(self.n_folds):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < self.n_folds - 1 else len(years)
            test_years = years[start_idx:end_idx]
            train_years = [y for y in years if y not in test_years]
            folds.append({'train': train_years, 'test': test_years})

        if self.verbose:
            print(f"\n🔄 Кросс-валидация ({self.n_folds} фолдов):")

        cv_scores = []
        fold_results = []

        for fold_idx, fold in enumerate(folds):
            # Оценка на train
            train_results = defaultdict(list)
            for year in fold['train']:
                for _ in range(self.n_simulations):
                    sim = GenerationSimulation(
                        initial_population_size=self.optimizer.sim_cfg['pop_size'],
                        max_age_limit=self.optimizer.sim_cfg['max_age'],
                        simulation_years=year - self.optimizer.sim_cfg['base_year'],
                        validation_year=year,
                        base_year=self.optimizer.sim_cfg['base_year'],
                        mutation_frequencies=individual['genes'].copy(),
                        verbose=False,
                        use_cache=self.use_cache
                    )
                    results = sim.run_validation_simulation(target_year=year)
                    if results and results.get('validation_point_reached'):
                        train_results[year].append(results.get('final_prevalence', 0.0))

            # Временно меняем train_years для вычисления fitness
            original_train_years = self.train_years
            self.train_years = fold['train']
            train_fitness = self._weighted_fitness(train_results, individual)
            self.train_years = original_train_years

            # Оценка на test
            test_errors = []
            test_year_metrics = {}

            for year in fold['test']:
                ci_data = self.confidence_intervals.get(year, {})
                real_value = ci_data.get('prevalence', 0)

                test_prev = []
                for _ in range(self.n_simulations):
                    sim = GenerationSimulation(
                        initial_population_size=self.optimizer.sim_cfg['pop_size'],
                        max_age_limit=self.optimizer.sim_cfg['max_age'],
                        simulation_years=year - self.optimizer.sim_cfg['base_year'],
                        validation_year=year,
                        base_year=self.optimizer.sim_cfg['base_year'],
                        mutation_frequencies=individual['genes'].copy(),
                        verbose=False,
                        use_cache=self.use_cache
                    )
                    results = sim.run_validation_simulation(target_year=year)
                    if results and results.get('validation_point_reached'):
                        test_prev.append(results.get('final_prevalence', 0.0))

                if test_prev:
                    model_prev = np.median(test_prev)
                    det_factor = self.optimizer._detection_factor(individual['detection'], year)
                    observed = model_prev * det_factor

                    error_pct = abs(observed - real_value) / real_value * 100 if real_value > 0 else 0
                    test_errors.append(error_pct)

                    test_year_metrics[year] = {
                        'real': real_value,
                        'observed': observed,
                        'error_pct': error_pct,
                        'in_ci': ci_data.get('ci_lower', 0) <= observed <= ci_data.get('ci_upper', 1)
                    }

            avg_test_error = np.mean(test_errors) if test_errors else float('inf')
            cv_scores.append(avg_test_error)

            fold_results.append({
                'fold': fold_idx + 1,
                'train_years': fold['train'],
                'test_years': fold['test'],
                'train_fitness': train_fitness,
                'test_error': avg_test_error,
                'test_errors': test_errors,
                'test_year_metrics': test_year_metrics
            })

            if self.verbose:
                print(f"   Фолд {fold_idx + 1}: train={fold['train']}, "
                      f"test={fold['test']}, ошибка={avg_test_error:.2f}%")

        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)

        if self.verbose:
            print(f"\n   CV среднее: {cv_mean:.2f}% ± {cv_std:.2f}%")

        return {
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'cv_scores': cv_scores,
            'fold_results': fold_results
        }

    def _test_on_future(self, individual: Dict) -> dict:
        """
        Тестирование на будущих годах (которые модель не видела)
        """
        if self.verbose:
            print(f"\n🔮 ТЕСТИРОВАНИЕ НА БУДУЩИХ ГОДАХ: {self.test_years}")

        test_results = {}

        for year in self.test_years:
            if year not in self.real_data['years'].values:
                if self.verbose:
                    print(f"   ⚠️ Год {year} отсутствует в данных")
                continue

            real_value = self.real_data[self.real_data['years'] == year]['prevalence_0_49'].iloc[0]
            ci_data = self.confidence_intervals.get(year, {})

            # Запускаем симуляции (больше для надежности)
            prevalences = []
            for _ in range(self.n_simulations * 2):
                sim = GenerationSimulation(
                    initial_population_size=self.optimizer.sim_cfg['pop_size'],
                    max_age_limit=self.optimizer.sim_cfg['max_age'],
                    simulation_years=year - self.optimizer.sim_cfg['base_year'],
                    validation_year=year,
                    base_year=self.optimizer.sim_cfg['base_year'],
                    mutation_frequencies=individual['genes'].copy(),
                    verbose=False
                )
                res = sim.run_validation_simulation(target_year=year)
                if res and res.get('validation_point_reached'):
                    prevalences.append(res.get('final_prevalence', 0.0))

            if prevalences:
                true_prev = np.mean(prevalences)
                std_prev = np.std(prevalences)
                det_factor = self.optimizer._detection_factor(individual['detection'], year)
                obs_prev = true_prev * det_factor

                error_pct = abs(obs_prev - real_value) / real_value * 100 if real_value > 0 else 0

                # Проверка попадания в доверительный интервал
                in_ci = False
                if ci_data:
                    in_ci = ci_data.get('ci_lower', 0) <= obs_prev <= ci_data.get('ci_upper', 1)

                test_results[year] = {
                    'real': real_value,
                    'observed': obs_prev,
                    'biological': true_prev,
                    'std': std_prev,
                    'detection': det_factor,
                    'error_pct': error_pct,
                    'in_ci': in_ci,
                    'ci_lower': ci_data.get('ci_lower', None),
                    'ci_upper': ci_data.get('ci_upper', None)
                }

                if self.verbose:
                    status = "✅" if in_ci else "❌"
                    print(f"   {year}: ошибка {error_pct:.2f}% {status}")

        return test_results

    def optimize(self) -> Dict:
        """
        Запуск улучшенной оптимизации
        """
        print(f"\n{'=' * 80}")
        print("🚀 ЗАПУСК ОПТИМИЗАЦИИ")
        print(f"{'=' * 80}")

        # Запускаем оригинальную оптимизацию
        start_time = time.time()
        opt_results = self.optimizer.optimize()
        opt_time = time.time() - start_time

        best_individual = opt_results['best_individual']

        # Дополнительная оценка лучшей особи
        print(f"\n📊 ОЦЕНКА ЛУЧШЕЙ ОСОБИ:")
        evaluation = self._evaluate_with_ci(best_individual)

        # Кросс-валидация
        cv_results = self._cross_validate(best_individual)

        # Тестирование на будущих годах
        test_results = self._test_on_future(best_individual)

        # Собираем все результаты
        enhanced_results = {
            'best_individual': best_individual,
            'history': opt_results['history'],
            'fitness': opt_results['fitness'],
            'evaluation': evaluation,
            'cv_results': cv_results,
            'test_results': test_results,
            'confidence_intervals': self.confidence_intervals,
            'train_years': self.train_years,
            'test_years': self.test_years,
            'optimization_time': opt_time,
            'optimizer_stats': {
                'cache_hits': self.optimizer.cache_hits,
                'cache_misses': self.optimizer.cache_misses,
                'cache_ratio': self.optimizer.cache_hits / (
                            self.optimizer.cache_hits + self.optimizer.cache_misses) * 100 if (
                                                                                                          self.optimizer.cache_hits + self.optimizer.cache_misses) > 0 else 0
            }
        }

        # Финальный отчет
        self._print_final_report(enhanced_results)

        return enhanced_results

    def _print_final_report(self, results: Dict):
        """
        Печать итогового отчета
        """
        from tabulate import tabulate

        print(f"\n{'=' * 100}")
        print(" " * 35 + "📊 ИТОГОВЫЙ ОТЧЕТ")
        print(f"{'=' * 100}")

        best = results['best_individual']

        # 1. Параметры диагностики
        print(f"\n📈 ОПТИМИЗИРОВАННАЯ ДИАГНОСТИКА:")
        print(f"   intercept (базовая) = {best['detection']['intercept']:.3f}")
        print(f"   slope (скорость роста) = {best['detection']['slope']:.3f}")

        # Формула
        base_year = self.optimizer.base_detection_year
        print(
            f"   Формула: detection(t) = {best['detection']['slope']:.3f}·(t-{base_year}) + {best['detection']['intercept']:.2f}")

        # Таблица диагностики
        print(f"\n📅 ВЫЯВЛЯЕМОСТЬ ПО ГОДАМ:")
        det_table = []
        for year in [2012, 2014, 2016, 2018, 2020, 2022, 2024]:
            det = best['detection']['slope'] * (year - base_year) + best['detection']['intercept']
            det = max(0.3, min(1.0, det))
            det_table.append([year, f"{det:.1%}"])

        print(tabulate(det_table, headers=["Год", "Выявляемость"], tablefmt="simple"))

        # 2. Результаты на обучающих годах
        if 'evaluation' in results and results['evaluation']['year_metrics']:
            print(f"\n📚 ОБУЧАЮЩИЕ ГОДЫ ({self.train_years}):")
            train_table = []
            for year, metrics in results['evaluation']['year_metrics'].items():
                status = "✅" if metrics.get('in_ci', False) else "❌"
                train_table.append([
                    year,
                    f"{metrics['real']:.6f}",
                    f"{metrics['observed']:.6f}",
                    f"{metrics['error_pct']:.2f}%",
                    status
                ])

            headers = ["Год", "Реальная", "Модель", "Ошибка", "В CI"]
            print(tabulate(train_table, headers=headers, tablefmt="simple"))

        # 3. Результаты кросс-валидации
        if 'cv_results' in results:
            cv = results['cv_results']
            print(f"\n📊 КРОСС-ВАЛИДАЦИЯ:")
            print(f"   Средняя ошибка CV: {cv['cv_mean']:.2f}% ± {cv['cv_std']:.2f}%")

            cv_table = []
            for fold in cv['fold_results']:
                cv_table.append([
                    f"Фолд {fold['fold']}",
                    f"{fold['train_years']}",
                    f"{fold['test_years']}",
                    f"{fold['test_error']:.2f}%"
                ])

            headers = ["Фолд", "Train", "Test", "Ошибка"]
            print(tabulate(cv_table, headers=headers, tablefmt="simple"))

        # 4. Результаты на тестовых годах
        if 'test_results' in results and results['test_results']:
            print(f"\n🔮 ТЕСТОВЫЕ ГОДЫ ({self.test_years}):")
            test_table = []
            test_errors = []
            for year, res in results['test_results'].items():
                status = "✅" if res.get('in_ci', False) else "❌"
                test_table.append([
                    year,
                    f"{res['real']:.6f}",
                    f"{res['observed']:.6f}",
                    f"{res['error_pct']:.2f}%",
                    f"{res['detection']:.1%}",
                    status
                ])
                test_errors.append(res['error_pct'])

            headers = ["Год", "Реальная", "Модель", "Ошибка", "Диагн.", "В CI"]
            print(tabulate(test_table, headers=headers, tablefmt="simple"))

            if test_errors:
                print(f"\n   Средняя ошибка на тесте: {np.mean(test_errors):.2f}%")
                print(f"   Медианная ошибка: {np.median(test_errors):.2f}%")

        # 5. Гены
        print(f"\n🧬 ОПТИМИЗИРОВАННЫЕ ГЕНЫ:")
        gene_table = []
        for mutation in ['M694V', 'V726A', 'M680I', 'R761H']:
            opt_val = best['genes'].get(mutation, 0)
            base_val = self.base_freqs.get(mutation, 0)
            change = ((opt_val - base_val) / base_val * 100) if base_val > 0 else float('inf')

            bounds = GeneticOptimizer.BOUNDS.get(mutation, (0, 1))
            in_bounds = bounds[0] <= opt_val <= bounds[1]
            status = "✅" if in_bounds else "⚠️"

            change_str = f"{change:+.1f}%" if base_val > 0 else "новый"
            gene_table.append([
                mutation,
                f"{opt_val:.6f}",
                f"{base_val:.6f}",
                change_str,
                status
            ])

        headers = ["Мутация", "Оптимиз.", "Исходно", "Изменение", "Статус"]
        print(tabulate(gene_table, headers=headers, tablefmt="simple"))

        # 6. Итоговая оценка
        print(f"\n{'=' * 100}")
        print("🎯 ИТОГОВАЯ ОЦЕНКА МОДЕЛИ")
        print(f"{'=' * 100}")

        # Оценка по тестовым данным
        if test_errors:
            avg_test_error = np.mean(test_errors)

            if avg_test_error < 15:
                rating = "🏆 ОТЛИЧНО"
                desc = "Модель отлично прогнозирует будущие годы"
            elif avg_test_error < 25:
                rating = "👍 ХОРОШО"
                desc = "Модель хорошо прогнозирует будущие годы"
            elif avg_test_error < 35:
                rating = "👌 УДОВЛЕТВОРИТЕЛЬНО"
                desc = "Модель приемлемо прогнозирует"
            else:
                rating = "👎 ПЛОХО"
                desc = "Модель плохо прогнозирует"

            print(f"\n  {rating}")
            print(f"  {desc}")
            print(f"\n  📊 Средняя ошибка на тесте: {avg_test_error:.2f}%")

        # Оценка стабильности
        if 'cv_results' in results:
            cv_mean = results['cv_results']['cv_mean']
            cv_std = results['cv_results']['cv_std']

            if cv_std < cv_mean * 0.3:
                stability = "✅ СТАБИЛЬНА"
            else:
                stability = "⚠️ НЕСТАБИЛЬНА"

            print(f"  📈 Стабильность модели: {stability} (CV={cv_mean:.2f}%±{cv_std:.2f}%)")

        # Время и кэш
        print(f"\n  ⏱️  Время оптимизации: {results['optimization_time']:.1f} сек")

        stats = results['optimizer_stats']
        print(f"  💾 Кэш: {stats['cache_ratio']:.1f}% попаданий "
              f"({stats['cache_hits']} хитов, {stats['cache_misses']} промахов)")

        print(f"\n{'=' * 100}")
        print("✅ РАБОТА ЗАВЕРШЕНА")
        print(f"{'=' * 100}")


def run_optimization_with_metrics(mode: str = 'full'):
    """
    Запуск оптимизации с метриками
    """
    # Загружаем данные
    df = pd.read_csv('FMF_data2.csv')
    df['prevalence_0_49'] = df['Registered_0-49'] / df['0-49 population_Total']

    print(f"\n{'=' * 80}")
    print("🔬 ЗАПУСК ОПТИМИЗАЦИИ С МЕТРИКАМИ")
    print(f"{'=' * 80}")

    if mode == 'quick':
        # Быстрый режим для тестирования
        optimizer = GeneticOptimizerWithMetrics(
            real_data=df,
            sim_params=SIM_PARAMS,
            base_freqs=MUTATION_FREQS,
            train_years=[2012, 2014],
            test_years=[2016, 2018],
            population_size=8,
            generations=4,
            n_simulations=2,
            n_folds=2,
            verbose=True
        )
    elif mode == 'full':
        # Полный режим с вашими рекомендациями
        optimizer = GeneticOptimizerWithMetrics(
            real_data=df,
            sim_params=SIM_PARAMS,
            base_freqs=MUTATION_FREQS,
            train_years=[2012, 2014, 2016, 2018],  # Обучаем на росте
            test_years=[2020, 2022, 2024],  # Тестируем будущее
            population_size=15,
            generations=10,
            n_simulations=3,
            n_folds=3,
            confidence_level=0.95,
            optimize_detection=True,
            verbose=True,
            use_cache=True
        )
    else:
        # Кастомный режим
        optimizer = GeneticOptimizerWithMetrics(
            real_data=df,
            sim_params=SIM_PARAMS,
            base_freqs=MUTATION_FREQS,
            train_years=[2012, 2014, 2016, 2018],
            test_years=[2020, 2022],
            population_size=12,
            generations=8,
            n_simulations=3,
            n_folds=3,
            verbose=True
        )

    return optimizer.optimize()


# ========== ТОЧКА ВХОДА ==========
if __name__ == "__main__":
    import warnings

    warnings.filterwarnings('ignore')

    # Проверяем наличие необходимых классов
    try:
        from tabulate import tabulate
    except ImportError:
        print("Устанавливаем tabulate...")
        import subprocess

        subprocess.check_call(['pip', 'install', 'tabulate'])
        from tabulate import tabulate

    print("\n" + "═" * 80)
    print(" " * 20 + "🧬 FMF УЛУЧШЕННАЯ ОПТИМИЗАЦИЯ")
    print("═" * 80)

    print("\nВыберите режим:")
    print("[1] Быстрый тест (2012-2014 train, 2016-2018 test)")
    print("[2] Полная оптимизация (2012-2018 train, 2020-2024 test) [РЕКОМЕНДУЕТСЯ]")
    print("[3] Кастомный режим")

    choice = input("\nВаш выбор (1-3): ").strip()

    if choice == '1':
        results = run_optimization_with_metrics('quick')
    elif choice == '2':
        results = run_optimization_with_metrics('full')
    elif choice == '3':
        results = run_optimization_with_metrics('custom')
    else:
        results = run_optimization_with_metrics('full')

    # Сохраняем результаты
    # with open('optimization_results.json', 'w') as f:
    #     # Конвертируем numpy типы в обычные Python типы
    #     def convert_to_serializable(obj):
    #         if isinstance(obj, np.integer):
    #             return int(obj)
    #         elif isinstance(obj, np.floating):
    #             return float(obj)
    #         elif isinstance(obj, np.ndarray):
    #             return obj.tolist()
    #         elif isinstance(obj, dict):
    #             return {key: convert_to_serializable(value) for key, value in obj.items()}
    #         elif isinstance(obj, list):
    #             return [convert_to_serializable(item) for item in obj]
    #         else:
    #             return obj

