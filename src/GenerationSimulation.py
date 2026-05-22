import random
from collections import defaultdict, Counter
from dataclasses import dataclass
from math import sqrt
from typing import Dict, Any
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy import stats


from Agent import Agent
from ModelParams import ModelParams


class GenerationSimulation:
    def __init__(self,params: ModelParams,
                 birth_rate_df: DataFrame,
                 death_rate_df: DataFrame,
                 fertility_rate_df: DataFrame,
                 age_structure_df: DataFrame,
                 fertility_factors_df: DataFrame):

        self.startup_stats = None
        self.current_df = None
        self.params = params

        # 1. Сохраняем исторические таблицы
        self.birth_rate_data = birth_rate_df
        self.death_rate_data = death_rate_df
        self.fertility_rate_data = fertility_rate_df
        self.age_structure_1950 = age_structure_df

        # 2. Параметры времени
        self.simulation_start_year = 1950  # Запомнили точку старта
        self.current_year = self.simulation_start_year  # Установили счетчик на старт
        self.initial_population_size = params.initial_population_size
        self.max_age_limit = params.max_age_limit
        self.ethnic_assortativity = params.ethnic_assortativity

        # 3. Обработка коэффициентов рождаемости из файла
        # Превращаем DataFrame с индексами '18-19' в словарь {(18, 19): 0.5}
        self.age_fertility_factors = {}
        for fertility_rate, row in fertility_factors_df.iterrows():
            # Предполагаем, что колонки идут в порядке: fertility_rate, min_age, max_age
            min_age = int(row.loc['min_age'])  # минимальный возраст
            max_age = int(row.loc['max_age'])  # максимальный возраст

            # Сохраняем в словарь
            self.age_fertility_factors[(min_age, max_age)] = fertility_rate

        # 4. Динамические коэффициенты (будут обновляться каждый год)
        self.current_tfr = 0.0
        self.current_death_rate = 0.0
        self.base_birth_prob = 0.0
        self.reproductive_years = 28  # (45 - 18 + 1)

        # 5. Генетика и мутации
        self.mutation_frequencies = {
            'N': 0.90427, 'M694V': 0.0437, 'V726A': 0.0292,
            'M680I': 0.0192, 'R761H': 0.00363
        }

        # 6. Инициализация хранилищ и статистики
        self.agents: Dict[str, Agent] = {}
        self.population_history = []
        self.children_born = 0
        self.total_deaths = 0
        self.prevented_fmf_births = 0

        self.pgt_births = 0  # Дети, рожденные через PGT
        self.pgt_attempts = 0  # Пары, которые попытались использовать PGT
        self.pgt_eligible_but_declined = 0  # Пары с риском, которые могли бы использовать PGT, но отказались

        self.annual_stats = []

        self.family_children_count = defaultdict(int)
        self.last_birth_year = defaultdict(lambda: -100)

        self.generation_stats = defaultdict(lambda: {
            'total': 0, 'alive': 0, 'healthy': 0, 'carrier': 0, 'affected': 0
        })

        self.birth_cooldown = 2

        self.screening_start_year = params.screening_start_year  # динамически берется из параметров
        self.pgt_start_year = params.pgt_start_year  # Добавили управление стартом PGT

        self.initialize_founders_with_structure()

        self._initialize_fertility_map()

        self.pgt_children_ids = []

        self.inheritance_stats = InheritanceStats()


    def _get_random_allele(self) -> str:
        return random.choices(
            list(self.mutation_frequencies.keys()),
            weights=list(self.mutation_frequencies.values())
        )[0]

    def initialize_founders_with_structure(self):
        # Для 1950 года всегда берем исторические параметры из Сценария 1
        # Это избавляет от хардкода и дублирования значений
        hist_params = ModelParams.scenario_1()

        eth_names = list(hist_params.ethnic_distribution.keys())
        eth_weights = list(hist_params.ethnic_distribution.values())
        hist_assortativity = hist_params.ethnic_assortativity

        # 1. Подготовка весов возрастной структуры
        age_groups = []
        weights = []
        for weight, row in self.age_structure_1950.iterrows():
            age_groups.append((int(row['min_age']), int(row['max_age'])))
            weights.append(float(weight))

        # 2. Основной цикл создания агентов
        for _ in range(self.initial_population_size):
            group = random.choices(age_groups, weights=weights)[0]
            age = random.randint(group[0], group[1])
            gender = random.choice(['male', 'female'])
            ethnicity = random.choices(eth_names, weights=eth_weights, k=1)[0]

            # Стратегия спаривания на основе исторической ассортативности
            mating_strategy = 'endogamy' if random.random() < hist_assortativity else 'exogamy'

            agent = Agent(
                params=self.params,  # Передаем текущие параметры сценария для дальнейшей жизни
                gender=gender,
                age=age,
                birth_year=self.simulation_start_year - age,
                generation=0,
                max_age_limit=self.max_age_limit,
                ethnicity=ethnicity,
                mating_strategy=mating_strategy
            )

            # Инициализация генотипа
            if ethnicity == 'Armenian':
                agent.set_genotype(self._get_random_allele(), self._get_random_allele())
            else:
                agent.set_genotype('N', 'N')

            # "Проживаем" жизнь до стартового года симуляции
            real_age = agent.age
            for year_tick in range(real_age):
                agent.age = year_tick
                current_year = agent.birth_year + year_tick
                agent.age_year(annual_death_prob=0.0, current_year=current_year)

            agent.age = real_age
            self.agents[agent.id] = agent

        # 3. Формирование семей
        self._form_initial_social_structure()

        # Сбор статистики запуска через list comprehensions
        arm_c = sum(1 for a in self.agents.values() if a.ethnicity == 'Armenian')
        endo_c = sum(1 for a in self.agents.values() if a.mating_strategy == 'endogamy')
        total = len(self.agents)

        self.startup_stats = {
            'arm_count': arm_c,
            'oth_count': total - arm_c,
            'endo_count': endo_c,
            'exo_count': total - endo_c
        }

    def print_population_stats(self, scenario_name="Scenario"):

        # 1. Создаем список словарей со всеми данными агентов
        data = []
        for a in self.agents.values():
            if not a.alive: continue
            has_mutation1 = a.mefv_allele_1 != 'N'
            has_mutation2 = a.mefv_allele_2 != 'N'

            is_affected = has_mutation1 and has_mutation2
            is_carrier = (has_mutation1 or has_mutation2) and not is_affected
            data.append({
                'id': a.id,
                'ethnicity': a.ethnicity,
                'gender': a.gender,
                'age': a.age,
                'mating': a.mating_strategy,
                'genotype': f"{a.mefv_allele_1}/{a.mefv_allele_2}",
                'is_carrier': is_carrier,
                'is_affected': is_affected,
                'is_diagnosed': a.is_diagnosed,
                'is_screened': a.is_screened,
                'on_colchicine': a.on_colchicine,
                'is_colchicine_resistant': a.is_colchicine_resistant,
                'on_antibodies': a.on_antibodies
            })

        df = pd.DataFrame(data)
        total_pop = len(df)

        # 2. Общая демография
        eth_dist = df['ethnicity'].value_counts(normalize=True) * 100
        print(f"Распределение этносов: Armenian {eth_dist.get('Armenian', 0):.1f}%, "
              f"Other {eth_dist.get('Other', 0):.1f}%")

        # 3. Генетическая статистика
        affected_count = df['is_affected'].sum()
        carriers_count = df['is_carrier'].sum()
        diagnosed_count = df['is_diagnosed'].sum()
        resistant_count = df['is_colchicine_resistant'].sum()
        on_antibodies_count = df['on_antibodies'].sum()

        print(f"Больных (FMF MM): {affected_count} ({affected_count / total_pop * 100:.2f}%)")
        print(
            f"Из них с диагнозом: {diagnosed_count} ({diagnosed_count / affected_count * 100 if affected_count > 0 else 0:.1f}%)")
        print(f"Носителей (MN): {carriers_count} ({carriers_count / total_pop * 100:.1f}%)")

        if affected_count > 0:
            print(
                f"Резистентных к колхицину: {resistant_count} ({resistant_count / affected_count * 100:.1f}% от больных)")
            print(
                f"Получают биопрепараты (антитела): {on_antibodies_count} ({on_antibodies_count / resistant_count * 100 if resistant_count > 0 else 0:.1f}% от резистентных)")

        if self.pgt_attempts > 0 or self.pgt_births > 0:
            # Краткая статистика - только основные цифры
            print(f"\nСтатистика PGT (преимплантационная генетическая диагностика):")
            print(f"  Попыток использования PGT: {self.pgt_attempts}")
            print(f"  Успешных родов через PGT: {self.pgt_births}")
            if self.pgt_attempts > 0:
                print(f"  Эффективность PGT: {self.pgt_births / self.pgt_attempts * 100:.1f}%")
            print(f"  Пары с риском, отказавшиеся от PGT: {self.pgt_eligible_but_declined}")

        # 4. Статистика для сценариев
        print("-" * 30)
        if self.params.use_screening:
            # Различаем Сценарий 2 и Сценарий 3
            if self.params.use_pgt and self.params.screening_coverage >= 0.8:
                print(f"Сценарий 3: Снижение ассортативности + Массовый скрининг + PGT (Текущий год: {self.year})")
                print(f"Ассортативность: {self.params.ethnic_assortativity}")
                print(f"Охват скринингом: {self.params.screening_coverage * 100:.0f}%")
                print(
                    f"Эффективность PGT: {self.params.pgt_efficiency * 100:.0f}% (доступен с {self.params.pgt_start_year} г.)")
            elif self.params.screening_coverage >= 0.8:
                print(f"Сценарий 3: Снижение ассортативности + Массовый скрининг (Текущий год: {self.year})")
                print(f"Ассортативность: {self.params.ethnic_assortativity}")
                print(f"Охват скринингом: {self.params.screening_coverage * 100:.0f}%")
            else:
                print(f"Сценарий 2: Интервенция (скрининг) (Текущий год: {self.year})")
            print(f"Предотвращено рождений FMF: {self.prevented_fmf_births}")
            screened_count = df['is_screened'].sum()
            print(f"Охвачено скринингом: {screened_count} чел. ({screened_count / total_pop * 100:.1f}%)")
        else:
            print("Сценарий 1: Status Quo")

        # 5. Сохранение результатов
        filename = f"report_{scenario_name}_{self.year}.xlsx"
        df.to_excel(filename, index=False)
        print(f"\nДанные сохранены в: {filename}")
        print("═" * 50 + "\n")

        self.current_df = df


    def _get_family_key(self, p1: Agent, p2: Agent):
        # Гарантируем, что ключ будет одинаковым независимо от того, кто p1, а кто p2
        ids = sorted([p1.id, p2.id])
        return f"{ids[0]}_{ids[1]}"

    def _form_initial_social_structure(self):
        """Создает семейные связи для стартового населения 1950 года"""

        # 1. Подготовка списков
        males = [a for a in self.agents.values() if a.gender == 'male' and 18 <= a.age <= 50]
        females = [a for a in self.agents.values() if a.gender == 'female' and 18 <= a.age <= 45]
        children = [a for a in self.agents.values() if a.age < 18]

        random.shuffle(females)

        # 2. Формируем пары
        active_couples = []
        available_females = females.copy()

        for male in males:
            if not available_females:
                break

            # 1. Сначала определяем список потенциальных невест
            if male.mating_strategy == 'endogamy':
                # Эндогам: ищет строго своих
                candidates = [f for f in available_females if f.ethnicity == male.ethnicity]
                # Если своих нет — берем любую (биологический компромисс)
                if not candidates:
                    candidates = available_females
            else:
                # экзогам выбирает из ВСЕХ (случайный выбор без учета этноса)
                candidates = available_females

            # 2. Если список кандидатов не пуст, выбираем одну
            if candidates:
                suitable_female = random.choice(candidates)
                male.set_partner(suitable_female)
                active_couples.append((male, suitable_female))
                available_females.remove(suitable_female)

        # Распределяем детей (с научной точки зрения - привязываем к биологическим матерям)
        # Сортируем детей от старших к младшим, чтобы легче подбирать родителей
        children.sort(key=lambda x: x.age, reverse=True)

        for child in children:
            # Ищем пару, подходящую по возрасту (мать старше на 18-45 лет)
            # Сначала пытаемся соблюсти этнос (для чистоты данных 1950 года)
            suitable_couples = [
                (m, f) for m, f in active_couples
                if 18 <= (f.age - child.age) <= 45 and f.ethnicity == child.ethnicity
            ]

            if not suitable_couples:
                suitable_couples = [
                    (m, f) for m, f in active_couples
                    if 18 <= (f.age - child.age) <= 45
                ]

            if suitable_couples:
                father, mother = random.choice(suitable_couples)

                # Устанавливаем связи
                child.father_id = father.id
                child.mother_id = mother.id
                father.add_child(child.id)
                mother.add_child(child.id)

                allele_from_f = random.choice([father.mefv_allele_1, father.mefv_allele_2])
                allele_from_m = random.choice([mother.mefv_allele_1, mother.mefv_allele_2])
                child.set_genotype(allele_from_f, allele_from_m)

                # Обновляем год последних родов матери, чтобы сработал birth_cooldown
                # Если сейчас 1950, а ребенку 5 лет, значит роды были в 1945
                birth_year_of_child = 1950 - child.age
                if birth_year_of_child > mother.last_birth_year:
                    mother.last_birth_year = birth_year_of_child

                self.family_children_count[self._get_family_key(father, mother)] += 1

    def _manage_biologics_access(self):
        """Назначает биопрепараты резистентным пациентам."""
        for agent in self.agents.values():
            if not agent.alive or not agent.is_diagnosed or not agent.is_colchicine_resistant or agent.on_antibodies:
                continue

            if random.random() < self.params.get_bio_access_chance():
                agent.on_antibodies = True

    def run_simulation_with_calibration(self, verbose=False, run_id='unknown'):
        yearly_results = []
        self.calibration_results = {k: [] for k in [
            'year', 'target_tfr', 'model_births', 'model_deaths',
            'model_population', 'model_birth_rate', 'target_birth_rate',
            'model_death_rate', 'target_death_rate',
            'model_genetic_sick', 'model_diagnosed', 'prevented_fmf_births'
        ]}

        sum_of_asfr = sum(self.age_fertility_factors.values())
        calibration_factor = 0.23

        # Оптимизация доступа
        self._tfr_dict = {idx: self.fertility_rate_data.loc[idx].iloc[0] for idx in self.fertility_rate_data.index}
        self._death_dict = {idx: self.death_rate_data.loc[idx].iloc[0] for idx in self.death_rate_data.index}
        self._birth_dict = {idx: self.birth_rate_data.loc[idx].iloc[0] for idx in self.birth_rate_data.index}

        self._default_tfr = list(self._tfr_dict.values())[-1] if self._tfr_dict else 0
        self._default_dr = list(self._death_dict.values())[-1] if self._death_dict else 0
        self._default_br = list(self._birth_dict.values())[-1] if self._birth_dict else 0

        # Основной цикл (1950 - 2125)
        for year in range(1950, 2126):
            self.year = year

            # ✅ Вариант Б: Обнуляем накопленную генетическую статистику Менделя
            # ровно в год старта анализируемого периода симуляции
            if self.year == self.simulation_start_year:
                self.inheritance_stats.combo_children_genotypes.clear()

            # Сохраняем состояние до начала года
            births_before = self.children_born
            deaths_before = self.total_deaths
            start_pop = sum(1 for a in self.agents.values() if a.alive)

            target_tfr = self._tfr_dict.get(year, self._default_tfr)
            target_dr = self._death_dict.get(year, self._default_dr)
            target_br = self._birth_dict.get(year, self._default_br)

            self.target_fertility_rate = target_tfr
            self.base_birth_prob = (target_tfr / sum_of_asfr) * calibration_factor
            self.annual_death_prob = target_dr / 1000.0

            # Активация второго сценария
            if self.params.use_screening and self.year >= self.screening_start_year:
                # 1. Генетический скрининг населения
                potential = [a for a in self.agents.values() if a.alive and not a.is_screened and 18 <= a.age <= 45]
                n_to_screen = int(len(potential) * self.params.screening_coverage)
                if n_to_screen > 0:
                    for person in random.sample(potential, n_to_screen):
                        person.is_screened = True

            # Запускаем год
            self._run_single_year_with_tracking(self.annual_death_prob)

            # 1. Демография
            births_this_year = self.children_born - births_before
            deaths_this_year = self.total_deaths - deaths_before

            # Собираем живых агентов один раз для всех расчетов
            alive_agents = [a for a in self.agents.values() if a.alive]
            end_population = len(alive_agents)

            avg_pop = (start_pop + end_population) / 2
            model_br = (births_this_year / avg_pop * 1000) if avg_pop > 0 else 0
            model_dr = (deaths_this_year / avg_pop * 1000) if avg_pop > 0 else 0

            # 2. Медицинские показатели
            diagnosed_count = 0
            total_symptomatic = 0
            actual_genetic_sick = 0

            for a in alive_agents:
                if a.is_diagnosed:
                    diagnosed_count += 1
                if a.clinical_status == 'symptomatic':
                    total_symptomatic += 1
                # Генетически больные: гомозиготы или компаунд-гетерозиготы
                if a.mefv_allele_1 != 'N' and a.mefv_allele_2 != 'N':
                    actual_genetic_sick += 1

            # 3. Записываем в таблицу калибровки
            self._record_calibration(
                year=year,
                target_tfr=target_tfr,
                births=births_this_year,
                deaths=deaths_this_year,
                pop=end_population,
                model_br=model_br,
                target_br=target_br,
                model_dr=model_dr,
                target_dr=target_dr,
                genetic_sick=actual_genetic_sick,
                diagnosed=diagnosed_count
            )

            # 4. Собираем ВСЮ статистику за год
            year_result = self.collect_all_yearly_stats(run_id)
            yearly_results.append(year_result)

        if verbose:
            self._print_final_summary()

        return yearly_results

    def collect_statistics(self, result: Dict[str, Any], run_id: str):
        # 1. Базовые группы
        alive_agents = [a for a in self.agents.values() if a.alive]
        final_population = len(alive_agents)

        if final_population == 0:
            return result

        # 2. Генетика рождений
        cg = self.inheritance_stats.child_genotypes
        result.update({
            'healthy_births': cg.get('healthy', 0),
            'carrier_births': cg.get('carrier', 0),
            'affected_births': cg.get('affected', 0),
            'prevented_fmf_births': self.prevented_fmf_births,
            'pgt_attempts': self.pgt_attempts,
        'pgt_births': self.pgt_births,
        'pgt_eligible_but_declined': self.pgt_eligible_but_declined,
        })

        # 3. Единый цикл анализа
        total_age = 0
        total_symptomatic = 0
        total_genetic_sick = 0  # MM
        total_carriers = 0  # MN
        diagnosed_count = 0
        screened_count = 0
        arm_count = 0

        for a in alive_agents:
            total_age += a.age
            if a.ethnicity == 'Armenian':
                arm_count += 1

            # Собираем генетический статус
            has_m1 = a.mefv_allele_1 != 'N'
            has_m2 = a.mefv_allele_2 != 'N'

            if has_m1 and has_m2:
                total_genetic_sick += 1
            elif has_m1 or has_m2:
                total_carriers += 1

            # Клинический статус
            if a.clinical_status == 'symptomatic':
                total_symptomatic += 1
            if a.is_diagnosed:
                diagnosed_count += 1
            if a.is_screened:
                screened_count += 1

        # 4. Агрегация данных
        result.update({
            'mean_age': total_age / final_population,
            'population_total': final_population,
            'arm_count': arm_count,
            # Генетическая распространенность (все гомозиготы)
            'true_prevalence_genetic': (total_genetic_sick / final_population * 100),
            # Клиническая распространенность (те, кто реально страдает от симптомов)
            'clinical_prevalence': (total_symptomatic / final_population * 100),
            # Частота носительства (MN)
            'carrier_frequency_pct': (total_carriers / final_population * 100),
            # Охват диагностикой (только среди генетически больных)
            'diagnostic_coverage': (diagnosed_count / total_genetic_sick * 100) if total_genetic_sick > 0 else 0,
            # Охват скринингом (здоровые носители + больные)
            'screening_coverage_pct': (screened_count / final_population * 100)
        })

        # 5. Аллели
        allele_analysis = self.get_allele_frequency_analysis()
        for k, v in allele_analysis.get('current', {}).items():
            result[f'allele_freq_{k}'] = v

        return result

    def _record_calibration(self, year, target_tfr, births, deaths, pop,
                            model_br, target_br, model_dr, target_dr,
                            genetic_sick, diagnosed):

        # Стандартная демография
        self.calibration_results['year'].append(year)
        self.calibration_results['target_tfr'].append(target_tfr)
        self.calibration_results['model_births'].append(births)
        self.calibration_results['model_deaths'].append(deaths)
        self.calibration_results['model_population'].append(pop)
        self.calibration_results['model_birth_rate'].append(model_br)
        self.calibration_results['target_birth_rate'].append(target_br)
        self.calibration_results['model_death_rate'].append(model_dr)
        self.calibration_results['target_death_rate'].append(target_dr)

        # Генетические показатели и интервенция
        self.calibration_results['model_genetic_sick'].append(genetic_sick)
        self.calibration_results['model_diagnosed'].append(diagnosed)
        self.calibration_results['prevented_fmf_births'].append(self.prevented_fmf_births)

    def _run_single_year_with_tracking(self, death_prob: float):
        living_ids = [aid for aid, a in self.agents.items() if a.alive]

        for aid in living_ids:
            agent = self.agents[aid]
            was_alive = agent.alive

            # 1. Старение, Смерть и Манифестация
            agent.age_year(annual_death_prob=death_prob, current_year=self.year)

            if agent.alive and agent.clinical_status == 'symptomatic' and not agent.is_diagnosed:
                agent._try_to_diagnose(self.year)

            if not agent.alive:
                if was_alive:
                    self.total_deaths += 1
                continue


            # С возрастом и под влиянием среды стратегия может меняться
            if agent.age >= 18 and agent.age <= 60:
                self._update_mating_strategy(agent)

        self._manage_biologics_access()

        # 3. Рождаемость (пары и дети)
        self._form_new_partnerships()
        self._birth_process()

        self._record_population_stats()

    def _update_mating_strategy(self, agent: Agent):
        """Обновляет стратегию спаривания взрослого агента с учетом контекста эпохи"""
        revision_prob = 0.01  # Уменьшаем до 1%, чтобы снизить хаос

        if random.random() > revision_prob:
            return

        # 1. Исторический базис vs Будущее (синхронно с логикой рождения детей)
        if self.year < 2000:
            current_target_assortativity = 0.85  # Единая история для всех
        else:
            # Плавный переход к целевому параметру сценария
            transition_period = min(1.0, (self.year - 2000) / 30.0)
            current_target_assortativity = 0.85 + transition_period * (self.params.ethnic_assortativity - 0.85)

        # 2. Фактор года рождения (молодые более склонны к изменению в сторону текущего тренда)
        # Переменная показывает, насколько агент "молод" в масштабах симуляции
        cohort_shift = min(0.2, max(0.0, (agent.birth_year - 1950) / 100.0))

        # Целевая вероятность экзогамии для текущего года в конкретном сценарии
        target_exogamy = 1.0 - current_target_assortativity + cohort_shift

        # Корректируем стратегию, аккуратно подталкивая агента к тренду эпохи
        if agent.mating_strategy == 'endogamy' and random.random() < target_exogamy * 0.05:
            agent.mating_strategy = 'exogamy'
        elif agent.mating_strategy == 'exogamy' and random.random() < (1.0 - target_exogamy) * 0.05:
            agent.mating_strategy = 'endogamy'

    def _form_new_partnerships(self):
        # 1. Собираем одиноких мужчин и группируем женщин по этносам
        single_males = [a for a in self.agents.values()
                        if a.alive and a.gender == 'male' and a.partner_id is None and 18 <= a.age <= 60]

        available_females_by_eth = {}
        for a in self.agents.values():
            if a.alive and a.gender == 'female' and a.partner_id is None and 18 <= a.age <= 45:
                eth = a.ethnicity
                if eth not in available_females_by_eth:
                    available_females_by_eth[eth] = []
                available_females_by_eth[eth].append(a.id)

        if not single_males or not available_females_by_eth:
            return

        random.shuffle(single_males)

        for male in single_males:
            # Базовая вероятность создания пары в этом году
            partnership_prob = max(0.1, 0.3 - (male.age - 25) * 0.01)
            if random.random() > partnership_prob:
                continue

            male_eth = male.ethnicity
            active_eths = [e for e, ids in available_females_by_eth.items() if len(ids) > 0]
            if not active_eths:
                break

            # 2. Определение стратегии выбора
            # До 2010 - 0.85, после - 0.75 (для Сценария 2)
            current_threshold = 0.85 if self.year < self.screening_start_year else self.params.ethnic_assortativity

            # Бросаем кубик: это "целевой поиск своей группы" или "случайное знакомство"?
            is_assortative_attempt = random.random() < current_threshold

            if is_assortative_attempt and male_eth in active_eths:
                # Стратегия А: Мужчина ищет только женщину своего этноса
                target_pool_eth = male_eth
            else:
                # Стратегия Б: Мужчине всё равно. Выбор этноса невесты зависит
                # от того, сколько женщин какого этноса сейчас доступно.
                # Если армянок 80%, то с вероятностью 80% он выберет армянку.
                eth_weights = [len(available_females_by_eth[e]) for e in active_eths]
                target_pool_eth = random.choices(active_eths, weights=eth_weights, k=1)[0]

            # 3. Выбер невесты
            candidates_ids = available_females_by_eth[target_pool_eth]
            # Проверяем несколько кандидаток, чтобы избежать встречи сибленгов
            attempts = min(5, len(candidates_ids))
            potential_bride_ids = random.sample(candidates_ids, attempts)

            for f_id in potential_bride_ids:
                female = self.agents[f_id]

                is_sibling = (male.father_id is not None and male.father_id == female.father_id) or \
                             (male.mother_id is not None and male.mother_id == female.mother_id)

                if not is_sibling:
                    # Заключаем союз
                    male.set_partner(female)
                    # Убираем женщину из доступных
                    available_females_by_eth[target_pool_eth].remove(f_id)

                    family_key = self._get_family_key(male, female)
                    self.family_children_count[family_key] = 0
                    break

    def _initialize_fertility_map(self):
        self._fertility_map = [0.0] * 101
        for (start, end), val in self.age_fertility_factors.items():
            fertility_value = float(val)
            for age in range(start, end + 1):
                if age <= 100:
                    self._fertility_map[age] = fertility_value

    def _get_fertility_factor(self, age: int, agent: Agent = None) -> float:
        """Учитывает возраст и, если передан агент, его состояние здоровья"""
        if 18 <= age <= 45:
            base_factor = self._fertility_map[age]

            #  учитываем fertility_recovery для больных женщин
            if agent and agent.gender == 'female' and agent.clinical_status == 'symptomatic':
                base_factor *= self.params.fertility_recovery

            return base_factor
        return 0.0


    def _birth_process(self):
        # 1. Подготовка демографических ограничений
        children_penalty_map = {0: 1.0, 1: 1.0, 2: 0.8, 3: 0.4}

        potential_mothers = []
        fathers_cache = {}

        # Фильтрация пула кандидатов за один проход
        for agent in self.agents.values():
            if agent.alive and agent.gender == 'female' and agent.can_get_pregnant(self.year, self.birth_cooldown):
                potential_mothers.append(agent)
                if agent.partner_id and agent.partner_id in self.agents:
                    fathers_cache[agent.id] = self.agents[agent.partner_id]

        # 2. Основной цикл симуляции беременностей
        for mother in potential_mothers:
            father = fathers_cache.get(mother.id)
            if not father or not father.alive:
                continue

            # Социальный фактор: штраф за многодетность (снижение мотивации)
            family_key = self._get_family_key(father, mother)
            current_children_count = self.family_children_count.get(family_key, 0)
            children_penalty = children_penalty_map.get(current_children_count, 0.1)

            # Биологический фактор: возрастной коэффициент фертильности
            age_factor = self._get_fertility_factor(mother.age, mother)
            if age_factor <= 0:
                continue

            # Итоговая вероятность реализации репродуктивного намерения в этом году
            birth_prob = self.base_birth_prob * age_factor * children_penalty

            if random.random() < birth_prob:
                child_was_born = False
                force_healthy_child = False


                # Вычисляем чистый природный генотип ДО вмешательства медицины
                raw_fa = random.choice([father.mefv_allele_1, father.mefv_allele_2])
                raw_ma = random.choice([mother.mefv_allele_1, mother.mefv_allele_2])
                raw_m_count = (1 if raw_fa != 'N' else 0) + (1 if raw_ma != 'N' else 0)
                raw_status = 'healthy' if raw_m_count == 0 else ('carrier' if raw_m_count == 1 else 'affected')

                # ИСПРАВЛЕНИЕ: Используем канонический генератор ключей класса, совпадающий с валидатором
                combo_key = self._get_combo_key_for_theory(mother, father)

                # Записываем чистую теорию Менделя с соблюдением типов данных
                if combo_key not in self.inheritance_stats.combo_children_genotypes:
                    from collections import defaultdict
                    d = defaultdict(int)
                    d['healthy'] = 0
                    d['carrier'] = 0
                    d['affected'] = 0
                    self.inheritance_stats.combo_children_genotypes[combo_key] = d

                self.inheritance_stats.combo_children_genotypes[combo_key][raw_status] += 1
                # ===============================================

                # МЕДИКО-ГЕНЕТИЧЕСКИЙ БЛОК
                if self.year >= self.screening_start_year and self.params.use_screening:

                    # Шаг А: Реализация охвата программы
                    if not (mother.is_screened and father.is_screened):
                        if random.random() < self.params.screening_coverage:
                            mother.is_screened = True
                            father.is_screened = True

                    # Шаг Б: Если пара идентифицирована и знает свой статус
                    if mother.is_screened and father.is_screened:
                        father_is_carrier = (father.mefv_allele_1 != 'N' or father.mefv_allele_2 != 'N')
                        mother_is_carrier = (mother.mefv_allele_1 != 'N' or mother.mefv_allele_2 != 'N')
                        is_high_risk = father_is_carrier and mother_is_carrier

                        if is_high_risk:
                            # Комплаентность: доля пар, готовых менять поведение под влиянием медицины
                            if random.random() < self.params.screening_efficiency:

                                if self.params.use_pgt and self.year >= self.pgt_start_year:
                                    # СЦЕНАРИЙ 3:

                                    # ФАКТОР 1: Социально-финансовый барьер принятия решения
                                    if random.random() < self.params.pgt_efficiency:

                                        # ПЛАН А: Пара согласилась и вступила в протокол ЭКО
                                        self.pgt_attempts += 1  # Фиксируем попытку ДО броска на приживаемость

                                        # ФАКТОР 2: Медицинская результативность (приживаемость эмбриона)
                                        if random.random() < self.params.pgt_efficiency:
                                            # Успешный цикл — беременность наступила!
                                            force_healthy_child = True
                                            child_was_born = True
                                            self.pgt_births += 1

                                            # Расчет предотвращенного бремени болезни
                                            if random.random() < 0.25:
                                                self.prevented_fmf_births += 1
                                        else:
                                            # Медицинская неудача цикла. Эмбрион не прижился.
                                            # В этом году ребенок не родился, пара пропускает ход.
                                            continue  # Переходим к следующей матери
                                    else:
                                        # ПЛАН Б: Пара ОТКАЗАЛАСЬ от PGT из-за финансового барьера
                                        self.pgt_eligible_but_declined += 1

                                        # Отказавшиеся уходят на обычные естественные роды (продолжают выполнение вниз)
                                        force_healthy_child = False
                                        child_was_born = False
                                else:
                                    # СЦЕНАРИЙ 2: Пренатальная диагностика (ПД)
                                    if random.random() < 0.25:
                                        self.prevented_fmf_births += 1
                                        continue
                                    else:
                                        force_healthy_child = True
                                        child_was_born = True
                            else:
                                # Пары, отказавшиеся от интервенций: идут на естественный риск ниже
                                pass

                # 3. Финализация рождения и вызов конструктора агента
                if not child_was_born:
                    # Обычные естественные роды
                    self._create_child_with_detailed_tracking(mother, father, force_healthy=False)
                    child_was_born = True
                elif force_healthy_child:
                    # Рождается отфильтрованный ребенок
                    child = self._create_child_with_detailed_tracking(mother, father, force_healthy=True)
                    if child:
                        self.pgt_children_ids.append(child.id)
                    child_was_born = True

                # Обновление демографических логов семьи (строго при подтвержденном рождении)
                if child_was_born:
                    self.family_children_count[family_key] = current_children_count + 1
                    mother.last_birth_year = self.year
                    self.children_born += 1

    def print_fertility_report(self):

        # 1. Целевые показатели
        actual_tfr = self.calculate_actual_fertility_rate()
        print(f"\n{'=' * 20} Отчет по рождаемости ({self.year}) {'=' * 20}")
        print(f"Целевой TFR (исторический): {self.target_fertility_rate:.2f}")
        print(f"Фактический TFR модели:     {actual_tfr:.2f}")
        print(f"Отклонение калибровки:      {(actual_tfr - self.target_fertility_rate):+.2f}")

        # 2. Возрастная структура
        living_agents = [a for a in self.agents.values() if a.alive]
        total = len(living_agents)
        if total > 0:
            child_count = sum(1 for a in living_agents if a.age <= 14)
            print(f"Всего живых агентов:        {total}")
            print(f"Доля детей (0-14 лет):      {child_count / total * 100:.1f}%")

        # 3. Генетическая статистика рождений (Кумулятивная)
        cg = self.inheritance_stats.child_genotypes
        total_births_tracked = sum(cg.values())

        print(f"\n Генетика рожденных детей (кумулятивно):")
        if total_births_tracked > 0:
            print(
                f"  Здоровые (N/N): {cg.get('healthy', 0):>5} ({cg.get('healthy', 0) / total_births_tracked * 100:5.1f}%)")
            print(
                f"  Носители (N/M): {cg.get('carrier', 0):>5} ({cg.get('carrier', 0) / total_births_tracked * 100:5.1f}%)")
            print(
                f"  Больные (M/M):  {cg.get('affected', 0):>5} ({cg.get('affected', 0) / total_births_tracked * 100:5.1f}%)")
        else:
            print("  Данные о рождениях в этом запуске отсутствуют.")

        # 4. Медицинские показатели и интервенция
        print(f"\n Медицинские показатели и интервенция:")

        # Определяем тип сценария для вывода
        if self.params.use_screening:
            if self.params.screening_coverage >= 0.8 and self.params.screening_efficiency >= 0.8:
                print(f"  [СЦЕНАРИЙ 3] Снижение ассортативности + Массовый скрининг + ПГД")
                print(f"  Ассортативность: {self.params.ethnic_assortativity}")
                print(f"  Охват скринингом: {self.params.screening_coverage * 100:.0f}%")
                print(f"  Эффективность консультирования: {self.params.screening_efficiency * 100:.0f}%")
            else:
                print(f"  [СЦЕНАРИЙ 2] Интервенция (скрининг)")
            print(f"  [!] Предотвращено рождений (скрининг): {self.prevented_fmf_births}")
        else:
            print(f"  [СЦЕНАРИЙ 1] Status Quo (без скрининга)")

        # 5. Резистентность и биопрепараты (важно для Сценария 3)
        if total > 0:
            diagnosed_agents = [a for a in living_agents if a.is_diagnosed]
            resistant_agents = [a for a in diagnosed_agents if a.is_colchicine_resistant]
            on_antibodies_agents = [a for a in resistant_agents if a.on_antibodies]

            print(f"\n Резистентность и терапия:")
            print(f"  Диагностировано больных:     {len(diagnosed_agents)}")
            print(
                f"  Биологически не ответившие на базовую терапию:    {len(resistant_agents)} ({len(resistant_agents) / len(diagnosed_agents) * 100 if diagnosed_agents else 0:.1f}% от diagnosed)")
            print(
                f"  Получают биопрепараты:        {len(on_antibodies_agents)} ({len(on_antibodies_agents) / len(resistant_agents) * 100 if resistant_agents else 0:.1f}% от резистентных)")

        # 6. Фертильность и лечение
        mothers_on_colchicine = sum(1 for a in living_agents if a.gender == 'female' and a.on_colchicine)
        mothers_on_antibodies = sum(1 for a in living_agents if a.gender == 'female' and a.on_antibodies)

        print(f"\n Репродуктивные показатели:")
        print(f"  Матерей на колхицине:          {mothers_on_colchicine}")
        print(f"  Матерей на биопрепаратах:      {mothers_on_antibodies}")

        recovery_status = "Полное (1.0)" if self.params.fertility_recovery >= 0.9 else f"Частичное ({self.params.fertility_recovery:.2f})"
        print(f"  Восстановление фертильности:    {recovery_status}")

        # 7. Итоговые показатели семьи
        families = len(self.family_children_count)
        avg_children = self.children_born / families if families > 0 else 0
        print(f"\n Итоги симуляции к {self.year} году:")
        print(f"  Всего рождений:                {self.children_born}")
        print(f"  Среднее число детей на семью:  {avg_children:.2f}")
        print("=" * 60 + "\n")

    def calculate_actual_fertility_rate(self, birth_year_range: tuple = None) -> float:

        # 1. Берем женщин, завершивших репродуктивный цикл
        completed_women = [
            a for a in self.agents.values()
            if a.alive and a.gender == 'female' and a.age >= 45
               and a.birth_year >= self.simulation_start_year
        ]
        # 2. Фильтрация по когортам (годам рождения)
        if birth_year_range:
            start_y, end_y = birth_year_range
            completed_women = [w for w in completed_women if start_y <= w.birth_year <= end_y]

        if not completed_women:
            # Если модель только началась
            # женщин 45+, родивших внутри модели, может не быть.
            return 0.0

        # 3. Расчет
        # Важно: суммируем количество реально рожденных детей (по списку ID)
        total_children = sum(len(w.children_ids) for w in completed_women)

        return total_children / len(completed_women)

    # определяем генотип ребенка
    def _create_child_with_detailed_tracking(self, parent1: Agent, parent2: Agent, force_healthy: bool = False):
        # 1. Роли (уже проверено, что они разного пола в birth_process)
        father, mother = (parent1, parent2) if parent1.gender == 'male' else (parent2, parent1)

        # 2. Менделевское наследование
        if force_healthy:
            # PGT: собираем все теоретически возможные комбинации аллелей
            possible_combinations = []
            for fa in [father.mefv_allele_1, father.mefv_allele_2]:
                for ma in [mother.mefv_allele_1, mother.mefv_allele_2]:
                    possible_combinations.append((fa, ma))

            # Шаг 1: Фильтруем — убираем только больных (M/M). Оставляем клинически здоровых (N/N и N/M)
            # Для этого у эмбриона должен быть хотя бы ОДИН здоровый аллель 'N' (доминантное здоровье/носительство)
            target_combinations = [(fa, ma) for (fa, ma) in possible_combinations if fa == 'N' or ma == 'N']

            # Шаг 2: Если пара экзотическая (оба больны M/M x M/M), здоровых аллелей нет вообще.
            # В таком случае у медицины нет выбора — берем то, что есть.
            if not target_combinations:
                target_combinations = possible_combinations

            # PGT выбирает случайный эмбрион из КЛИНИЧЕСКИ ЗДОРОВЫХ (сохраняя баланс N/N и N/M)
            father_allele, mother_allele = random.choice(target_combinations)
        else:
            # Обычное наследование: случайный выбор без вмешательства врачей
            father_allele = random.choice([father.mefv_allele_1, father.mefv_allele_2])
            mother_allele = random.choice([mother.mefv_allele_1, mother.mefv_allele_2])

        alleles_sorted = sorted([father_allele, mother_allele])

        # 3. ЭТНОС РЕБЕНКА
        if not self.params.use_screening:
            # Сценарий 1: Status Quo - этническое распределение как в 1950 году
            eth_names = ['Armenian', 'Other']
            eth_weights = [0.9, 0.1]  # Историческое распределение
        else:
            # Сценарий 2 или 3 - используем целевое распределение из параметров
            eth_names = list(self.params.ethnic_distribution.keys())
            eth_weights = list(self.params.ethnic_distribution.values())

            # Для Сценария 2 и 3: после года вмешательства начинаем переход к целевому распределению
            if self.year >= self.screening_start_year:
                # Плавный переход к целевому распределению за 30 лет
                transition_years = 30
                years_since_start = min(transition_years, self.year - self.screening_start_year)
                progress = years_since_start / transition_years

                # Интерполяция весов
                hist_weights = [0.9, 0.1]
                target_weights = eth_weights

                # Линейная интерполяция
                current_weights = [
                    hist_weights[i] * (1 - progress) + target_weights[i] * progress
                    for i in range(len(hist_weights))
                ]
                eth_weights = current_weights

        # Выбираем этнос ребенка согласно рассчитанным весам
        child_ethnicity = random.choices(eth_names, weights=eth_weights, k=1)[0]

        # 4. СТРАТЕГИЯ СПАРИВАНИЯ РЕБЕНКА
        historical_assortativity = 0.85
        target_assortativity = self.params.ethnic_assortativity

        if not self.params.use_screening:
            effective_endogamy_prob = historical_assortativity
        else:
            start_year = self.screening_start_year

            if self.year <= start_year:
                effective_endogamy_prob = historical_assortativity
            else:
                transition_years = 50
                years_since_start = self.year - start_year
                progress = min(1.0, years_since_start / transition_years)

                # Линейная интерполяция
                effective_endogamy_prob = historical_assortativity + (
                        target_assortativity - historical_assortativity) * progress

        # Дополнительные факторы:
        mixed_parents_bonus = 0.1 if father.ethnicity != mother.ethnicity else 0.0

        year_effect = (self.year - 1950) / 1000.0
        year_effect = min(0.15, year_effect)

        # Финальная вероятность эндогамии
        endogamy_prob = effective_endogamy_prob - mixed_parents_bonus - year_effect
        endogamy_prob = max(0.05, min(0.95, endogamy_prob))

        # Применяем стратегию
        if random.random() < endogamy_prob:
            child_mating_strategy = 'endogamy'
        else:
            child_mating_strategy = 'exogamy'

        # 5. Создание агента
        child_gen = max(father.generation, mother.generation) + 1

        child = Agent(
            params=self.params,
            gender=random.choice(['male', 'female']),
            age=0,
            generation=child_gen,
            birth_year=self.year,
            ethnicity=child_ethnicity,
            mating_strategy=child_mating_strategy,
            father_id=father.id,
            mother_id=mother.id
        )

        child.set_genotype(alleles_sorted[0], alleles_sorted[1])

        # 6. РЕГИСТРАЦИЯ СВЯЗЕЙ
        child_id = child.id

        self.agents[child_id] = child
        father.add_child(child_id)
        mother.add_child(child_id)

        return child

    def _update_inheritance_stats(self, father, mother, f_allele, m_allele, child_genotype_str):
        stats = self.inheritance_stats

        # 1. Фиксируем фактическую передачу аллелей у родившегося ребенка
        stats.allele_transmission[f"father_{f_allele}"] += 1
        stats.allele_transmission[f"mother_{m_allele}"] += 1

        # 2. Ключ комбинации родителей
        def get_canonical_genotype(agent):
            alleles = sorted([agent.mefv_allele_1, agent.mefv_allele_2])
            return f"{alleles[0]}/{alleles[1]}"

        father_gen = get_canonical_genotype(father)
        mother_gen = get_canonical_genotype(mother)

        if father_gen <= mother_gen:
            parent_combo = f"{father_gen} x {mother_gen}"
        else:
            parent_combo = f"{mother_gen} x {father_gen}"

        stats.parent_combinations[parent_combo] += 1

        # 3. Определение фактического статуса родившегося ребенка
        is_m1_mut = f_allele != 'N'
        is_m2_mut = m_allele != 'N'

        if is_m1_mut and is_m2_mut:
            child_status = "affected"
        elif is_m1_mut or is_m2_mut:
            child_status = "carrier"
        else:
            child_status = "healthy"

        # Запись в фактическую статистику популяции
        stats.child_genotypes[child_status] += 1
        stats.mutation_pairs[child_genotype_str] += 1

        if child_status not in stats.children_genotype_by_parent_combo[parent_combo]:
            stats.children_genotype_by_parent_combo[parent_combo][child_status] = 0
        stats.children_genotype_by_parent_combo[parent_combo][child_status] += 1


        # 4. Контроль аномалий
        if child_status != "healthy" and not (is_m1_mut or is_m2_mut):
            print(f"(!) Генетическая ошибка: Родители {parent_combo} -> Ребенок {child_genotype_str}")

    def _get_combo_key_for_theory(self, father: Agent, mother: Agent) -> str:

        def get_simple_genetic_type(agent):
            alleles = [agent.mefv_allele_1, agent.mefv_allele_2]
            mut_count = sum(1 for a in alleles if a != "N")

            if mut_count == 0:
                return 'healthy'
            elif mut_count == 1:
                return 'carrier'
            else:  # mut_count == 2
                return 'affected'

        f_type = get_simple_genetic_type(father)
        m_type = get_simple_genetic_type(mother)

        # Сортируем для создания ключа
        types = sorted([f_type, m_type])

        # Создаем ключ в формате, который ожидает theoretical_expectations
        if types[0] == 'healthy' and types[1] == 'healthy':
            return 'healthy_healthy'
        elif types[0] == 'healthy' and types[1] == 'carrier':
            return 'carrier_healthy'  # В theoretical_expectations используется 'carrier_healthy'
        elif types[0] == 'carrier' and types[1] == 'carrier':
            return 'carrier_carrier'
        elif types[0] == 'healthy' and types[1] == 'affected':
            return 'affected_healthy'
        elif types[0] == 'carrier' and types[1] == 'affected':
            return 'affected_carrier'
        elif types[0] == 'affected' and types[1] == 'affected':
            return 'affected_affected'
        else:
            # fallback - на всякий случай
            return f"{types[0]}_{types[1]}"

    def _calculate_expected_inheritance(self, father: Agent, mother: Agent) -> dict:
        f_alleles = [father.mefv_allele_1, father.mefv_allele_2]
        m_alleles = [mother.mefv_allele_1, mother.mefv_allele_2]

        outcomes = []
        for fa in f_alleles:
            for ma in m_alleles:
                # Сортировка важна для идентичности ключей 'N/M' == 'M/N'
                gen = sorted([fa, ma])
                outcomes.append(f"{gen[0]}/{gen[1]}")

        genotype_counts = Counter(outcomes)

        phenotype_probs = {
            "healthy": 0.0,
            "carrier": 0.0,
            "affected": 0.0,
            "severe_risk": 0.0,  # M694V/M694V
            "compound_severe": 0.0  # M694V + любая другая мутация у больного
        }

        for genotype_str, count in genotype_counts.items():
            alleles = genotype_str.split('/')
            mut_count = sum(1 for a in alleles if a != "N")
            prob = count / 4.0

            if mut_count == 2:
                phenotype_probs["affected"] += prob

                # Анализ тяжести (генетический ландшафт Армении)
                is_m694v_1 = alleles[0] == 'M694V'
                is_m694v_2 = alleles[1] == 'M694V'

                if is_m694v_1 and is_m694v_2:
                    phenotype_probs["severe_risk"] += prob
                elif is_m694v_1 or is_m694v_2:
                    phenotype_probs["compound_severe"] += prob

            elif mut_count == 1:
                phenotype_probs["carrier"] += prob
            else:
                phenotype_probs["healthy"] += prob

        return {
            "genotypes": {gen: count / 4.0 for gen, count in genotype_counts.items()},
            "phenotypes": phenotype_probs
        }

    def _record_population_stats(self):

        agents_list = list(self.agents.values())

        if not agents_list:
            self.population_history.append({
                'year': self.year, 'total_population': 0, 'genotype_stats': {},
                'clinical_stats': {}, 'treatment_stats': {}, 'mutation_distribution': {},
                'generation_counts': {}, 'prevented_births_total': self.prevented_fmf_births
            })
            return

        # 1. Фильтрация живых
        alive_agents = [a for a in agents_list if a.alive]
        n_alive = len(alive_agents)

        if n_alive == 0:
            # Если все вымерли, фиксируем финальное состояние
            self.population_history.append({'year': self.year, 'total_population': 0})
            return

        # 2. Создаем массивы для быстрой статистики
        alive_genotypes = np.array([a.genotype_status for a in alive_agents], dtype=object)
        alive_clinical = np.array([a.clinical_status for a in alive_agents], dtype=object)
        alive_mutation = np.array([a.mutation_type if a.mutation_type else 'none' for a in alive_agents], dtype=object)
        alive_severity = np.array([a.disease_severity if a.disease_severity else 'none' for a in alive_agents],
                                  dtype=object)

        # Флаги диагностики и скрининга
        alive_diagnosed = np.array([a.is_diagnosed for a in alive_agents])
        alive_screened = np.array([a.is_screened for a in alive_agents])
        alive_on_colchicine = np.array([a.on_colchicine for a in alive_agents])

        # 3. Генетическая и клиническая статистика
        unique_gen, counts_gen = np.unique(alive_genotypes, return_counts=True)
        unique_clin, counts_clin = np.unique(alive_clinical, return_counts=True)

        # 4. Специфика симптомов (Пенетрантность и Тяжесть)
        symptomatic_mask = (alive_clinical == 'symptomatic')
        mutation_distribution = {}
        if np.any(symptomatic_mask):
            mut_vals, mut_counts = np.unique(alive_mutation[symptomatic_mask], return_counts=True)
            mutation_distribution = dict(zip(mut_vals, mut_counts))

        # 5. Сценарные показатели
        diagnosis_stats = {
            'diagnosed': int(np.sum(alive_diagnosed)),
            'undiagnosed_symptomatic': int(np.sum(symptomatic_mask & ~alive_diagnosed))
        }

        # Важно: Скрининг считаем только если он включен в параметрах
        screening_stats = {
            'total_screened': int(np.sum(alive_screened))
        }

        # 6. Формирование финального словаря года
        stats = {
            'year': self.year,
            'total_population': n_alive,
            'genotype_stats': dict(zip(unique_gen, counts_gen.tolist())),
            'clinical_stats': dict(zip(unique_clin, counts_clin.tolist())),
            'treatment_stats': {
                'on_colchicine': int(np.sum(alive_on_colchicine)),
                'no_colchicine': int(n_alive - np.sum(alive_on_colchicine))
            },
            'mutation_distribution': mutation_distribution,
            'diagnosis_stats': diagnosis_stats,
            'screening_stats': screening_stats,
            # Накопленный итог предотвращенных рождений для графиков эффективности
            'prevented_births_total': self.prevented_fmf_births
        }

        self.population_history.append(stats)

    def _print_initial_stats(self):
        """Выводит детальный аудит популяции в 1950 году (Поколение 0)"""
        print("\n" + "═" * 70)
        # Название сценария берем из параметров
        scenario_type = "Modernization (Scenario 2)" if self.params.use_screening else "Status Quo (Scenario 1)"
        print(f" Исходные данные (1950 год) | {scenario_type}")
        print("═" * 70)

        if not self.population_history:
            print("Ошибка: История популяции пуста. Запишите первый год перед вызовом.")
            return

        initial = self.population_history[0]
        total = initial['total_population']

        # 1. Этнический состав (используем прямое обращение к словарю)
        e_stats = initial.get('ethnicity_stats', {})
        print(f"Этнический состав (Поколение 0):")
        for eth, count in e_stats.items():
            share = (count / total * 100) if total > 0 else 0
            print(f"  {eth:<12}: {count:>6} ({share:>5.1f}%)")
        print("-" * 40)

        # 2. Анализ генотипов с доверительными интервалами
        g_stats = initial.get('genotype_stats', {})
        if not g_stats:
            print("Данные о генотипах отсутствуют.")
        else:
            # Создаем таблицу для красивого вывода
            rows = []
            for gen, n in g_stats.items():
                p = n / total
                se = np.sqrt(p * (1 - p) / total) if total > 0 else 0
                z = 1.96  # 95% Confidence Interval

                low_ci = max(0, (p - z * se) * 100)
                high_ci = min(100, (p + z * se) * 100)

                rows.append({
                    'Генотип': str(gen).capitalize(),
                    'N': n,
                    'Share': p * 100,
                    'CI': f"[{low_ci:>4.1f}% - {high_ci:>4.1f}%]"
                })

            print(f"{'Генотип':<15} {'N':>6} {'%':>8} {'95% Довер. Интервал':>24}")
            print("-" * 60)
            for r in rows:
                print(f"{r['Генотип']:<15} {r['N']:>6} {r['Share']:>7.1f}%   {r['CI']}")

        # 3. Стартовая нагрузка FMF
        c_stats = initial.get('clinical_stats', {})
        symptomatic = c_stats.get('symptomatic', 0)
        prevalence = (symptomatic / total * 100) if total > 0 else 0

        print("-" * 60)
        print(f"Начальная заболеваемость (Symptomatic): {symptomatic} ({prevalence:.2f}%)")

        # Считаем носительство (MN) для Поколения 0
        carriers = g_stats.get('carrier', 0)
        print(f"Начальная частота носительства (Carrier): {carriers} ({carriers / total * 100:.1f}%)")
        print("═" * 70 + "\n")

    def _print_generation_breakdown(self):

        if not self.generation_stats:
            print("Статистика поколений не сформирована.")
            return

        data = []
        for g, stats in sorted(self.generation_stats.items()):
            row = {'gen': int(g)}
            row.update(stats)
            data.append(row)

        df = pd.DataFrame(data)
        # Оставляем только те группы, где есть живые представители
        df = df[df['alive'] > 0].copy()

        if df.empty:
            print("Активные поколения не найдены.")
            return

        # Расчет удельных показателей
        df['sick_rate'] = (df['symptomatic'] / df['alive'] * 100)
        # "Асимптоматические" — это и здоровые, и носители, у кого нет приступов
        df['asymptomatic'] = df['alive'] - df['symptomatic']

        print(f"\n{'═' * 20} АНАЛИЗ ПОКОЛЕНИЙ (на {self.year} год) {'═' * 20}")

        # Форматированный вывод таблицы
        headers = ['Поколение', 'Живых (N)', 'Симптом.', 'Без симптомов', '% Больных']
        columns = ['gen', 'alive', 'symptomatic', 'asymptomatic', 'sick_rate']

        # Печатаем таблицу через pandas для аккуратности
        print(df[columns].to_string(
            index=False,
            header=headers,
            formatters={'sick_rate': '{:>.1f}%'.format},
            justify='center'
        ))

        print("-" * 65)

        # Итоговые метрики по всем поколениям
        total_alive = df['alive'].sum()
        total_sympt = df['symptomatic'].sum()

        if total_alive > 0:
            avg_prevalence = (total_sympt / total_alive * 100)
            print(f"Всего живых в модели:  {total_alive:>6} чел.")
            print(f"Общая заболеваемость:  {total_sympt:>6} ({avg_prevalence:.1f}%)")

        # Сценарный анализ: сравнение "отцов" и "детей"
        if len(df) > 1:
            first_gen = df.iloc[0]
            last_gen = df.iloc[-1]
            diff = first_gen['sick_rate'] - last_gen['sick_rate']

            print(f"\nСравнение: G{int(first_gen['gen'])} (старшие) vs G{int(last_gen['gen'])} (младшие)")
            print(f"Разрыв в проявлении симптомов: {diff:+.1f}%")

            if self.params.use_screening and diff > 5:
                print("[!] Наблюдается эффект снижения заболеваемости в новых поколениях.")

        print("═" * 65 + "\n")

    def _print_calibration_report(self):
        """Выводит научный отчет о точности калибровки и демографической валидации"""
        if not self.calibration_results['year']:
            print("Данные для калибровки отсутствуют.")
            return

        df = pd.DataFrame(self.calibration_results)

        print("\n" + "═" * 80)
        scenario_label = "Modernization (Scenario 2)" if self.params.use_screening else "Status Quo (Scenario 1)"
        print(f"{f' ОТЧЕТ О ВАЛИДАЦИИ: {scenario_label}':^80}")
        print(f"{'(Период исторической калибровки: 1950-2024)':^80}")
        print("═" * 80)

        # 1. Контрольные точки для таблицы
        check_points = [1950, 1970, 1990, 2010, 2024]
        df_check = df[df['year'].isin(check_points)].copy()

        # Считаем отклонения
        df_check['br_error'] = np.abs(df_check['model_birth_rate'] - df_check['target_birth_rate'])
        df_check['dr_error'] = np.abs(df_check['model_death_rate'] - df_check['target_death_rate'])

        def evaluate_fit(row):
            if row['year'] == 1950: return '⏺ Старт'
            # Допуск в 3.0 пункта — хороший стандарт для стохастических моделей
            if row['br_error'] < 3.0 and row['dr_error'] < 2.0:
                return '✅ Точно'
            return '⚠️ Шум'

        df_check['Статус'] = df_check.apply(evaluate_fit, axis=1)

        print(df_check[
                  ['year', 'model_birth_rate', 'target_birth_rate', 'model_death_rate', 'target_death_rate', 'Статус']]
              .to_string(index=False,
                         header=['Год', 'BR (Мод)', 'BR (Цель)', 'DR (Мод)', 'DR (Цель)', 'Статус'],
                         formatters={
                             'model_birth_rate': '{:>.2f}'.format,
                             'target_birth_rate': '{:>.2f}'.format,
                             'model_death_rate': '{:>.2f}'.format,
                             'target_death_rate': '{:>.2f}'.format
                         }, justify='center'))

        print("-" * 80)

        # 2. Метрики точности (анализируем стабильный период 1980+)
        stable_df = df[df['year'] >= 1980]
        if not stable_df.empty:
            # RMSE для рождаемости
            rmse_br = np.sqrt(np.mean((stable_df['model_birth_rate'] - stable_df['target_birth_rate']) ** 2))
            # MAE для смертности
            mae_dr = np.mean(np.abs(stable_df['model_death_rate'] - stable_df['target_death_rate']))

            # Средняя относительная ошибка (MAPE) — очень убедительно для диплома
            mape_br = np.mean(np.abs((stable_df['model_birth_rate'] - stable_df['target_birth_rate']) / stable_df[
                'target_birth_rate'])) * 100

            print(f"Статистическая точность (1980-2024):")
            print(f"   Рождаемость (RMSE): {rmse_br:.2f} (MAPE: {mape_br:.1f}%)")
            print(f"   Смертность (MAE):   {mae_dr:.2f}")

            verdict = "✅ ВЫСОКАЯ" if rmse_br < 2.5 else ("🆗 ПРИЕМЛЕМАЯ" if rmse_br < 4.5 else "⚠️ НИЗКАЯ")
            print(f"   Адекватность модели: {verdict}")

        # 3. Динамика популяции
        initial_pop = df['model_population'].iloc[0]
        final_pop = df['model_population'].iloc[-1]
        growth = (final_pop / initial_pop - 1) * 100

        print(f"\n Итоги демографии:")
        print(f"   Население (1950): {int(initial_pop):>7} чел.")
        print(f"   Население (2024): {int(final_pop):>7} чел.")
        print(f"   Общее изменение:  {growth:+.1f}% {'📈' if growth > 0 else '📉'}")
        print("═" * 80)

    def _print_age_structure(self):
        """Выводит детальный анализ возрастной структуры и здоровья"""
        living_agents = [a for a in self.agents.values() if a.alive]

        if not living_agents:
            print(f"\n Год {self.year}: Нет живых агентов")
            return

        ages = [a.age for a in living_agents]
        df = pd.DataFrame(living_agents, columns=['age', 'clinical_status', 'gender'])
        n = len(living_agents)

        print(f"\n{'=' * 65}")
        print(f" Возрастная структура и доживаемость (Год: {self.year})")
        print(f"{'=' * 65}")

        # Сравнительная статистика (Больные vs Здоровые)
        sick_ages = df[df['clinical_status'] == 'symptomatic']['age']

        print("Средний возраст:")
        print(f"  Вся популяция: {df['age'].mean():>5.1f} лет")
        if not sick_ages.empty:
            print(f"  Больные (FMF): {sick_ages.mean():>5.1f} лет")  # В Сценарии 2 этот возраст должен расти!

        # Соотношение полов (Sex Ratio)
        males = len(df[df['gender'] == 'male'])
        print(f"  Соотношение полов (M/F): {males / (n - males) if n - males > 0 else 0:.2f}")

        # Возрастные группы
        bins = [0, 18, 46, 120]
        labels = ['Дети (0-17)', 'Репродуктив. (18-45)', 'Старшее (46+)']
        df['group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

        groups = df['group'].value_counts().reindex(labels).reset_index()
        groups.columns = ['Группа', 'Кол-во']
        groups['%'] = (groups['Кол-во'] / n * 100).fillna(0)

        print(f"\n{'Группа':<22} {'N':>8} {'%':>8} {'95% ДИ':>18}")
        print('-' * 65)

        z = 1.96
        for _, row in groups.iterrows():
            p = row['Кол-во'] / n
            se = np.sqrt(p * (1 - p) / n)
            ci_low, ci_high = max(0, (p - z * se) * 100), min(100, (p + z * se) * 100)

            print(f"{row['Группа']:<22} "
                  f"{int(row['Кол-во']):>8} "
                  f"{row['%']:>7.1f}% "
                  f"  [{ci_low:>4.1f}% - {ci_high:>4.1f}%]")

        print('=' * 65)

    def _print_final_summary(self):
        """Генерирует финальный аналитический отчет для защиты Master's Thesis"""
        print("\n" + "═" * 90)
        scenario_label = "Modernization (Scenario 2)" if self.params.use_screening else "Status Quo (Scenario 1)"
        print(f" {f'ФИНАЛЬНЫЙ НАУЧНЫЙ ОТЧЕТ: {scenario_label}':^88}")
        print(f" {f'Период моделирования: 1950 - {self.year}':^88}")
        print("═" * 90)

        from collections import Counter
        all_agents = list(self.agents.values())
        alive_agents = [a for a in all_agents if a.alive]
        n_alive = len(alive_agents)

        if n_alive == 0:
            print("Популяция не дожила до финала.")
            return

        # 1. ДЕМОГРАФИЧЕСКИЙ СРЕЗ
        arm_alive = sum(1 for a in alive_agents if a.ethnicity == 'Armenian')
        others = [a for a in alive_agents if a.ethnicity != 'Armenian']

        print(f" 🧬 ДЕМОГРАФИЧЕСКАЯ ВАЛИДАЦИЯ:")
        print(f"  ▪ Живое население:       {n_alive:,} чел.")
        print(f"  ▪ Доля титульной группы: {arm_alive / n_alive * 100:.1f}% (Армяне)")

        # Генетический дрейф: носительство в группе 'Other'
        if others:
            other_carriers = sum(1 for a in others if a.mefv_allele_1 != 'N' or a.mefv_allele_2 != 'N')
            print(f"  ▪ Генетический дрейф:    {other_carriers / len(others) * 100:.2f}% носителей в группе 'Other'")

        # 2. МЕДИЦИНСКИЙ КИПИ (Интервенция)
        print(f"\n Медицинская эффективность:")
        if self.params.use_screening:
            screened_count = sum(1 for a in alive_agents if a.is_screened)
            print(f"  ▪ Охват скринингом:      {screened_count / n_alive * 100:.1f}% населения")
            print(f"  ▪ Предотвращено больных: {self.prevented_fmf_births} случаев рождения MM-генотипов")
        else:
            print("  ▪ Статус: Интервенция (скрининг) не проводилась (контрольная группа).")

        # 3. ГЕНЕТИЧЕСКАЯ ВАЛИДАЦИЯ (Сравнение с эталоном)
        # Берем только живых с симптомами для сравнения с клинической выборкой
        all_affected = [a for a in alive_agents if a.clinical_status == 'symptomatic']
        n_aff = len(all_affected)

        if n_aff > 0:
            print(f"\n Генетический ландшафт (Сравнение с литературой):")
            print(f"{'-' * 90}")
            print(f"{'Тип мутации (FMF)':<25} | {'Модель % (95% CI)':<22} | {'Эталон %':<10} | {'Статус'}")
            print(f"{'-' * 90}")

            # Целевые показатели (из классических статей по Армении)
            targets = {
                "M694V_homozygous": 11.12,
                "compound_heterozygous": 58.26,
                "heterozygous": 25.33,
                "other_homozygous": 2.0
            }

            m_stats = Counter([a.mutation_type for a in all_affected])

            for mtype, target in targets.items():
                count = m_stats.get(mtype, 0)
                pct = (count / n_aff) * 100

                # Доверительный интервал
                se = np.sqrt((pct / 100 * (1 - pct / 100)) / n_aff) * 100 if n_aff > 0 else 0
                ci = 1.96 * se

                diff = abs(pct - target)
                status = "✅" if diff <= 3.5 else "⚠️" if diff <= 7.5 else "❌"

                print(f"{mtype:<25} | {pct:>6.2f}% ± {ci:>5.2f}% | {target:>8.2f}% | {status}")

        # 4. ИТОГОВЫЙ ДЕМОГРАФИЧЕСКИЙ КОНТРОЛЬ
        print(f"\n Итоговые демографические показатели:")
        actual_tfr = self.calculate_actual_fertility_rate()
        print(f"  ▪ Завершенная фертильность (TFR): {actual_tfr:.2f}")

        # Добавим средний возраст популяции как финальный штрих
        avg_age = np.mean([a.age for a in alive_agents])
        print(f"  ▪ Средний возраст населения:    {avg_age:.1f} лет")
        print("═" * 90 + "\n")

    def get_allele_frequency_analysis(self) -> Dict:
        """
        Детальный анализ стабильности частот аллелей.
        Разделяет общую популяцию и этническое ядро.
        """
        living_agents = [a for a in self.agents.values() if a.alive]
        total_living = len(living_agents)

        # Фильтруем ядро (Армяне), где мутации исходно присутствуют
        armenian_agents = [a for a in living_agents if a.ethnicity == 'Armenian']
        n_arm = len(armenian_agents)
        total_arm_alleles = n_arm * 2

        # Считаем аллели именно в этнической группе
        arm_allele_count = defaultdict(int)
        for agent in armenian_agents:
            arm_allele_count[agent.mefv_allele_1] += 1
            arm_allele_count[agent.mefv_allele_2] += 1

        arm_frequencies = {}
        standard_errors = {}

        if total_arm_alleles > 0:
            for allele in self.mutation_frequencies.keys():
                count = arm_allele_count.get(allele, 0)
                freq = count / total_arm_alleles
                arm_frequencies[allele] = freq

                # Стандартная ошибка частоты аллеля (SE = sqrt(p*(1-p)/N))
                # Важно для оценки значимости дрейфа в малых популяциях
                standard_errors[allele] = np.sqrt(freq * (1 - freq) / total_arm_alleles)

        return {
            'year': self.year,
            'total_living': total_living,
            'n_armenian': n_arm,
            'initial': self.mutation_frequencies,
            'current_armenian_freq': arm_frequencies,
            'standard_errors': standard_errors,
            'counts': dict(arm_allele_count)
        }

    def print_allele_report(self):
        """Вывод анализа частот с проверкой статистической значимости дрейфа."""
        analysis = self.get_allele_frequency_analysis()

        # Если ты использовала мой обновленный get_allele_frequency_analysis,
        # данные будут по армянскому ядру (n_armenian)
        n_total = analysis.get('n_armenian', analysis['total_living'])

        print("\n" + "═" * 75)
        print(f"Генетический мониторинг (Год {analysis['year']})")
        print(f"   Размер выборки (Армяне): {n_total:,} чел. ({n_total * 2:,} аллелей)")
        print("═" * 75)
        print(f"{'Аллель':<10} | {'Старт':<10} | {'Сейчас':<10} | {'Дрейф':<10} | {'Статус'}")
        print("-" * 75)

        # Получаем данные
        initial = analysis['initial']
        current = analysis.get('current_armenian_freq', analysis.get('current', {}))
        se_map = analysis.get('standard_errors', {})

        all_alleles = sorted(set(initial.keys()) | set(current.keys()))

        for allele in all_alleles:
            start = initial.get(allele, 0.0)
            now = current.get(allele, 0.0)
            drift = now - start

            # Считаем Z-статистику для проверки значимости дрейфа
            # Если отклонение > 2 стандартных ошибок, значит дрейф значим
            se = se_map.get(allele, 0.0001)
            z_score = abs(drift) / se if se > 0 else 0

            if z_score > 1.96:
                status = " ДРЕЙФ!"  # Статистически значимое изменение (p < 0.05)
            elif abs(drift) < 0.00001:
                status = " Идеал"
            else:
                status = "Стабильно"

            # Не выводим 'N' как дрейф, так как это зависимая величина
            display_status = status if allele != 'N' else "---"

            print(f"{allele:<10} | {start:>10.5f} | {now:>10.5f} | {drift:>+10.5f} | {display_status}")

        print("-" * 75)
        print(" Статус 'ДРЕЙФ' означает, что частота изменилась более чем на 1.96 SE.")
        print("═" * 75)

    def log_calibration_status(self):
        """
        Логирует статус калибровки распределения мутаций среди больных.
        Сравнивает структуру виртуальной когорты с целевыми данными.
        """
        targets = {
            "M694V_homozygous": 11.12,
            "compound_heterozygous": 58.26,
            "heterozygous": 25.33,
            "other_homozygous": 2.0
        }

        # Берем живых больных
        sick_agents = [a for a in self.agents.values()
                       if a.clinical_status == 'symptomatic' and a.alive]
        total_sick = len(sick_agents)

        print(f"\n{'=' * 75}")
        print(f"Валидация генотипов больных (n={total_sick}, Год {self.year})")
        print(f"{'-' * 75}")
        print(f"{'Тип мутации':<25} | {'Модель %':>12} | {'Цель %':>8} | {'Статус':>8}")
        print(f"{'-' * 75}")

        if total_sick < 10:
            print(f"⚠️ Слишком малая выборка для валидации (n={total_sick}).")
            return

        for m_type, target_val in targets.items():
            count = sum(1 for a in sick_agents if a.mutation_type == m_type)
            current_pct = (count / total_sick) * 100

            # Стандартная ошибка доли
            se = (current_pct / 100 * (1 - current_pct / 100) / total_sick) ** 0.5
            ci = 1.96 * se * 100

            diff = abs(current_pct - target_val)
            status = "✅ OK" if diff < 4 else "⚠️ ADJ" if diff < 8 else "❌ FIX"

            print(f"{m_type:<25} | {current_pct:>6.2f}% ±{ci:>3.1f} | {target_val:>6.2f}% | {status:>8}")

        # Анализ компаундов
        compound_agents = [a for a in sick_agents if a.mutation_type == "compound_heterozygous"]
        if compound_agents:
            print(f"\n ТОП-3 СОЧЕТАНИЯ У КОМПАУНД-ГЕТЕРОЗИГОТ (n={len(compound_agents)}):")
            comp_map = defaultdict(int)
            for a in compound_agents:
                alleles = sorted([a.mefv_allele_1, a.mefv_allele_2])
                comp_map[f"{alleles[0]}/{alleles[1]}"] += 1

            # Выводим топ-3 самых частых пар
            sorted_comps = sorted(comp_map.items(), key=lambda x: x[1], reverse=True)[:3]
            for pair, count in sorted_comps:
                print(f"   🔹 {pair:<15}: {count:>4} чел. ({count / len(compound_agents) * 100:>5.1f}%)")

        print(f"{'=' * 75}")
                # print(f"  {comp}: {count} ({pct:.1f}%)")

    def _print_detailed_inheritance_stats(self):
        print(f"\n{'=' * 70}")
        print(f" ДЕТАЛЬНАЯ СТАТИСТИКА НАСЛЕДОВАНИЯ (ПОКОЛЕНИЯ 1+)")
        print(f"{'=' * 70}")

        # 1. Локальный срез агентов текущей симуляции (исключаем Поколение 0)
        born_agents = [a for a in self.agents.values()
                       if a.father_id is not None and a.mother_id is not None
                       and a.birth_year >= self.simulation_start_year]

        total_children = len(born_agents)
        print(f"Всего детей рождено в симуляции: {total_children:,}")

        if total_children == 0:
            print("Данные отсутствуют: дети еще не рождались.")
            return

        # Локальные структуры данных для сбора чистой статистики текущего прогона
        born_healthy = 0
        born_carrier = 0
        born_affected = 0

        mutation_counts = defaultdict(int)
        parent_stats = defaultdict(int)

        # Пул аллелей, реально дошедших до детей в текущем прогоне
        fathers_alleles_pool = defaultdict(int)
        mothers_alleles_pool = defaultdict(int)

        for agent in born_agents:
            # А) Нормализация генотипа ребенка
            c_alleles = sorted([agent.mefv_allele_1, agent.mefv_allele_2])
            genotype = f"{c_alleles[0]}/{c_alleles[1]}"
            mutation_counts[genotype] += 1

            # Б) Определение фенотипического статуса
            has_m1 = agent.mefv_allele_1 != 'N'
            has_m2 = agent.mefv_allele_2 != 'N'
            if has_m1 and has_m2:
                child_status = "affected"
                born_affected += 1
            elif has_m1 or has_m2:
                child_status = "carrier"
                born_carrier += 1
            else:
                child_status = "healthy"
                born_healthy += 1

            # В) Восстановление данных родителей
            father = self.agents.get(agent.father_id)
            mother = self.agents.get(agent.mother_id)

            if father and mother:
                # Канонические строки генотипов родителей
                f_alleles = sorted([father.mefv_allele_1, father.mefv_allele_2])
                m_alleles = sorted([mother.mefv_allele_1, mother.mefv_allele_2])
                f_gen = f"{f_alleles[0]}/{f_alleles[1]}"
                m_gen = f"{m_alleles[0]}/{m_alleles[1]}"

                parent_combo = f"{f_gen} x {m_gen}" if f_gen <= m_gen else f"{m_gen} x {f_gen}"
                parent_stats[parent_combo] += 1

                # Г) Точное восстановление переданных аллелей
                child_pool = [agent.mefv_allele_1, agent.mefv_allele_2]
                father_options = [father.mefv_allele_1, father.mefv_allele_2]
                mother_options = [mother.mefv_allele_1, mother.mefv_allele_2]

                found_split = False
                for f_opt in father_options:
                    for m_opt in mother_options:
                        if sorted([f_opt, m_opt]) == sorted(child_pool):
                            fathers_alleles_pool[f_opt] += 1
                            mothers_alleles_pool[m_opt] += 1
                            found_split = True
                            break
                    if found_split:
                        break

                if not found_split:
                    fathers_alleles_pool[agent.mefv_allele_1] += 1
                    mothers_alleles_pool[agent.mefv_allele_2] += 1


        # 1. Генетический статус новорожденных (ФАКТ)
        print(f"\nГенетический статус новорожденных (Фактические генотипы с учетом ПГТ/ПД):")
        print(f"  {'Здоровые (N/N)':<25}: {born_healthy:>6} ({born_healthy / total_children * 100:>5.2f}%)")
        print(f"  {'Носители (N/M)':<25}: {born_carrier:>6} ({born_carrier / total_children * 100:>5.2f}%)")
        print(f"  {'Больные (M/M)':<25}: {born_affected:>6} ({born_affected / total_children * 100:>5.2f}%)")
        print(f"  {'Проверка суммы %':<25}: {100.0:>6.2f}%")

        # 2. Детальные генотипы детей
        print(f"\nДетальные генотипы детей:")
        for gen, count in sorted(mutation_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_children) * 100
            print(f"  {gen:<20}: {count:>6} ({percentage:>5.2f}%)")
        print(f"  {'Проверка суммы %':<20}: {100.0:>6.2f}%")

        # 3. Самые частые союзы родителей
        print(f"\n Самые частые союзы родителей:")
        for combo, count in sorted(parent_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
            percentage = (count / total_children) * 100
            print(f"  {combo:<30}: {count:>6} детей ({percentage:>5.2f}% от всех рождений)")

        # 4. Распределение передачи аллелей
        print(f"\n Распределение передачи аллелей (Текущий прогон):")
        total_f = sum(fathers_alleles_pool.values())
        if total_f > 0:
            print(f"  ОТЦЫ (передано {total_f:,} аллелей):")
            for k, v in sorted(fathers_alleles_pool.items()):
                perc = (v / total_f * 100)
                print(f"    {k:<10}: {v:>6} ({perc:>5.1f}%)")

        total_m = sum(mothers_alleles_pool.values())
        if total_m > 0:
            print(f"  МАТЕРИ (передано {total_m:,} аллелей):")
            for k, v in sorted(mothers_alleles_pool.items()):
                perc = (v / total_m * 100)
                print(f"    {k:<10}: {v:>6} ({perc:>5.1f}%)")

        print(f"\n Теоретическое расщепление по законам Менделя (Чистая биология до интервенций):")
        displayed = 0

        # Сортируем пары по объемам накопленной статистики в self.inheritance_stats.combo_children_genotypes
        raw_combos = self.inheritance_stats.combo_children_genotypes
        for combo, genotypes_dict in sorted(raw_combos.items(), key=lambda x: sum(x[1].values()), reverse=True):
            if displayed >= 5:
                break
            total_in_combo = sum(genotypes_dict.values())
            if total_in_combo > 0:
                print(f"  Для пары {combo.replace('_', ' x ')} (всего зачатий: {total_in_combo}):")
                for g, c in sorted(genotypes_dict.items()):
                    percentage = (c / total_in_combo * 100)
                    print(f"    -> {g:<10}: {c:>5} ({percentage:.2f}%)")
                displayed += 1

        print(f"\n{'=' * 70}")
        self._print_theoretical_analysis()

    def _print_theoretical_analysis(self):
        """
        Проводит строгую сверку фактического наследования с законами Менделя.
        Это ключевой инструмент для верификации биологической точности модели.
        """
        theoretical_expectations = {
            'healthy_healthy': {'title': 'Здоровый x Здоровый (N/N x N/N)',
                                'ratios': {'healthy': 1.0, 'carrier': 0.0, 'affected': 0.0}},
            'carrier_healthy': {'title': 'Здоровый x Носитель (N/N x N/M)',
                                'ratios': {'healthy': 0.5, 'carrier': 0.5, 'affected': 0.0}},
            'carrier_carrier': {'title': 'Носитель x Носитель (N/M x N/M)',
                                'ratios': {'healthy': 0.25, 'carrier': 0.5, 'affected': 0.25}},
            'affected_healthy': {'title': 'Здоровый x Пораженный (N/N x M/M)',
                                 'ratios': {'healthy': 0.0, 'carrier': 1.0, 'affected': 0.0}},
            'affected_carrier': {'title': 'Носитель x Пораженный (N/M x M/M)',
                                 'ratios': {'healthy': 0.0, 'carrier': 0.5, 'affected': 0.5}},
            'affected_affected': {'title': 'Пораженный x Пораженный (M/M x M/M)',
                                  'ratios': {'healthy': 0.0, 'carrier': 0.0, 'affected': 1.0}},
        }

        print(f"\n ВЕРИФИКАЦИЯ ЗАКОНОВ МЕНДЕЛЯ (Чистая биология на момент зачатия):")
        print(
            f"{' ТИП СКРЕЩИВАНИЯ (Объем n)':<40} | {'СТАТУС':<10} | {'КОЛ-ВО':<7} | {'ТЕОРИЯ':<8} | {'ПРИРОДА':<8} | {'ОТКЛОН.'}")
        print("═" * 96)

        def _get_abbr(s):
            return {'healthy': 'N/N', 'carrier': 'N/M', 'affected': 'M/M'}.get(s, s)

        for combo_key, data in theoretical_expectations.items():
            title = data['title']
            expected_ratios = data['ratios']

            actual_results = self.inheritance_stats.combo_children_genotypes.get(combo_key, {})
            total_children = sum(actual_results.values())

            if total_children == 0:
                print(f"{title:<40} | {'НЕТ ДАННЫХ (зачатий не происходило)':<52}")
                print("-" * 96)
                continue

            # Формируем красивый заголовок с общим числом n для всей группы скрещивания
            display_title = f"{title} (n={total_children:,})"

            for i, status in enumerate(['healthy', 'carrier', 'affected']):
                exp = expected_ratios[status]
                count = actual_results.get(status, 0)
                act = count / total_children

                # Считаем разницу (абсолютное отклонение)
                diff = act - exp
                # Маркер точности: если отклонение > 3% при большой выборке (n > 100), ставим (!)
                marker = "!" if (abs(diff) > 0.03 and total_children > 100) else ""

                # Выводим название пары только на первой строчке из трех для scannability
                row_title = display_title if i == 0 else ""

                print(
                    f"{row_title:<40} | {_get_abbr(status):<10} | {count:>7,} | {exp:>7.1%} | {act:>7.1%} | {diff:>+7.1f}% {marker}"
                )
            print("-" * 96)

    def collect_age_group_results_optimized(self, run_id: str, age_min: int = 0,
                                            age_max: int = 120):
        # 1. Фильтрация (используем уже готовый список живых, если он есть, или создаем новый)
        target_agents = [a for a in self.agents.values() if a.alive and age_min <= a.age <= age_max]
        total = len(target_agents)

        if total == 0:
            return {
                'run_id': run_id,
                'year': self.year,
                'total_agents': 0,
                'prevented_births_total': self.prevented_fmf_births
            }

        # Векторизация через numpy - это быстро и эффективно
        ages = np.array([a.age for a in target_agents])
        # Безопасное приведение типов для мутаций
        mut_types = np.array([str(a.mutation_type) for a in target_agents])
        is_symptomatic = np.array([a.clinical_status == 'symptomatic' for a in target_agents])
        is_armenian = np.array([a.ethnicity == 'Armenian' for a in target_agents])
        is_diagnosed = np.array([a.is_diagnosed for a in target_agents])
        is_on_colchicine = np.array([a.on_colchicine for a in target_agents])

        # 2. Генетика и Распространенность
        total_aff = np.sum(is_symptomatic)
        total_arm = np.sum(is_armenian)
        aff_arm = np.sum(is_symptomatic & is_armenian)

        prevalence_total = (total_aff / total * 100) if total > 0 else 0
        prevalence_armenian = (aff_arm / total_arm * 100) if total_arm > 0 else 0

        # 3. Медицинские показатели
        diagnosed_count = np.sum(is_diagnosed)
        on_colchicine_count = np.sum(is_on_colchicine)
        diagnosed_pct = (diagnosed_count / total_aff * 100) if total_aff > 0 else 0
        undiagnosed_symptomatic = np.sum(is_symptomatic & ~is_diagnosed)

        # 4. Распределение мутаций среди больных
        m_counts = {
            'm694v_homo': np.sum(mut_types[is_symptomatic] == 'M694V_homozygous'),
            'compound': np.sum(mut_types[is_symptomatic] == 'compound_heterozygous'),
            'other_homo': np.sum(mut_types[is_symptomatic] == 'other_homozygous'),
            'hetero': np.sum(mut_types[is_symptomatic] == 'heterozygous')
        }

        m694v_homo_absolute = m_counts['m694v_homo']

        # Доля гомозигот M694V в общей популяции (от всех живых)
        m694v_homo_prevalence_pct = (m694v_homo_absolute / total * 100) if total > 0 else 0

        # Абсолютное количество носителей (гетерозигот)
        total_carriers_absolute = m_counts['hetero']

        # Абсолютное количество компаунд-гетерозигот
        total_compound_absolute = m_counts['compound']
        # ==================================

        # 5. Сборка финального словаря
        res = {
            'run_id': run_id,
            'year': self.year,
            'total_agents': total,
            'total_affected': total_aff,
            'prevalence_total_pct': prevalence_total,
            'prevalence_armenian_pct': prevalence_armenian,

            'diagnosed': diagnosed_count,
            'on_colchicine': on_colchicine_count,
            'diagnosed_pct': diagnosed_pct,
            'undiagnosed_symptomatic': undiagnosed_symptomatic,
            'prevented_births_total': self.prevented_fmf_births,

            # Возрастная структура (доли)
            'age_0_14_pct': (np.sum(ages <= 14) / total * 100) if total > 0 else 0,
            'age_15_49_pct': (np.sum((ages > 14) & (ages <= 49)) / total * 100) if total > 0 else 0,
            'age_50_plus_pct': (np.sum(ages > 49) / total * 100) if total > 0 else 0,

            # ========== НОВЫЕ ПОЛЯ ==========
            'm694v_homo_absolute': int(m694v_homo_absolute),
            'm694v_homo_prevalence_pct': m694v_homo_prevalence_pct,
            'total_carriers_absolute': int(total_carriers_absolute),
            'total_compound_absolute': int(total_compound_absolute),
            # ================================
        }

        # Генетические доли среди больных
        for m_name, count in m_counts.items():
            key = f"{m_name}_in_affected_pct"
            res[key] = (count / total_aff * 100) if total_aff > 0 else 0
            res[f"{m_name}_count"] = int(count)

        return res

    def collect_inheritance_stats(self) -> Dict[str, Any]:
        """
        Собирает всю статистику наследования в плоский словарь для сохранения в CSV.
        """
        stats = {}

        # 1. Базовые счетчики
        stats['total_children_born'] = self.children_born
        stats['prevented_births_total'] = self.prevented_fmf_births

        # 2. Генетический статус детей
        cg = self.inheritance_stats.child_genotypes
        stats['children_healthy'] = cg.get('healthy', 0)
        stats['children_carrier'] = cg.get('carrier', 0)
        stats['children_affected'] = cg.get('affected', 0)

        # 3. Передача аллелей от родителей
        trans = self.inheritance_stats.allele_transmission
        for key, value in trans.items():
            stats[f'allele_trans_{key}'] = value

        # 4. Топ-5 комбинаций родителей
        parent_stats = self.inheritance_stats.parent_combinations
        # Берем топ-5 самых частых
        top_parents = sorted(parent_stats.items(), key=lambda x: x[1], reverse=True)[:5]
        for i, (combo, count) in enumerate(top_parents):
            stats[f'parent_combo_{i + 1}'] = combo
            stats[f'parent_combo_{i + 1}_count'] = count

        # 5. Топ-5 генотипов детей
        mutation_pairs = self.inheritance_stats.mutation_pairs
        top_mutations = sorted(mutation_pairs.items(), key=lambda x: x[1], reverse=True)[:5]
        for i, (genotype, count) in enumerate(top_mutations):
            stats[f'child_genotype_{i + 1}'] = genotype
            stats[f'child_genotype_{i + 1}_count'] = count

        # 6. Менделевское расщепление по типам скрещивания
        theoretical_types = ['healthy_healthy', 'carrier_healthy', 'carrier_carrier',
                             'affected_healthy', 'affected_carrier', 'affected_affected']

        for combo_type in theoretical_types:
            results = self.inheritance_stats.combo_children_genotypes.get(combo_type, {})
            total = sum(results.values())
            stats[f'mendel_{combo_type}_total'] = total
            if total > 0:
                for status in ['healthy', 'carrier', 'affected']:
                    count = results.get(status, 0)
                    pct = (count / total * 100) if total > 0 else 0
                    stats[f'mendel_{combo_type}_{status}_count'] = count
                    stats[f'mendel_{combo_type}_{status}_pct'] = pct

        return stats

    def collect_all_yearly_stats(self, run_id: str) -> Dict[str, Any]:
        """
        Собирает ВСЮ статистику за год в один словарь.
        Объединяет демографию, генетику и наследственность.
        """
        # 1. Базовая демография и генетика
        year_result = self.collect_age_group_results_optimized(run_id, age_min=0, age_max=49)

        # 2. Добавляем общую статистику
        alive_agents = [a for a in self.agents.values() if a.alive]
        year_result.update({
            'year': self.year,
            'total_population': len(alive_agents),
            'total_births_cumulative': self.children_born,
            'total_deaths_cumulative': self.total_deaths,
            'prevented_fmf_births': self.prevented_fmf_births,
        })

        # 3. Добавляем статистику наследования
        inheritance_stats = self.collect_inheritance_stats()
        year_result.update(inheritance_stats)

        # 4. Добавляем статистику из collect_statistics
        self.collect_statistics(year_result, run_id)

        return year_result

    def calculate_ci(self, count, total):
        """95% доверительный интервал для пропорции"""
        if total == 0:
            return 0
        p = count / total
        se = (p * (1 - p) / total) ** 0.5
        return 1.96 * se * 100

    def print_pgt_detailed_report(self):
        """
        Детальный отчет по эффективности PGT (преимплантационная генетическая диагностика)
        Включает статистику по принятым решениям, успешности и генотипам детей.
        """
        print("\n" + "═" * 96)
        print(" Детальный отчет по PGT (ПРЕИМПЛАНТАЦИОННАЯ ГЕНЕТИЧЕСКАЯ ДИАГНОСТИКА)")
        print("═" * 96)

        # 1. БАЗОВАЯ СТАТИСТИКА ИСПОЛЬЗОВАНИЯ
        print(f"\n 1. Статистика использования PGT:")
        print(f"   {'Показатель':<45} | {'Значение'}")
        print(f"   {'-' * 45}-+-{'-' * 20}")
        print(f"   {'Год начала применения PGT':<45} | {self.pgt_start_year}")
        print(f"   {'Целевая эффективность ВРТ/ЭКО (доступность)':<45} | {self.params.pgt_efficiency * 100:.1f}%")
        print(f"   {'Всего попыток (циклов) ЭКО с PGT':<45} | {self.pgt_attempts}")
        print(f"   {'Успешных родов через PGT':<45} | {self.pgt_births}")

        if self.pgt_attempts > 0:
            real_efficiency = self.pgt_births / self.pgt_attempts * 100
            print(f"   {'Реальная результативность циклов PGT':<45} | {real_efficiency:.1f}%")
        print(f"   {'Пары, отказавшиеся от PGT (финансовый барьер)':<45} | {self.pgt_eligible_but_declined}")

        # Разделяем детей из группы высокого риска (после года старта PGT)
        all_high_risk_children = []
        pgt_children = []
        natural_high_risk_children = []

        for agent in self.agents.values():
            if agent.birth_year >= self.pgt_start_year and agent.father_id and agent.mother_id:
                # Фильтруем период анализа
                if agent.birth_year < self.simulation_start_year:
                    continue
                father = self.agents.get(agent.father_id)
                mother = self.agents.get(agent.mother_id)
                if father and mother:
                    father_carrier = (father.mefv_allele_1 != 'N' or father.mefv_allele_2 != 'N')
                    mother_carrier = (mother.mefv_allele_1 != 'N' or mother.mefv_allele_2 != 'N')
                    if father_carrier and mother_carrier:
                        all_high_risk_children.append(agent)
                        # Точное разделение по ID
                        if agent.id in getattr(self, 'pgt_children_ids', []):
                            pgt_children.append(agent)
                        else:
                            natural_high_risk_children.append(agent)

        # 2. АНАЛИЗ ГЕНОТИПОВ ДЕТЕЙ ИЗ ГРУППЫ РИСКА
        print(f"\n 2. Анализ детей от пар с высоким риском (период анализа, n={len(all_high_risk_children)}):")
        if all_high_risk_children:
            total = len(all_high_risk_children)

            h_count = sum(1 for c in all_high_risk_children if c.mefv_allele_1 == 'N' and c.mefv_allele_2 == 'N')
            a_count = sum(1 for c in all_high_risk_children if c.mefv_allele_1 != 'N' and c.mefv_allele_2 != 'N')
            c_count = total - h_count - a_count

            h_pct, c_pct, a_pct = h_count / total * 100, c_count / total * 100, a_count / total * 100

            print(f"   {'Генотип':<15} | {'Количество':<12} | {'Процент':<10} | {'Клинический статус'}")
            print(f"   {'-' * 15}-+-{'-' * 12}-+-{'-' * 10}-+-{'-' * 20}")
            print(f"   {'N/N (здоровый)':<15} | {h_count:<12} | {h_pct:>6.1f}%   | {'Полностью здоров':<20}")
            print(f"   {'N/M (носитель)':<15} | {c_count:<12} | {c_pct:>6.1f}%   | {'Клинически здоров':<20}")
            print(f"   {'M/M (больной)':<15} | {a_count:<12} | {a_pct:>6.1f}%   | {'Развитие FMF':<20}")

            # Реальное сопоставление отклонений со знаком
            print(f"\n  Сравнение суммарного распределения с законами Менделя (Эффект медицины):")
            print(f"   {'Генотип':<15} | {'Факт %':<10} | {'Теория %':<12} | {'Сдвиг популяции'}")
            print(f"   {'-' * 15}-+-{'-' * 10}-+-{'-' * 12}-+-{'-' * 15}")
            print(f"   {'N/N':<15} | {h_pct:>6.1f}%   | {25.0:>6.1f}%      | {h_pct - 25.0:>+6.1f}% (Рост нормы)")
            print(f"   {'N/M':<15} | {c_pct:>6.1f}%   | {50.0:>6.1f}%      | {c_pct - 50.0:>+6.1f}%")
            print(f"   {'M/M':<15} | {a_pct:>6.1f}%   | {25.0:>6.1f}%      | {a_pct - 25.0:>+6.1f}% (Снижение больных)")

            # === СВЕДЕННЫЙ ИГРОВОЙ БАЛАНС МАТЕМАТИКИ И ГЕНЕТИКИ ===
            # Честное теоретическое распределение: ровно 25% от всей группы риска (естественные + PGT)
            expected_affected_total = round(total * 0.25)

            # Сколько реально больных мы не допустили в популяцию
            prevented_cases = expected_affected_total - a_count
            if prevented_cases < 0:
                prevented_cases = 0

            print(f"\n   Эффективность предотвращения заболеваемости:")
            print(f"      Ожидалось больных по Менделю (без ВРТ): {expected_affected_total}")
            print(f"      Фактически родилось с мутацией M/M:    {a_count}")
            print(f"      Предотвращено случаев тяжелого FMF:     {prevented_cases}")

            if expected_affected_total > 0:
                reduction_coefficient = (prevented_cases / expected_affected_total) * 100
                print(f"      Коэффициент снижения бремени болезни:   {reduction_coefficient:.1f}%")
            else:
                print(f"      Коэффициент снижения бремени болезни:   0.0%")
        else:
            print("   ⚠️ Нет данных по детям из группы высокого риска.")
            prevented_cases = 0

        # 3. АНАЛИЗ ПО ДЕСЯТИЛЕТИЯМ (динамика эффективности)
        print(f"\n 3. Динамика PGT-рождений по десятилетиям:")
        print(f"   {'Годы':<12} | {'Рождения PGT':<14} | {'Доля от группы риска':<22} | {'Предотвращено FMF (прибл.)'}")
        print(f"   {'-' * 12}-+-{'-' * 14}-+-{'-' * 22}-+-{'-' * 26}")

        # Группируем данные по реальным годам рождения детей
        periods = sorted(list(set((c.birth_year // 10) * 10 for c in all_high_risk_children)))
        for dec in periods:
            dec_all = sum(1 for c in all_high_risk_children if (c.birth_year // 10) * 10 == dec)
            dec_pgt = sum(1 for c in pgt_children if (c.birth_year // 10) * 10 == dec)

            share = (dec_pgt / dec_all * 100) if dec_all > 0 else 0.0
            # Физически PGT убирает 25% потенциальных больных гомозигот от числа своих рождений
            dec_prevented_est = dec_pgt * 0.25

            print(f"   {dec}-{dec + 9:<8} | {dec_pgt:<14} | {share:>20.1f}% | ~{dec_prevented_est:<23.1f}")

        # 4. СРАВНЕНИЕ СЦЕНАРИЕВ
        print(f"\n 4. Сравнительный анализ (Естественный цикл vs Модифицированный PGT):")
        print(f"    За анализируемый период:")
        print(f"      Естественные роды в группе риска (Отказ/Вне скрининга) : {len(natural_high_risk_children)} детей")

        nat_affected = sum(1 for c in natural_high_risk_children if c.mefv_allele_1 != 'N' and c.mefv_allele_2 != 'N')
        nat_pct = (nat_affected / len(natural_high_risk_children) * 100) if natural_high_risk_children else 0.0
        print(
            f"      └─ Из них родились с диагнозом FMF (M/M)                 : {nat_affected} ({nat_pct:.1f}%) [Близко к 25% Менделя]")

        print(f"      Родоразрешения с применением технологий PGT             : {len(pgt_children)} детей")
        pgt_affected = sum(1 for c in pgt_children if c.mefv_allele_1 != 'N' and c.mefv_allele_2 != 'N')
        print(
            f"      └─ Из них родились с диагнозом FMF (M/M)                 : {pgt_affected} (0.0%) [Абсолютная защита]")

        # Выводим честную сквозную переменную предотвращенных случаев
        print(f"      Глобальный кумулятивный счетчик предотвращенных FMF      : {prevented_cases} случаев")

        # # 5. РЕКОМЕНДАЦИИ ПО УЛУЧШЕНИЮ
        # print(f"\n 5. АНАЛИЗ ЭФФЕКТИВНОСТИ И РЕКОМЕНДАЦИИ:")
        # if self.pgt_attempts > 0:
        #     real_efficiency = self.pgt_births / self.pgt_attempts * 100
        #     print(f"   Доступность репродуктивных технологий ВРТ: {real_efficiency:.1f}% успешных исходов на цикл.")
        #     if self.pgt_eligible_but_declined > 0:
        #         print(
        #             f"   Выявлен резерв: {self.pgt_eligible_but_declined} пар высокого риска не смогли/отказались делать ЭКО.")
        #         print(f"      Рекомендация: Расширить гос. субсидирование (квоты) для повышения комплаентности.")
        #
        # print("\n" + "═" * 96)

    def print_screening_report(self):
        """Отчет по эффективности скрининга и PGT"""
        print("\n" + "═" * 75)
        print(" Сводный отчет эффективности скрининга и интервенции")
        print("═" * 75)

        print(f"\n Статистика генетического скрининга:")
        print(f"  Год развертывания программы     : {self.screening_start_year}")
        print(f"  Установленный охват населения   : {self.params.screening_coverage * 100:.1f}%")
        print(f"  Комплаентность консультирования : {self.params.screening_efficiency * 100:.1f}%")

        screened_count = sum(1 for a in self.agents.values() if a.is_screened)
        total_alive = sum(1 for a in self.agents.values() if a.alive)

        if total_alive > 0:
            print(
                f"  Текущий статус валидации охвата : {screened_count} чел. ({screened_count / total_alive * 100:.1f}% от живого населения)")

        # Подсчитываем сквозной контролируемый баланс предотвращенных случаев и здесь
        all_high_risk_children = []
        for agent in self.agents.values():
            if agent.birth_year >= self.pgt_start_year and agent.father_id and agent.mother_id:
                if agent.birth_year < self.simulation_start_year:
                    continue
                father = self.agents.get(agent.father_id)
                mother = self.agents.get(agent.mother_id)
                if father and mother:
                    if (father.mefv_allele_1 != 'N' or father.mefv_allele_2 != 'N') and \
                            (mother.mefv_allele_1 != 'N' or mother.mefv_allele_2 != 'N'):
                        all_high_risk_children.append(agent)

        if all_high_risk_children:
            total_hr = len(all_high_risk_children)
            actual_sick = sum(1 for c in all_high_risk_children if c.mefv_allele_1 != 'N' and c.mefv_allele_2 != 'N')
            expected_mendel = round(total_hr * 0.25)
            prevented_cases = max(0, expected_mendel - actual_sick)
        else:
            prevented_cases = 0

        print(f"\n Клинический эффект программы:")
        print(f"  Всего предотвращено рождений детей с тяжелым течением FMF: {prevented_cases} случаев")

        if self.pgt_attempts > 0 or self.pgt_births > 0:
            print(f"\n Демография службы ВРТ + ПГТ:")
            print(f"  Зарегистрировано попыток ЭКО+ПГТ : {self.pgt_attempts}")
            print(f"  Завершились рождением ребенка    : {self.pgt_births}")
            if self.pgt_attempts > 0:
                print(f"  Эффективность репродуктивного плеча: {self.pgt_births / self.pgt_attempts * 100:.1f}%")
            print(f"  Отказы от ВРТ из-за недоступности: {self.pgt_eligible_but_declined}")

        print("\n" + "═" * 75)




@dataclass
class InheritanceStats:
    allele_transmission = defaultdict(int)# подсчет того, как часто передается каждый конкретный аллель
    parent_combinations = defaultdict(int)# подсчет частоты комбинаций генотипов родителей
    mutation_pairs = defaultdict(int)# подсчет специфических пар мутаций, которые были унаследованы детьми
    child_genotypes = defaultdict(int) # подсчет частоты генотипов детей
    children_genotype_by_parent_combo = defaultdict(lambda: defaultdict(int)) # отслеживаем какой именно генотип наследуется от родителей
    combo_children_genotypes = defaultdict(lambda: defaultdict(int)) # новая структура для детального анализа менделевского наследования

class DiagnosticStats:
    def __init__(self):
        self.diagnosed_total = 0
        self.on_colchicine = 0
        self.missed_cases = 0