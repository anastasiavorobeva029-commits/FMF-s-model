import pandas as pd
import numpy as np
import random
import uuid
from typing import Optional, Any, Union, Dict
from collections import defaultdict

import matplotlib.pyplot as plt


def data_load():

    files = {
        'birth_rate':'birth_rate.csv',
        'age_fertility_dist':'age_fertility_dist.csv',
        'age_structure_1950':'age_structure_1950.csv',
        'death_rate':'death_rate.csv',
        'fertility_rate':'fertility_rate.csv'
    }

    data = {} # здесь будет словарь с готовыми таблицами - название - ключ, таблица - значение
    for name, filename in files.items():

            df = pd.read_csv(filename, sep=';', skipinitialspace=True, decimal=',')

            if name in ['age_structure_1950', 'age_fertility_dist']:
                df.set_index(df.columns[0], inplace=True)
                df.index.name = 'Age_Group'
            else:
                df.set_index(df.columns[0], inplace=True)
                df.index.name = 'Year'
                df.columns = [name]

            data[name] = df

    return (
        data['birth_rate'],
        data['age_fertility_dist'],
        data['age_structure_1950'],
        data['death_rate'],
        data['fertility_rate']
    )

class Agent:
    def __init__(self, gender: str, age: int,
                 generation: int,
                 birth_year: int,
                 max_age_limit: int = 85,
                 ethnicity: str = 'Armenian',
                 father_id: Optional[str] = None,
                 mother_id: Optional[str] = None):

        self.id = str(uuid.uuid4())[:8]
        self.gender = gender
        self.age = age
        self.birth_year = birth_year
        self.max_age_limit = max_age_limit
        self.generation = generation
        self.ethnicity = ethnicity
        self.alive = True

        # Клинические параметры
        self.clinical_status = 'asymptomatic'
        self.disease_severity = None
        self.age_of_onset = None
        self.on_colchicine = False
        self.is_diagnosed = False

        # Генетика
        self.mefv_allele_1 = 'N'
        self.mefv_allele_2 = 'N'
        self.genotype_status = 'healthy'
        self.mutation_type = None

        # Семейные параметры
        self.father_id = father_id
        self.mother_id = mother_id
        self.partner_id = None
        self.children_ids = []
        self.last_birth_year = -100


    def set_genotype(self, allele_1: str, allele_2: str):

        self.mefv_allele_1 = allele_1
        self.mefv_allele_2 = allele_2

        self.update_genotype_status()

    def update_genotype_status(self):

        alleles = [self.mefv_allele_1, self.mefv_allele_2]
        mutant_count = 0

        for allele in alleles:
            if allele != 'N':
                mutant_count += 1

        if mutant_count == 0:
            self.genotype_status = 'healthy'
            self.mutation_type = None

        elif mutant_count == 1:
            self.genotype_status = 'carrier'
            self.mutation_type = 'heterozygous'

        else:
            self.genotype_status = 'at_risk'

            if len(set(alleles)) == 1:
                allele_type = alleles[0]

                if allele_type == "M694V":
                    self.mutation_type = 'M694V_homozygous'

                else:
                    self.genotype_status = 'other_homozygous'

            else:
                self.mutation_type = 'compound_heterozygous'

    def age_year(self, annual_death_prob: float, current_year: int):

        if not self.alive:
            return

        self.age += 1

        if self.age > self.max_age_limit:
            self.alive = False
            return

        if self.age < 1:
            age_weight = 2.0  # младенческая смертность
        elif self.age < 15:
            age_weight = 0.15  # низкая смертность в детстве
        elif self.age < 45:
            age_weight = 0.3  # взрослые
        elif self.age < 65:
            age_weight = 1.2  # средний возраст
        else:
            # Линейный рост смертности от 65 до max_age_limit
            base_weight = 2.5
            extra_years = self.age - 65
            max_extra = self.max_age_limit - 65
            age_weight = base_weight + (extra_years / max_extra) * 4.0

        if random.random() < annual_death_prob * age_weight:
            self.alive = False

        # сценарий 2

        if self.clinical_status == 'asymptomatic' and self.mutation_type:
            prob = self._calculate_annual_onset_probability()

            if self.age >= 40:
                prob *= 0.1

            if random.random() < prob:
                self.clinical_status = 'symptomatic'
                self.age_of_onset = self.age
                self._determine_disease_severity()

                self._try_to_diagnose(current_year)

            elif self.clinical_status == 'symptomatic' and not self.is_diagnosed:
                self._try_to_diagnose(current_year)

    def _try_to_diagnose(self, current_year: int):
        base_k = 0.011603 * current_year - 23.0295

        if self.age <= 18:
            access_rate = base_k * 1.3  # дети
        else:
            access_rate = base_k * 0.8  # взрослые

        access_rate = max(0.05, min(access_rate, 0.98))

        if random.random() < access_rate:
            self.is_diagnosed = True
            self.on_colchicine = True


    def _calculate_annual_onset_probability(self):

        if self.clinical_status != 'asymptomatic':
            return 0.0

        if self.mutation_type == "M694V_homozygous":
            return 0.0195

        elif self.mutation_type == "compound_heterozygous":
            return 0.0365

        elif self.mutation_type == "heterozygous":
            return 0.00032

        elif self.mutation_type == "other_homozygous":
            return 0.0045

        return 0.0

    def _determine_disease_severity(self):
        # Степень тяжести зависит от агрессивности мутаций
        if self.mutation_type == "M694V_homozygous":
            weights = [0.1, 0.3, 0.6]  # Преимущественно тяжелое
        elif self.mutation_type == "compound_heterozygous":
            # Если в компаунде есть M694V, течение тяжелее
            if "M694V" in [self.mefv_allele_1, self.mefv_allele_2]:
                weights = [0.2, 0.4, 0.4]
            else:
                weights = [0.4, 0.4, 0.2]
        else:  # Гетерозиготы и другие гомозиготы
            weights = [0.6, 0.3, 0.1]

        self.disease_severity = random.choices(["mild", "moderate", "severe"], weights=weights)[0]

    def set_partner(self, partner: 'Agent'):
        self.partner_id = partner.id
        partner.partner_id = self.id

    def add_child(self, child_id: str):  # ссылаемся на родителей и даем уникальный номер детям
        if child_id not in self.children_ids:  # проверяем нет ли номера ребенка в списке
            self.children_ids.append(child_id)

    def get_children_count(self): # возвращаем количество детей
        return len(self.children_ids)

    def can_get_pregnant(self, current_year: int, birth_cooldown: int) -> bool:

        if not self.alive or self.gender != 'female' or self.partner_id is None:
            return False

        if self.age < 18 or self.age > 45:
            return False

        if current_year - self.last_birth_year < birth_cooldown:
            return False

        # сценарий 2
        # у женщины есть симптомы, но нет лечения
        if self.clinical_status == 'symptomatic' and not self.on_colchicine:
            # мы знаем, что у женщин с нелеченым FMF фертильность снижена, поэтому
            # вводим вероятность того, что она не может забеременеть
            if random.random() < 0.30:
                return False

        return True

class GenerationSimulation:
    def __init__(self,
                 birth_rate_df,
                 death_rate_df,
                 tfr_df,
                 age_structure_df,
                 fertility_factors_df,
                 initial_population_size: int = 50000,
                 max_age_limit: int = 85,  # Увеличим для реалистичности
                 mutation_frequencies: Dict[str, float] = None,
                 ethnic_assortativity: float = 0.70):
        # 1. Сохраняем исторические таблицы
        self.birth_rate_data = birth_rate_df
        self.death_rate_data = death_rate_df
        self.tfr_data = tfr_df
        self.age_structure_1950 = age_structure_df

        # 2. Параметры времени
        self.current_year = 1950  # Стартуем с года данных
        self.initial_population_size = initial_population_size
        self.max_age_limit = max_age_limit

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

        # Репродуктивное окно
        self.reproductive_years = 28  # (45 - 18 + 1)

        # 5. Генетика и мутации (оставляем без изменений)
        self.mutation_frequencies = mutation_frequencies or {
            'N': 0.90427, 'M694V': 0.0437, 'V726A': 0.0292,
            'M680I': 0.0192, 'R761H': 0.00363
        }


        self.agents: Dict[str, Agent] = {}
        self.population_history = []
        self.children_born = 0
        self.total_deaths = 0

        self.annual_stats = []  # список словарей для графиков

        # счетчики, где будем хранить кол-во реальных больных, диагностированных, какой коэффициент выявляемости
        self.diagnostic_stats = {
            'diagnosed_total': 0,
            'on_colchicine': 0,
            'missed_cases': 0
        }

        self.family_children_count = defaultdict(int)
        self.last_birth_year = defaultdict(lambda: -100)

        self.generation_stats = defaultdict(lambda: {
            'total': 0, 'alive': 0, 'healthy': 0, 'carrier': 0, 'affected': 0
        })

        self.birth_cooldown = 2

        # сценарий 2
        # Коэффициенты регрессии (R^2 = 0.95)
        self.regression_slope = 0.011603
        self.regression_intercept = -23.0295

        # Параметры скрининга
        self.screening_coverage = 0.30  # 30% охват генетическим скринингом
        self.screening_start_year = 2024  # Год внедрения скрининга в модели

        self.initialize_founders_with_structure()

        self.inheritance_stats = {
            'allele_transmission': defaultdict(int),
            'parent_combinations': defaultdict(int),
            'mutation_pairs': defaultdict(int),
            'child_genotypes': defaultdict(int),
            'children_genotype_by_parent_combo': defaultdict(lambda: defaultdict(int)),
            'combo_children_genotypes': defaultdict(lambda: defaultdict(int))
        }

        self.ethnic_assortativity = ethnic_assortativity

    def _get_random_allele(self) -> str:
        return random.choices(
            list(self.mutation_frequencies.keys()),
            weights=list(self.mutation_frequencies.values())
        )[0]

    def initialize_founders_with_structure(self, verbose=False):
        """
        Создает население 1950 года, используя загруженные данные.
        """
        if verbose:
            print(f"Инициализация популяции 1950 года: {self.initial_population_size} агентов")

        age_structure_df = self.age_structure_1950

        # 1. Подготовка весов (предположим, индекс - это строка '0-4', '5-9' и т.д.)
        age_groups = []
        weights = []
        for weight, row in age_structure_df.iterrows():
            start = int(row['min_age'])
            end = int(row['max_age'])
            age_groups.append((start, end))
            weights.append(weight)  # Берем долю из первой колонки

        # 2. Основной цикл создания агентов
        for _ in range(self.initial_population_size):
            # Выбираем группу и конкретный возраст
            group = random.choices(age_groups, weights=weights)[0]
            age = random.randint(group[0], group[1])
            gender = random.choice(['male', 'female'])

            agent = Agent(
                gender = gender,
                age = age,
                birth_year = 1950 - age,
                generation = 0,
                max_age_limit = self.max_age_limit
            )

            # Генетика (Харди-Вайнберг для основателей)
            a1, a2 = self._get_random_allele(), self._get_random_allele()
            agent.set_genotype(a1, a2)

            real_age = agent.age
            start_year = agent.birth_year
            agent.age = 0

            for year_tick in range(real_age):
                current_tick_year = start_year + year_tick

                agent.age_year(annual_death_prob=0, current_year = current_tick_year)

            agent.age = real_age
            self.agents[agent.id] = agent

        # 3. Формирование семей (логика связей)
        self._form_initial_social_structure()

    def _form_initial_social_structure(self):
        males = [a for a in self.agents.values() if a.gender == 'male' and 18 <= a.age <= 50]
        females = [a for a in self.agents.values() if a.gender == 'female' and 18 <= a.age <= 45]
        children = [a for a in self.agents.values() if a.age < 18]

        random.shuffle(males)
        random.shuffle(females)

        n_pairs = min(len(males), len(females))
        active_couples = []

        for i in range(n_pairs):
            m, f = males[i], females[i]
            m.set_partner(f)
            active_couples.append((m, f))

            family_key = f"{m.id}_{f.id}"
            self.family_children_count[family_key] = 0

        children.sort(key=lambda x: x.age, reverse=True)

        for child in children:
            # 1. Сначала фильтруем по возрасту (как и было)
            potential_couples = [
                (m, f) for m, f in active_couples
                if 18 <= (f.age - child.age) <= 45
            ]

            # 2. Теперь применяем "биологический фильтр" (Сценарий 2)
            final_candidates = []
            for father, mother in potential_couples:
                # Был ли у матери нелеченый FMF в год рождения ребенка?
                was_sick_and_untreated = (
                        mother.clinical_status == 'symptomatic' and
                        (mother.age_of_onset is not None and mother.age_of_onset <= (mother.age - child.age)) and
                        not mother.on_colchicine
                )

                # Если была больна и без лечения, шанс попасть в список кандидатов ниже (например, 70% вместо 100%)
                if was_sick_and_untreated:
                    if random.random() < 0.7:  # Те самые 30% снижения фертильности
                        final_candidates.append((father, mother))
                else:
                    final_candidates.append((father, mother))

            if final_candidates:
                father, mother = random.choice(final_candidates)

                # Устанавливаем связи
                child.father_id = father.id
                child.mother_id = mother.id
                father.add_child(child.id)
                mother.add_child(child.id)

                birth_year_of_child = 1950 - child.age
                if birth_year_of_child > mother.last_birth_year:
                    mother.last_birth_year = birth_year_of_child

                family_key = f"{father.id}_{mother.id}"
                self.family_children_count[family_key] += 1









