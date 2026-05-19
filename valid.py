import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import random
import uuid
from tqdm import tqdm
from scipy import stats

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
        self.max_age_limit = max_age_limit
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

    def set_genotype(self, allele_1: str, allele_2: str):
        valid_values = ['N', 'M694V', "V726A", "M680I", "R761H"]
        if allele_1 not in valid_values or allele_2 not in valid_values:
            raise ValueError(f"Invalid allele values. Valid values: {valid_values}")

        self.mefv_allele_1 = allele_1
        self.mefv_allele_2 = allele_2
        self.update_genotype_status()
        self._determine_lifetime_risk()

    def update_genotype_status(self):
        alleles = [self.mefv_allele_1, self.mefv_allele_2]
        mutant_count = sum(1 for allele in alleles if allele != "N")

        if mutant_count == 0:
            self.genotype_status = "healthy"
            self.mutation_type = None
        elif mutant_count == 1:
            self.genotype_status = "carrier"
            self.mutation_type = "heterozygous"
        else:
            self.genotype_status = "affected"
            if len(set(alleles)) == 1:
                allele_type = alleles[0]
                if allele_type == "M694V":
                    self.mutation_type = "M694V_homozygous"
                else:
                    self.mutation_type = "other_homozygous"
            else:
                self.mutation_type = "compound_heterozygous"

    def _determine_lifetime_risk(self):
        """
        Определяет пожизненную вероятность заболевания (пенетрантность)
        и генерирует возраст начала болезни, если человек заболеет.
        Вызывается один раз при установке генотипа.
        """
        if self.genotype_status not in ['affected', 'carrier']:
            self.will_develop_symptoms = False
            self.age_of_onset = None
            return

        # 1. Устанавливаем пенетрантность по генотипу
        if self.mutation_type == "M694V_homozygous":
            self.penetrance = 0.95
        elif self.mutation_type in ["compound_heterozygous", "other_homozygous"]:
            self.penetrance = 0.75
        elif self.mutation_type == "heterozygous":
            self.penetrance = 0.03
        else:
            self.penetrance = 0.0

        # 2. Бросаем монетку: заболеет или нет
        self.will_develop_symptoms = random.random() < self.penetrance

        # 3. Если заболеет, генерируем возраст начала
        if self.will_develop_symptoms:
            self.age_of_onset = self._generate_age_of_onset()
            # Изначально клинический статус asymptomatic
            # Симптомы появятся, когда age >= age_of_onset
        else:
            self.age_of_onset = None
            self.clinical_status = 'never_symptomatic'  # Новый статус

    def _generate_age_of_onset(self):
        """
        Генерирует возраст начала болезни на основе клинических данных.
        """
        if self.mutation_type == "M694V_homozygous":
            age_groups = [
                (2, 10, 0.20),  # Снизили с 0.30 (меньше детей болеет сразу)
                (11, 20, 0.40),  # Снизили с 0.45
                (21, 30, 0.25),  # Увеличили (сдвиг в молодость)
                (31, 40, 0.10),  # Увеличили
                (41, 70, 0.05)
            ]

        elif self.mutation_type in ["compound_heterozygous", "other_homozygous"]:
            age_groups = [
                (2, 20, 0.35),  # Снизили с 0.50 (существенный сдвиг)
                (21, 30, 0.35),  # Увеличили
                (31, 40, 0.20),  # Увеличили
                (41, 70, 0.10)
            ]

        elif self.mutation_type == "heterozygous":
            # ИСПРАВЛЕНО: Еще более позднее начало
            # Гетерозиготы ОЧЕНЬ редко болеют в детстве
            age_groups = [
                (2, 20, 0.02),  # Почти исключаем детское начало (было 0.05)
                (21, 30, 0.15),  # Было 0.20
                (31, 40, 0.35),  # Было 0.40
                (41, 50, 0.38),  # Сдвигаем основной пик сюда (было 0.25)
                (51, 70, 0.10)
            ]
        else:
            return None

        # Нормализуем вероятности (на всякий случай)
        group_probs = [g[2] for g in age_groups]
        total = sum(group_probs)
        group_probs = [p / total for p in group_probs]

        selected_group = random.choices(age_groups, weights=group_probs, k=1)[0]
        min_age, max_age, _ = selected_group
        return random.randint(min_age, max_age)


#Пенетрантность — это пожизненная вероятность заболеть для данного генотипа
# Onset probability — это годовая вероятность начала болезни среди тех, кто в итоге заболеет

#Сначала определяем, попадает ли человек вообще в группу "кто заболеет когда-нибудь"
#Затем распределяем возраст начала болезни среди этих людей

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
                # Только 70% гетерозигот сразу получают диагноз
                # Остальные ходят с недиагностированными легкими симптомами
                if random.random() < 0.7:
                    self.clinical_status = 'symptomatic'
                    self._determine_disease_severity()
                # else: остаются asymptomatic, но с возрастом симптомы усилятся
            else:
                self.clinical_status = 'symptomatic'
                self._determine_disease_severity()

        if (self.clinical_status == 'symptomatic' and
            not self.on_colchicine):

            base_prob = 0.02 # Это математически отражает доступность 0.2

            if self.age <= 15:
                base_prob *= 2

            if self.disease_severity == 'severe':
                base_prob *= 3.0
            elif self.disease_severity == 'moderate':
                base_prob *= 2.0 # В сценарии с низкой доступностью лечение будут получать только самые тяжелые случаи
                # или те, кому очень повезло, что и приведет к ожидаемому результату —
                # «значительной доле тяжелых форм» (так как многие не будут купировать болезнь вовремя)

            base_prob = min(base_prob, 1.0)

            if random.random() < base_prob:
                self.on_colchicine = True

    def _determine_disease_severity(self):
        if self.mutation_type == "M694V_homozygous":
            severity_weights = {"mild": 0.10, "moderate": 0.20, "severe": 0.70}

        elif self.mutation_type in ["compound_heterozygous", "other_homozygous"]:
            severity_weights = {"mild": 0.30, "moderate": 0.40, "severe": 0.30}

        elif self.mutation_type == "heterozygous":
            severity_weights = {"mild": 0.95, "moderate": 0.04, "severe": 0.01}

        else:
            severity_weights = {"mild": 1.0, "moderate": 0.0, "severe": 0.0}

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

class ValidationSimulation:
    def __init__(self,
                 initial_population_size: int = 10000,
                 max_age_limit: int = 49,
                 mutation_frequencies: Dict[str, float] = None,
                 simulation_years: int = 150,
                 validation_year: int = 2012,
                 base_year: int = 1862):

        self.initial_population_size = initial_population_size
        self.max_age_limit = max_age_limit
        self.simulation_years = simulation_years
        self.validation_year = validation_year
        self.base_year = base_year


        self.mutation_frequencies = mutation_frequencies or {
            'N': 0.945,  # Возвращаемся к 0.945
            'M694V': 0.040,  # Золотая середина между 0.037 и 0.042
            'V726A': 0.007,
            'M680I': 0.006,
            'R761H': 0.002}

        self.agents: Dict[str, Agent] = {}
        self.year = 0
        self.population_history = []
        self.children_born = 0

        # Валидационные данные
        self.validation_point_reached = False
        self.validation_data = None

    def initialize_founders(self):
        #Инициализация основателей популяции
        for i in range(self.initial_population_size):
            age = random.choices(
                [18, 22, 25, 28, 30, 32, 35, 38, 40, 42],
                weights=[0.05, 0.10, 0.15, 0.15, 0.15, 0.10, 0.10, 0.08, 0.07, 0.05]
            )[0]

            gender = random.choice(['male', 'female'])

            agent = Agent(
                gender=gender, age=age, generation=0,
                max_age_limit=self.max_age_limit,
                father_id=None, mother_id=None
            )

            # 3. Устанавливаем генетику (вызовется метод с is_susceptible)
            allele1 = self._get_random_allele()
            allele2 = self._get_random_allele()
            agent.set_genotype(allele1, allele2)

            self.agents[agent.id] = agent

        self._form_initial_partnerships()

    def _get_random_allele(self) -> str:
        return random.choices(
            list(self.mutation_frequencies.keys()),
            weights=list(self.mutation_frequencies.values())
        )[0]

    def _form_initial_partnerships(self):
        # Образование пар
        males = [a for a in self.agents.values()
                 if a.gender == 'male' and a.partner_id is None and a.age >= 18]
        females = [a for a in self.agents.values()
                   if a.gender == 'female' and a.partner_id is None and a.age >= 18]

        males.sort(key=lambda x: x.age)
        females.sort(key=lambda x: x.age)

        females_iter = iter(females)
        females = []

        # TODO: подбор пар неидеальный, но довольно быстрый
        #       можно сделать лучше (например, поиском паросочетаний в двудольном графе)
        for male in males:
            if male.partner_id is not None:
                continue

            while (f := next(females_iter, None)) is not None and (
                    f.partner_id is not None or abs(f.age - male.age) > 10):
                if abs(f.age - male.age) > 10:
                    females.append(f)

            if f is None:
                females_iter = iter(females)
                females = []
            else:
                male.set_partner(f)

    def run_validation_simulation(self, target_year: int = 2012) -> dict:
        self.initialize_founders()

        for year in range(1, self.simulation_years + 1):
            self.year = year
            self._run_single_year()

            current_calendar_year = self.base_year + year

            # Сбор данных для валидации
            if current_calendar_year == target_year:
                self.validation_point_reached = True
                prevalence = self.calculate_simulated_prevalence()
                population = len(self.agents)
                asymptomatic_carriers = sum(1 for a in self.agents.values()
                                            if a.alive and
                                            a.will_develop_symptoms and
                                            a.clinical_status == 'asymptomatic')

                mean_onset_age = np.mean([a.age_of_onset for a in self.agents.values()
                                          if a.alive and a.age_of_onset is not None])

                self.validation_data = {
                    'year': current_calendar_year,
                    'prevalence': prevalence,
                    'population': population,
                    'affected_count': len([a for a in self.agents.values()
                                           if a.alive and a.genotype_status == 'affected']),
                    'symptomatic_count': len([a for a in self.agents.values()
                                              if a.alive and a.clinical_status == 'symptomatic']),
                    'asymptomatic_carriers': asymptomatic_carriers,
                    'mean_onset_age': mean_onset_age,
                    'penetrance_by_genotype': {
                        'M694V_homozygous': 0.95,
                        'compound_heterozygous': 0.75,
                        'heterozygous': 0.04
                    }
                }

        results = {
            'validation_year': target_year,
            'final_population': len(self.agents),
            'final_prevalence': self.calculate_simulated_prevalence(),
            'validation_point_reached': self.validation_point_reached,
            'validation_data': self.validation_data,
            'children_born': self.children_born
        }

        return results

    def _run_single_year(self):

        # Старение всех агентов
        for (agent_id, agent) in list(self.agents.items()):
            if agent.alive:
                agent.age_year()
            if not agent.alive:
                del self.agents[agent_id]

        # Образование новых пар
        self._form_new_partnerships()

        # Рождение детей
        self._birth_process()

    def _form_new_partnerships(self):
        # Образование новых пар
        single_males = [a for a in self.agents.values()
                        if a.alive and a.gender == 'male' and a.partner_id is None
                        and 18 <= a.age <= 45]
        single_females = [a for a in self.agents.values()
                          if a.alive and a.gender == 'female' and a.partner_id is None
                          and 18 <= a.age <= 40]

        random.shuffle(single_males)
        random.shuffle(single_females)

        min_pairs = min(len(single_males), len(single_females))
        for i in range(min_pairs):
            if (single_males[i].alive and single_females[i].alive and
                    abs(single_males[i].age - single_females[i].age) <= 15):

                partnership_prob = 0.3
                if random.random() < partnership_prob:
                    single_males[i].set_partner(single_females[i])

    def _birth_process(self):
        # Процесс рождения детей
        potential_parents = [a for a in self.agents.values()
                             if a.alive and a.can_have_children() and
                             a.partner_id in self.agents]

        for parent in potential_parents:
            partner = self.agents[parent.partner_id]

            if not partner.can_have_children():
                continue

            # Вероятность рождения
            avg_age = (parent.age + partner.age) / 2
            if avg_age < 20:
                birth_prob = 0.07
            elif 20 <= avg_age < 25:
                birth_prob = 0.09
            elif 25 <= avg_age < 30:
                birth_prob = 0.10
            elif 30 <= avg_age < 35:
                birth_prob = 0.08
            elif 35 <= avg_age < 40:
                birth_prob = 0.05
            else:
                birth_prob = 0.02

            if random.random() < birth_prob:
                self._create_child(parent, partner)
                self.children_born += 1

    def _create_child(self, parent1: Agent, parent2: Agent):
        # Создание ребенка
        if parent1.gender == 'male':
            father, mother = parent1, parent2
        else:
            father, mother = parent2, parent1

        # Наследование аллелей
        father_allele = random.choice([father.mefv_allele_1, father.mefv_allele_2])
        mother_allele = random.choice([mother.mefv_allele_1, mother.mefv_allele_2])

        child_gender = random.choice(['male', 'female'])
        child_generation = max(father.generation, mother.generation) + 1

        child = Agent(
            gender=child_gender, age=0, generation=child_generation,
            max_age_limit=self.max_age_limit,
            father_id=father.id, mother_id=mother.id
        )

        child.set_genotype(father_allele, mother_allele)
        self.agents[child.id] = child

    def calculate_simulated_prevalence(self) -> float:
        # Расчет распространенности FMF
        alive_agents = self.agents.values()

        if not alive_agents:
            return 0.0

        # Считаем симптоматических пациентов
        symptomatic_count = sum(1 for a in alive_agents
                                if a.clinical_status == 'symptomatic')

        total_alive = len(alive_agents)

        return symptomatic_count / total_alive if total_alive > 0 else 0.0


def load_real_data(file_path: str = 'FMF_data2.csv') -> pd.DataFrame:
    # Загрузка и подготовка данных о FMF в Армении
    data = pd.read_csv(file_path)
    data['prevalence_0_49'] = data['Registered_0-49'] / data['0-49 population_Total']

    target_columns = [
        'years',
        'Registered_0-49',
        '0-49 population_Total',
        'prevalence_0_49'
    ]

    return data[target_columns].copy()


def validate_age_distributions(simulation):
    """Проверяет, что распределения возраста начала соответствуют задуманным"""
    agents = [a for a in simulation.agents.values()
              if a.will_develop_symptoms and a.age_of_onset is not None]

    print(f"\n=== ВАЛИДАЦИЯ РАСПРЕДЕЛЕНИЙ ===")
    print(f"Размер популяции: {len(simulation.agents)}")
    print(f"Всего в группе риска: {len(agents)}")
    print(f"Из них гомозигот: {sum(1 for a in agents if a.genotype_status == 'affected')}")
    print(f"Из них гетерозигот: {sum(1 for a in agents if a.genotype_status == 'carrier')}")

    symptomatic_ages = [a.age_of_onset for a in simulation.agents.values()
                       if a.clinical_status == 'symptomatic' and a.age_of_onset]

    print(f"\n=== ВАЛИДАЦИЯ РАСПРЕДЕЛЕНИЙ ===")
    print(f"Всего в группе риска: {len(agents)}")
    print(f"Уже заболели: {len(symptomatic_ages)}")

    by_genotype = {
        'M694V_homozygous': [],
        'compound_heterozygous': [],
        'heterozygous': []
    }

    for a in agents:
        if a.mutation_type in by_genotype:
            by_genotype[a.mutation_type].append(a.age_of_onset)

    for gt, ages in by_genotype.items():
        if ages:
            print(f"\n{gt}:")
            print(f"  n={len(ages)}")
            print(f"  Средний возраст: {np.mean(ages):.1f}")
            print(f"  Медиана: {np.median(ages):.1f}")
            print(f"  До 20 лет: {sum(1 for a in ages if a <= 20)/len(ages):.1%}")
            print(f"  20-30 лет: {sum(1 for a in ages if 20 < a <= 30)/len(ages):.1%}")
            print(f"  30-40 лет: {sum(1 for a in ages if 30 < a <= 40)/len(ages):.1%}")
            print(f"  После 40: {sum(1 for a in ages if a > 40)/len(ages):.1%}")

def run_validation_with_real_data(n_simulations: int = 20, target_year: int = 2012):
    # Запуск валидации модели с реальными данными
    print("=" * 80)
    print("ВАЛИДАЦИЯ МОДЕЛИ FMF С РЕАЛЬНЫМИ ДАННЫМИ")
    print("=" * 80)

    # ШАГ 1: Загрузка реальных данных
    print("\n1. ЗАГРУЗКА РЕАЛЬНЫХ ДАННЫХ")
    print("-" * 40)

    try:
        real_data = load_real_data('FMF_data2.csv')
        print(f"Загружены данные за {len(real_data)} лет")

        target_data = real_data[real_data['years'] == target_year]
        if len(target_data) == 0:
            print(f"Внимание: нет данных за {target_year} год")
            target_prevalence = real_data['prevalence_0_49'].mean()
        else:
            target_prevalence = target_data['prevalence_0_49'].iloc[0]

        print(f"\nЦелевая распространенность FMF (0-49 лет):")
        print(f"  Год: {target_year}")
        print(f"  Значение: {target_prevalence:.6f} (1:{int(1 / target_prevalence)} человек)")

    except FileNotFoundError:
        print("Ошибка: файл 'FMF_data2.csv' не найден!")
        target_prevalence = 0.001753
        print(f"Использую тестовое значение: {target_prevalence}")
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return

    #  Запуск симуляций
    print(f"\n2. ЗАПУСК {n_simulations} СИМУЛЯЦИЙ")
    print("-" * 40)

    all_results = []

    # Параметры для симуляции
    mutation_frequencies = {
        'N': 0.945,  # Возвращаемся к 0.945
        'M694V': 0.040,  # Золотая середина между 0.037 и 0.042
        'V726A': 0.007,
        'M680I': 0.006,
        'R761H': 0.002
    }

    # Упрощенный прогресс-бар без вложенных сообщений
    for sim_idx in tqdm(range(n_simulations), desc="Симуляции", leave=True):
        sim = ValidationSimulation(
            initial_population_size=10_000,
            max_age_limit=49,
            simulation_years=target_year - 1862,
            validation_year=target_year,
            base_year=1862,
            mutation_frequencies=mutation_frequencies  # Используем скорректированные частоты
        )

        results = sim.run_validation_simulation(target_year=target_year)
        all_results.append(results)

    # Анализ результатов
    print(f"\n3. АНАЛИЗ РЕЗУЛЬТАТОВ ({n_simulations} симуляций)")
    print("-" * 40)

    successful_sims = [r for r in all_results if r['validation_point_reached']]
    prevalences = [r['final_prevalence'] for r in successful_sims]

    if not prevalences:
        print("Нет успешных симуляций!")
        return

    print(f"Успешных симуляций: {len(successful_sims)}/{n_simulations}")
    print(f"Реальная распространенность: {target_prevalence:.6f}")
    print(f"Средняя смоделированная: {np.mean(prevalences):.6f}")
    print(f"Стандартное отклонение: {np.std(prevalences):.6f}")
    print(f"Диапазон: [{np.min(prevalences):.6f}, {np.max(prevalences):.6f}]")

    # Дополнительная статистика
    populations = [r['final_population'] for r in successful_sims]
    print(f"Средний размер популяции: {np.mean(populations):.0f}")

    # Статистические тесты
    print(f"\n4. СТАТИСТИЧЕСКИЕ ТЕСТЫ")
    print("-" * 40)

    t_stat, p_value = stats.ttest_1samp(prevalences, target_prevalence)
    print(f"t-тест для одной выборки:")
    print(f"  t-статистика: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")

    # Доверительный интервал
    ci_lower, ci_upper = stats.t.interval(
        confidence=0.95,
        df=len(prevalences) - 1,
        loc=np.mean(prevalences),
        scale=stats.sem(prevalences)
    )
    print(f"\n95% доверительный интервал для модели:")
    print(f"  [{ci_lower:.6f}, {ci_upper:.6f}]")

    within_ci = ci_lower <= target_prevalence <= ci_upper
    print(f"  Реальное значение в ДИ: {'ДА' if within_ci else 'НЕТ'}")

    # Оценка точности
    errors = [abs(p - target_prevalence) for p in prevalences]
    mean_error = np.mean(errors)
    relative_errors = [abs(p - target_prevalence) / target_prevalence for p in prevalences]
    mean_relative_error = np.mean(relative_errors) * 100

    print(f"\n5. ОЦЕНКА ТОЧНОСТИ МОДЕЛИ")
    print("-" * 40)
    print(f"Средняя абсолютная ошибка: {mean_error:.6f}")
    print(f"Средняя относительная ошибка: {mean_relative_error:.1f}%")

    # Критерии валидации
    validation_passed = (
            within_ci and
            mean_relative_error < 50 and
            p_value > 0.05
    )

    print(f"\n6. РЕЗУЛЬТАТ ВАЛИДАЦИИ")
    print("=" * 40)

    if validation_passed:
        print("✅ ВАЛИДАЦИЯ ПРОЙДЕНА")
        print("Модель адекватно воспроизводит реальную распространенность FMF!")
    else:
        print("❌ ВАЛИДАЦИЯ НЕ ПРОЙДЕНА")
        if not within_ci:
            print("  - Реальная распространенность вне доверительного интервала")
        if mean_relative_error >= 50:
            print(f"  - Ошибка слишком велика ({mean_relative_error:.1f}%)")
        if p_value <= 0.05:
            print(f"  - Статистически значимое отличие (p={p_value:.4f})")

    # Визуализация
    print(f"\n7. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
    print("-" * 40)

    plot_validation_results(all_results, target_prevalence, target_year)

    return {
        'target_prevalence': target_prevalence,
        'simulation_results': all_results,
        'statistics': {
            'mean_simulated': np.mean(prevalences),
            'std_simulated': np.std(prevalences),
            'mean_error': mean_error,
            'mean_relative_error': mean_relative_error,
            't_test': {'t_stat': t_stat, 'p_value': p_value},
            'confidence_interval': [ci_lower, ci_upper],
            'within_ci': within_ci,
            'validation_passed': validation_passed
        }
    }


def analyze_genotype_distribution(simulation):
    """Полный анализ распределения генотипов в популяции"""

    agents = list(simulation.agents.values())
    total = len(agents)

    if total == 0:
        print("Популяция пуста")
        return {
            'total_population': 0,
            'status_counts': {},
            'mutation_counts': {},
            'allele_frequencies': {}
        }

    print("\n" + "=" * 80)
    print("ГЕНЕТИЧЕСКИЙ АУДИТ ПОПУЛЯЦИИ")
    print("=" * 80)

    # 1. Базовые категории
    print("\n1. СТАТУС НОСИТЕЛЬСТВА:")
    print("-" * 40)

    status_counts = {
        'healthy': 0,
        'carrier': 0,
        'affected': 0
    }

    for agent in agents:
        if agent.genotype_status in status_counts:
            status_counts[agent.genotype_status] += 1

    for status, count in status_counts.items():
        percentage = count / total * 100
        print(f"  {status:15}: {count:6d} ({percentage:5.2f}%)")

    # 2. Детальные мутации
    print("\n2. ТИПЫ МУТАЦИЙ (АФФЕКТИРОВАННЫЕ И НОСИТЕЛИ):")
    print("-" * 40)

    mutation_types = {
        'M694V_homozygous': 0,
        'other_homozygous': 0,
        'compound_heterozygous': 0,
        'heterozygous': 0
    }

    for agent in agents:
        if agent.mutation_type in mutation_types:
            mutation_types[agent.mutation_type] += 1

    for mut_type, count in mutation_types.items():
        if count > 0:
            percentage = count / total * 100
            print(f"  {mut_type:25}: {count:6d} ({percentage:5.2f}%)")

    # 3. Распределение по генотипам и клиническому статусу
    print("\n3. КЛИНИЧЕСКИЙ СТАТУС ПО ГЕНОТИПАМ:")
    print("-" * 40)

    clinical_by_genotype = defaultdict(lambda: defaultdict(int))

    for agent in agents:
        if agent.genotype_status in ['affected', 'carrier']:
            clinical_by_genotype[agent.mutation_type][agent.clinical_status] += 1

    for mut_type in mutation_types.keys():
        if mut_type in clinical_by_genotype:
            print(f"\n  {mut_type}:")
            total_mut = sum(clinical_by_genotype[mut_type].values())
            print(f"    Всего: {total_mut}")

            for status in ['asymptomatic', 'symptomatic', 'never_symptomatic']:
                count = clinical_by_genotype[mut_type].get(status, 0)
                if count > 0:
                    pct = count / total_mut * 100
                    print(f"      {status:20}: {count:4d} ({pct:5.1f}%)")

    # 4. Аллельные частоты
    print("\n4. АЛЛЕЛЬНЫЕ ЧАСТОТЫ:")
    print("-" * 40)

    allele_counts = defaultdict(int)
    total_alleles = 0

    for agent in agents:
        if agent.mefv_allele_1:
            allele_counts[agent.mefv_allele_1] += 1
            total_alleles += 1
        if agent.mefv_allele_2:
            allele_counts[agent.mefv_allele_2] += 1
            total_alleles += 1

    for allele in ['N', 'M694V', 'M680I', 'V726A', 'R761H']:
        count = allele_counts.get(allele, 0)
        frequency = count / total_alleles if total_alleles > 0 else 0
        expected = simulation.mutation_frequencies.get(allele, 0)
        print(f"  {allele:8}: {count:6d} аллелей  freq={frequency:.4f}  (ожидалось: {expected:.4f})")

    return {
        'total_population': total,
        'status_counts': status_counts,
        'mutation_counts': mutation_types,
        'allele_frequencies': {k: allele_counts.get(k, 0) / total_alleles for k in allele_counts.keys()}
    }


def hardy_weinberg_test(simulation):
    """Тест Харди-Вайнберга для проверки равновесия"""

    agents = list(simulation.agents.values())
    total = len(agents)

    if total == 0:
        print("Популяция пуста")
        return {'p_value': 1, 'chi2': 0, 'hw_equilibrium': True}

    # Считаем аллельные частоты
    allele_counts = defaultdict(int)
    total_alleles = 0

    for agent in agents:
        allele_counts[agent.mefv_allele_1] += 1
        allele_counts[agent.mefv_allele_2] += 1
        total_alleles += 2

    # Частота нормального аллеля (p) и мутантных (q)
    p = allele_counts.get('N', 0) / total_alleles
    q = 1 - p

    # Ожидаемые генотипы по Харди-Вайнбергу
    expected_homozygous_normal = p * p * total
    expected_heterozygous = 2 * p * q * total
    expected_homozygous_mutant = q * q * total

    # Наблюдаемые
    observed_homozygous_normal = sum(1 for a in agents
                                     if a.mefv_allele_1 == 'N' and a.mefv_allele_2 == 'N')
    observed_heterozygous = sum(1 for a in agents
                                if (a.mefv_allele_1 == 'N') != (a.mefv_allele_2 == 'N'))
    observed_homozygous_mutant = sum(1 for a in agents
                                     if a.mefv_allele_1 != 'N' and a.mefv_allele_2 != 'N')

    print("\n" + "=" * 80)
    print("ТЕСТ ХАРДИ-ВАЙНБЕРГА")
    print("=" * 80)
    print(f"\nАллельные частоты:")
    print(f"  Нормальный аллель (p): {p:.4f}")
    print(f"  Мутантные аллели (q): {q:.4f}")

    print(f"\n{'Генотип':<25} {'Наблюдаемые':<15} {'Ожидаемые':<15} {'Разница':<10}")
    print("-" * 65)
    print(f"{'Гомозиготы нормальные':<25} {observed_homozygous_normal:<15} {expected_homozygous_normal:<15.1f} "
          f"{observed_homozygous_normal - expected_homozygous_normal:<+10.1f}")
    print(f"{'Гетерозиготы':<25} {observed_heterozygous:<15} {expected_heterozygous:<15.1f} "
          f"{observed_heterozygous - expected_heterozygous:<+10.1f}")
    print(f"{'Гомозиготы мутантные':<25} {observed_homozygous_mutant:<15} {expected_homozygous_mutant:<15.1f} "
          f"{observed_homozygous_mutant - expected_homozygous_mutant:<+10.1f}")

    chi2_stat = 0
    if expected_homozygous_normal > 0:
        chi2_stat += (observed_homozygous_normal - expected_homozygous_normal) ** 2 / expected_homozygous_normal
    if expected_heterozygous > 0:
        chi2_stat += (observed_heterozygous - expected_heterozygous) ** 2 / expected_heterozygous
    if expected_homozygous_mutant > 0:
        chi2_stat += (observed_homozygous_mutant - expected_homozygous_mutant) ** 2 / expected_homozygous_mutant

    from scipy.stats import chi2 as chi2_dist
    p_value = 1 - chi2_dist.cdf(chi2_stat, df=1)

    print(f"\nХи-квадрат: {chi2_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Равновесие Харди-Вайнберга: {'ДА' if p_value > 0.05 else 'НЕТ'}")

    return {
        'p_value': p_value,
        'chi2': chi2_stat,
        'hw_equilibrium': p_value > 0.05
    }


def detailed_mutation_report(simulation):
    """Подробный отчет по каждой мутации"""

    agents = list(simulation.agents.values())

    print("\n" + "=" * 80)
    print("ДЕТАЛЬНЫЙ ОТЧЕТ ПО МУТАЦИЯМ")
    print("=" * 80)

    # Группируем по типу мутации и статусу
    mutation_details = defaultdict(lambda: {
        'total': 0,
        'symptomatic': 0,
        'asymptomatic': 0,
        'never_symptomatic': 0,
        'on_colchicine': 0,
        'severity': {'mild': 0, 'moderate': 0, 'severe': 0},
        'mean_age': [],
        'mean_onset_age': []
    })

    for agent in agents:
        if agent.mutation_type:
            mt = agent.mutation_type
            mutation_details[mt]['total'] += 1

            if agent.clinical_status == 'symptomatic':
                mutation_details[mt]['symptomatic'] += 1
            elif agent.clinical_status == 'never_symptomatic':
                mutation_details[mt]['never_symptomatic'] += 1
            else:
                mutation_details[mt]['asymptomatic'] += 1

            if agent.on_colchicine:
                mutation_details[mt]['on_colchicine'] += 1

            if agent.disease_severity:
                mutation_details[mt]['severity'][agent.disease_severity] += 1

            if agent.age_of_onset:
                mutation_details[mt]['mean_onset_age'].append(agent.age_of_onset)

            mutation_details[mt]['mean_age'].append(agent.age)

    # Выводим отчет
    for mutation_type, data in mutation_details.items():
        print(f"\n{'=' * 50}")
        print(f"МУТАЦИЯ: {mutation_type}")
        print(f"{'=' * 50}")

        print(f"\n ОБЩАЯ СТАТИСТИКА:")
        print(f"  Всего носителей: {data['total']}")

        if data['total'] > 0:
            symptomatic_pct = data['symptomatic'] / data['total'] * 100
            print(f"  Симптоматические: {data['symptomatic']} ({symptomatic_pct:.1f}%)")
            print(f"  Бессимптомные (будут): {data['asymptomatic']}")
            print(f"  Никогда не заболеют: {data['never_symptomatic']}")
            print(f"  На колхицине: {data['on_colchicine']}")

        if data['mean_age']:
            print(f"\n ВОЗРАСТ:")
            print(f"  Средний текущий возраст: {np.mean(data['mean_age']):.1f} лет")
            print(f"  Диапазон: [{min(data['mean_age'])}, {max(data['mean_age'])}]")

        if data['mean_onset_age']:
            print(f"\n ВОЗРАСТ НАЧАЛА:")
            print(f"  Средний возраст начала: {np.mean(data['mean_onset_age']):.1f} лет")
            print(f"  Медиана: {np.median(data['mean_onset_age']):.1f} лет")
            print(f"  Диапазон: [{min(data['mean_onset_age'])}, {max(data['mean_onset_age'])}]")

        if sum(data['severity'].values()) > 0:
            print(f"\n ТЯЖЕСТЬ ЗАБОЛЕВАНИЯ:")
            total_severe = sum(data['severity'].values())
            for severity, count in data['severity'].items():
                pct = count / total_severe * 100
                print(f"  {severity:10}: {count:3d} ({pct:5.1f}%)")

    return mutation_details


def plot_population_audit(simulation, genotype_stats, penetrance_data):
    """Создание визуализаций для полного аудита популяции"""

    agents = list(simulation.agents.values())

    # Создаем фигуру с 4 подграфиками
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Полный аудит популяции FMF - Детальный анализ',
                 fontsize=16, fontweight='bold')

    # 1. Распределение генотипов
    ax1 = plt.subplot(3, 3, 1)
    status_counts = genotype_stats.get('status_counts', {})
    labels = list(status_counts.keys())
    sizes = list(status_counts.values())
    colors = ['#2ecc71', '#f39c12', '#e74c3c']

    if sizes and sum(sizes) > 0:
        wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                                           colors=colors, startangle=90)
    ax1.set_title('Распределение по генотипам', fontsize=12, fontweight='bold')

    # 2. Типы мутаций
    ax2 = plt.subplot(3, 3, 2)
    mutation_counts = genotype_stats.get('mutation_counts', {})
    mut_labels = list(mutation_counts.keys())
    mut_sizes = list(mutation_counts.values())

    # Упрощаем названия для графика
    short_labels = ['M694V\nгомозиготы', 'Другие\nгомозиготы',
                    'Комплексные\nгетерозиготы', 'Гетерозиготы']

    if mut_sizes:
        bars = ax2.barh(short_labels[:len(mut_sizes)], mut_sizes,
                        color=['#c0392b', '#e67e22', '#f1c40f', '#3498db'][:len(mut_sizes)])
        ax2.set_xlabel('Количество')
        ax2.set_title('Распределение типов мутаций', fontsize=12, fontweight='bold')

        # Добавляем значения на столбцы
        for i, (bar, val) in enumerate(zip(bars, mut_sizes)):
            ax2.text(val, bar.get_y() + bar.get_height() / 2, f'{val}',
                     ha='left', va='center', fontweight='bold')

    # 3. Аллельные частоты
    ax3 = plt.subplot(3, 3, 3)
    allele_freqs = genotype_stats.get('allele_frequencies', {})
    alleles = ['N', 'M694V', 'M680I', 'V726A', 'R761H']
    observed = [allele_freqs.get(a, 0) for a in alleles]
    expected = [simulation.mutation_frequencies.get(a, 0) for a in alleles]

    x = np.arange(len(alleles))
    width = 0.35

    ax3.bar(x - width / 2, observed, width, label='Наблюдаемые', color='#3498db')
    ax3.bar(x + width / 2, expected, width, label='Ожидаемые', color='#e74c3c', alpha=0.7)
    ax3.set_xlabel('Аллели')
    ax3.set_ylabel('Частота')
    ax3.set_title('Аллельные частоты: наблюдение vs ожидание', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(alleles)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Возраст начала заболевания по генотипам
    ax4 = plt.subplot(3, 3, 4)

    onset_ages_by_genotype = {
        'M694V_homozygous': [],
        'compound_heterozygous': [],
        'heterozygous': []
    }

    for agent in agents:
        if agent.age_of_onset is not None and agent.mutation_type in onset_ages_by_genotype:
            onset_ages_by_genotype[agent.mutation_type].append(agent.age_of_onset)

    # Создаем box plot
    box_data = [onset_ages_by_genotype[gt] for gt in onset_ages_by_genotype.keys()]
    box_labels = ['M694V\nгомозиготы', 'Комплексные\nгетерозиготы', 'Гетерозиготы']

    if any(len(data) > 0 for data in box_data):
        bp = ax4.boxplot(box_data, patch_artist=True)
        ax4.set_xticklabels(box_labels)

        # Раскрашиваем box plots
        colors_box = ['#c0392b', '#f39c12', '#3498db']
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    ax4.set_ylabel('Возраст начала (лет)')
    ax4.set_title('Распределение возраста начала по генотипам', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # 5. Пенетрантность
    ax5 = plt.subplot(3, 3, 5)

    mut_types = list(penetrance_data.keys())
    if mut_types:
        expected_pen = [penetrance_data[m]['expected'] for m in mut_types]
        actual_pen = [penetrance_data[m]['actual'] for m in mut_types]

        x = np.arange(len(mut_types))
        width = 0.35

        bars1 = ax5.bar(x - width / 2, expected_pen, width, label='Ожидаемая', color='#95a5a6')
        bars2 = ax5.bar(x + width / 2, actual_pen, width, label='Фактическая', color='#27ae60')

        ax5.set_xlabel('Тип мутации')
        ax5.set_ylabel('Пенетрантность')
        ax5.set_title('Пенетрантность: ожидаемая vs фактическая', fontsize=12, fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(['M694V\nгомозиготы', 'Комплексные\nгетерозиготы', 'Гетерозиготы'][:len(mut_types)])
        ax5.legend()
        ax5.set_ylim(0, 1)
        ax5.grid(True, alpha=0.3)

        # Добавляем значения на столбцы
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{height:.0%}', ha='center', va='bottom', fontweight='bold')

    # 6. Тяжесть заболевания по генотипам
    ax6 = plt.subplot(3, 3, 6)

    severity_data = defaultdict(lambda: defaultdict(int))
    for agent in agents:
        if agent.mutation_type and agent.disease_severity:
            severity_data[agent.mutation_type][agent.disease_severity] += 1

    # Подготавливаем данные для stacked bar chart
    mut_labels_short = ['M694V\nгомозиготы', 'Комплексные\nгетерозиготы', 'Гетерозиготы']
    severity_levels = ['mild', 'moderate', 'severe']
    severity_colors = {'mild': '#2ecc71', 'moderate': '#f39c12', 'severe': '#e74c3c'}

    bottom = np.zeros(len(mut_labels_short))

    for severity in severity_levels:
        values = []
        for i, mut in enumerate(['M694V_homozygous', 'compound_heterozygous', 'heterozygous']):
            values.append(severity_data[mut].get(severity, 0))

        if any(values):
            bars = ax6.bar(mut_labels_short, values, bottom=bottom,
                           label=severity.capitalize(), color=severity_colors[severity])
            bottom += values

    ax6.set_ylabel('Количество')
    ax6.set_title('Распределение тяжести заболевания', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')

    # 7. Возрастное распределение по клиническому статусу
    ax7 = plt.subplot(3, 3, 7)

    age_by_status = {
        'symptomatic': [],
        'asymptomatic': [],
        'never_symptomatic': []
    }

    for agent in agents:
        if agent.genotype_status in ['affected', 'carrier']:
            if agent.clinical_status in age_by_status:
                age_by_status[agent.clinical_status].append(agent.age)

    status_labels = ['Симптоматические', 'Бессимптомные\n(будут)', 'Никогда\nне заболеют']
    status_colors = ['#e74c3c', '#f39c12', '#3498db']

    box_data_status = [age_by_status['symptomatic'],
                       age_by_status['asymptomatic'],
                       age_by_status['never_symptomatic']]

    if any(len(data) > 0 for data in box_data_status):
        bp2 = ax7.boxplot(box_data_status, patch_artist=True)
        ax7.set_xticklabels(status_labels)

        for patch, color in zip(bp2['boxes'], status_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    ax7.set_ylabel('Возраст (лет)')
    ax7.set_title('Возрастное распределение по клиническому статусу', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)

    # 8. Пирамида возрастов
    ax8 = plt.subplot(3, 3, 8)

    ages_male = [a.age for a in agents if a.gender == 'male']
    ages_female = [a.age for a in agents if a.gender == 'female']

    if ages_male or ages_female:
        bins = np.arange(0, 51, 5)
        hist_male, _ = np.histogram(ages_male, bins=bins)
        hist_female, _ = np.histogram(ages_female, bins=bins)

        # Нормализуем для популяционной пирамиды
        hist_male_norm = -hist_male / len(agents) * 100
        hist_female_norm = hist_female / len(agents) * 100

        y_pos = bins[:-1] + 2.5

        ax8.barh(y_pos, hist_male_norm, height=4, label='Мужчины', color='#3498db')
        ax8.barh(y_pos, hist_female_norm, height=4, label='Женщины', color='#e74c3c')

    ax8.set_xlabel('Процент от популяции')
    ax8.set_ylabel('Возраст')
    ax8.set_title('Возрастно-половая пирамида', fontsize=12, fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # 9. Статистика по колхицину
    ax9 = plt.subplot(3, 3, 9)

    colchicine_data = defaultdict(lambda: {'on': 0, 'total': 0})

    for agent in agents:
        if agent.mutation_type and agent.clinical_status == 'symptomatic':
            colchicine_data[agent.mutation_type]['total'] += 1
            if agent.on_colchicine:
                colchicine_data[agent.mutation_type]['on'] += 1

    mut_types = ['M694V_homozygous', 'compound_heterozygous', 'heterozygous']
    treatment_rates = []
    mut_labels_treat = []

    for mut in mut_types:
        total = colchicine_data[mut]['total']
        on = colchicine_data[mut]['on']
        if total > 0:
            treatment_rates.append(on / total)
            mut_labels_treat.append(f'{mut.split("_")[0]}\n({on}/{total})')
        else:
            treatment_rates.append(0)
            mut_labels_treat.append(f'{mut.split("_")[0]}\n(0/0)')

    if any(rate > 0 for rate in treatment_rates):
        bars = ax9.bar(mut_labels_treat, treatment_rates, color=['#c0392b', '#e67e22', '#f1c40f'])
        ax9.set_ylabel('Доля на колхицине')
        ax9.set_title('Охват лечением среди симптоматических', fontsize=12, fontweight='bold')
        ax9.set_ylim(0, 1)
        ax9.grid(True, alpha=0.3, axis='y')

        # Добавляем значения
        for bar, rate in zip(bars, treatment_rates):
            height = bar.get_height()
            ax9.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{rate:.0%}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()


def plot_penetrance_comparison(simulation_results):
    """Сравнение пенетрантности в разных симуляциях"""

    fig, ax = plt.subplots(figsize=(12, 6))

    # Собираем данные по всем симуляциям
    penetrance_data = defaultdict(list)

    for sim_result in simulation_results:
        if 'validation_data' in sim_result and sim_result['validation_data']:
            vd = sim_result['validation_data']
            if 'penetrance_by_genotype' in vd:
                for gt, value in vd['penetrance_by_genotype'].items():
                    penetrance_data[gt].append(value)

    # Создаем box plot
    gt_labels = list(penetrance_data.keys())
    gt_short = ['M694V\nгомозиготы', 'Комплексные\nгетерозиготы', 'Гетерозиготы']

    if gt_labels:
        bp = ax.boxplot([penetrance_data[gt] for gt in gt_labels], patch_artist=True)
        ax.set_xticklabels(gt_short[:len(gt_labels)])

        colors = ['#c0392b', '#f39c12', '#3498db']
        for patch, color in zip(bp['boxes'], colors[:len(gt_labels)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Добавляем ожидаемые значения
        expected = [0.95, 0.75, 0.04]
        for i, exp in enumerate(expected[:len(gt_labels)]):
            ax.scatter(i + 1, exp, color='black', s=100, marker='*',
                       label='Ожидаемое' if i == 0 else '', zorder=5)

    ax.set_ylabel('Пенетрантность')
    ax.set_title('Вариабельность пенетрантности в разных симуляциях',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()


def full_population_audit(simulation, plot=True):
    """Полный аудит популяции: генетика + клиника + визуализация"""

    print("=" * 80)
    print(" ПОЛНЫЙ АУДИТ ПОПУЛЯЦИИ FMF")
    print("=" * 80)

    # 1. Базовый генетический анализ
    genotype_stats = analyze_genotype_distribution(simulation)

    # 2. Тест Харди-Вайнберга
    hardy_weinberg_test(simulation)

    # 3. Детальный отчет по мутациям
    detailed_mutation_report(simulation)

    # 4. Проверка пенетрантности
    print("\n" + "=" * 80)
    print("ПРОВЕРКА ПЕНЕТРАНТНОСТИ")
    print("=" * 80)

    penetrance_data = {}
    for mutation in ['M694V_homozygous', 'compound_heterozygous', 'heterozygous']:
        agents_with_mutation = [a for a in simulation.agents.values()
                                if a.mutation_type == mutation]

        if agents_with_mutation:
            total = len(agents_with_mutation)
            will_develop = sum(1 for a in agents_with_mutation if a.will_develop_symptoms)
            developed = sum(1 for a in agents_with_mutation
                            if a.clinical_status == 'symptomatic')

            expected_penetrance = {
                'M694V_homozygous': 0.95,
                'compound_heterozygous': 0.75,
                'heterozygous': 0.04
            }.get(mutation, 0)

            penetrance_data[mutation] = {
                'total': total,
                'will_develop': will_develop,
                'developed': developed,
                'expected': expected_penetrance,
                'actual': will_develop / total if total > 0 else 0
            }

            print(f"\n{mutation}:")
            print(f"  Всего: {total}")
            print(f"  В группе риска: {will_develop} ({will_develop / total * 100:.1f}%)")
            print(f"  Уже заболели: {developed}")
            print(f"  Ожидаемая пенетрантность: {expected_penetrance:.0%}")
            print(f"  Фактическая пенетрантность: {will_develop / total:.1%}")

            reached_onset_age = sum(1 for a in agents_with_mutation
                                    if a.will_develop_symptoms and
                                    a.age_of_onset is not None and
                                    a.age >= a.age_of_onset)
            print(f"  Достигли возраста начала: {reached_onset_age}")

    # 5. ВИЗУАЛИЗАЦИЯ
    if plot:
        plot_population_audit(simulation, genotype_stats, penetrance_data)

    return genotype_stats


def plot_validation_results(all_results: List[Dict], target_prevalence: float, target_year: int):
    # Визуализация результатов валидации"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Валидация модели FMF: сравнение с реальными данными ({target_year} год)',
                 fontsize=14, fontweight='bold')

    # 1. Распределение смоделированных распространенностей
    ax1 = axes[0, 0]
    prevalences = [r['final_prevalence'] for r in all_results if 'final_prevalence' in r]

    if prevalences:
        ax1.hist(prevalences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(x=target_prevalence, color='red', linestyle='--',
                    linewidth=2, label=f'Реальные данные: {target_prevalence:.6f}')
        ax1.axvline(x=np.mean(prevalences), color='green', linestyle=':',
                    linewidth=2, label=f'Модель (среднее): {np.mean(prevalences):.6f}')

        ax1.set_xlabel('Распространенность FMF (0-49 лет)')
        ax1.set_ylabel('Частота')
        ax1.set_title('Распределение смоделированных распространенностей')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # 2. Динамика ошибок
    ax2 = axes[0, 1]
    errors = [abs(r.get('final_prevalence', 0) - target_prevalence) for r in all_results]

    ax2.plot(range(1, len(errors) + 1), errors, 'bo-', linewidth=1, markersize=4)
    ax2.axhline(y=np.mean(errors), color='r', linestyle='--',
                label=f'Средняя ошибка: {np.mean(errors):.6f}')

    ax2.set_xlabel('Номер симуляции')
    ax2.set_ylabel('Абсолютная ошибка')
    ax2.set_title('Ошибки модели по симуляциям')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Зависимость от размера популяции
    ax3 = axes[1, 0]
    populations = [r.get('final_population', 0) for r in all_results]

    if prevalences and populations:
        ax3.scatter(populations, prevalences, alpha=0.6)
        ax3.axhline(y=target_prevalence, color='r', linestyle='--',
                    label=f'Цель: {target_prevalence:.6f}')

        ax3.set_xlabel('Размер популяции')
        ax3.set_ylabel('Распространенность')
        ax3.set_title('Зависимость распространенности от размера популяции')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # 4. Сводная статистика
    ax4 = axes[1, 1]
    ax4.axis('off')

    if prevalences:
        # Вычисляем статистику для одной успешной симуляции как пример
        example_sim = all_results[0]
        validation_data = example_sim.get('validation_data', {})

        stats_text = [
            "ПРИМЕР ОДНОЙ СИМУЛЯЦИИ",
            "=" * 30,
            f"Реальная распространенность:",
            f"  {target_prevalence:.6f}",
            "",
            f"Смоделированная ({target_year}):",
            f"  Распространенность: {validation_data.get('prevalence', 0):.6f}",
            f"  Население: {validation_data.get('population', 0):.0f}",
            f"  Симптоматические: {validation_data.get('symptomatic_count', 0)}",
            "",
            "СТАТИСТИКА ПО ВСЕМ СИМУЛЯЦИЯМ",
            "=" * 30,
            f"Средняя распространенность: {np.mean(prevalences):.6f}",
            f"Относительная ошибка: {np.mean(errors) / target_prevalence * 100:.1f}%",
            f"p-value: {stats.ttest_1samp(prevalences, target_prevalence)[1]:.4f}"
        ]

        ax4.text(0.05, 0.95, '\n'.join(stats_text), transform=ax4.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("НАЧАЛО ВАЛИДАЦИИ МОДЕЛИ FMF")
    print("=" * 80)

    # Сначала создаем популяцию для аудита
    audit_sim = ValidationSimulation(
        initial_population_size=100_000,
        max_age_limit=49,
        simulation_years=50,
        mutation_frequencies={
            'N': 0.945,  # Возвращаемся к 0.945
            'M694V': 0.040,  # Золотая середина между 0.037 и 0.042
            'V726A': 0.007,
            'M680I': 0.006,
            'R761H': 0.002
        }
    )

    # Инициализируем и прогоняем несколько лет
    audit_sim.initialize_founders()
    for _ in range(30):
        audit_sim._run_single_year()

    # Полный аудит с графиками
    full_population_audit(audit_sim, plot=True)


    # Затем валидация
    results = run_validation_with_real_data(
        n_simulations=100,
        target_year=2012
    )

    # Дополнительно: сравниваем пенетрантность в разных симуляциях
    if results and 'simulation_results' in results:
        plot_penetrance_comparison(results['simulation_results'])