import random
import uuid
from typing import Optional

from ModelParams import ModelParams


class Agent:

    # Класс агента для симуляции популяции с семейной средиземноморской лихорадкой (FMF).
    # Содержит демографические, генетические и клинические параметры, а также
    # логику жизненного цикла: старение, заболеваемость, диагностика и репродукция.


    def __init__(self, params: ModelParams, gender: str, age: int,
                 generation: int,
                 birth_year: int,
                 max_age_limit: int = 85,
                 ethnicity: str = 'Armenian',
                 mating_strategy: str = 'endogamy',
                 father_id: Optional[str] = None,
                 mother_id: Optional[str] = None):

        self.params = params

        # Уникальный короткий идентификатор для удобства логирования и экономии памяти
        self.id = str(uuid.uuid4())[:8]
        self.gender = gender
        self.age = age
        self.birth_year = birth_year
        self.max_age_limit = max_age_limit
        self.generation = generation
        self.ethnicity = ethnicity
        self.mating_strategy = mating_strategy
        self.alive = True

        # Клинические параметры и фенотип
        self.clinical_status = 'asymptomatic'  # Текущий статус: 'asymptomatic' или 'symptomatic'
        self.disease_severity = None  # Тяжесть течения: 'mild', 'moderate', 'severe'
        self.age_of_onset = None  # Возраст, в котором проявились первые симптомы
        self.on_colchicine = False  # Находится ли пациент на базовой терапии колхицином
        self.is_diagnosed = False  # Поставлен ли официальный клинический диагноз
        self.is_screened = False  # Прошел ли агент популяционный скрининг
        self.incidental_diagnosis_chance = 0.02  # Базовая вероятность случайной находки у врача общей практики
        self.is_colchicine_resistant = False  # Наличие резистентности к колхицину
        self.on_antibodies = False  # Получает ли таргетную биологическую терапию (ИЛ-1)

        # Генетический паспорт (ген MEFV)
        self.mefv_allele_1 = 'N'  # 'N' — норма, либо конкретный тип мутации (напр. 'M694V')
        self.mefv_allele_2 = 'N'
        self.genotype_status = 'healthy'  # 'healthy', 'carrier', 'at_risk'
        self.mutation_type = None  # Тип мутации (гетерозигота, гомозигота, компаунд)

        # Семейные связи и репродуктивная история
        self.father_id = father_id
        self.mother_id = mother_id
        self.partner_id = None
        self.children_ids = []
        self.last_birth_year = -100  # Нужно для контроля интервала между родами (cooldown)

    def set_genotype(self, allele_1: str, allele_2: str):
        # Принудительно устанавливает генотип (например, при рождении) и обновляет статус.
        self.mefv_allele_1 = allele_1
        self.mefv_allele_2 = allele_2
        self.update_genotype_status()

    def update_genotype_status(self):

        # Интерпретирует сочетание аллелей и классифицирует генетический статус агента.
        # Сюда заложена базовая медицинская логика оценки риска манифестации.

        alleles = [self.mefv_allele_1, self.mefv_allele_2]
        mutant_count = sum(1 for allele in alleles if allele != "N")

        if mutant_count == 0:
            self.genotype_status = "healthy"
            self.mutation_type = None

        elif mutant_count == 1:
            # Здоровый носитель, низкий риск проявления симптомов
            self.genotype_status = "carrier"
            self.mutation_type = "heterozygous"

        else:
            # Две мутации — агент попадает в группу высокого клинического риска
            self.genotype_status = "at_risk"

            if len(set(alleles)) == 1:
                # Обе мутации идентичны (истинная гомозигота)
                allele_type = alleles[0]
                if allele_type == "M694V":
                    # Мутация M694V выделена отдельно из-за доказанной тяжести течения в популяции
                    self.mutation_type = "M694V_homozygous"
                else:
                    self.mutation_type = "other_homozygous"
            else:
                # Две разные мутации в главе MEFV
                self.mutation_type = "compound_heterozygous"

    def age_year(self, annual_death_prob: float, current_year: int):

        # Основной шаг симуляции для агента за один календарный год.
        # Пересчитывает возраст, риски смерти, развитие симптомов и прохождение диагностики.
        if not self.alive:
            return

        self.age += 1

        # Естественная смерть по достижении предельного возраста модели
        if self.age > self.max_age_limit:
            self.alive = False
            return

        # 1. Расчет вероятности естественной смертности с учетом возрастной группы
        if random.random() < annual_death_prob * self._get_age_weight():
            self.alive = False
            return

        # 2. Манифестация заболевания (переход из бессимптомного в симптоматическое состояние)
        if self.clinical_status == 'asymptomatic' and self.mutation_type:
            prob = self._calculate_annual_onset_probability()

            # Большинство дебютов FMF приходится на возраст до 25 лет.
            # Вводим коэффициенты затухания вероятности для старших групп.
            age_modifier = 1.0 if self.age < 25 else (0.3 if self.age < 40 else 0.11)

            if random.random() < (prob * age_modifier):
                self.clinical_status = 'symptomatic'
                self.age_of_onset = self.age
                self._determine_disease_severity()

        # 3. Логика выявления болезни (только для агентов с активными симптомами)
        if self.clinical_status == 'symptomatic' and not self.is_diagnosed:
            detection_success = False

            # Канал А: Скрининговая программа (если развернута по сценарию)
            if current_year >= self.params.screening_start_year and self.params.use_screening:
                chance = self.params.screening_coverage * self.params.screening_efficiency
                if random.random() < chance:
                    detection_success = True

            # Канал Б: Системная диагностика (обращение к профильному врачу)
            if not detection_success and self.params.do_diagnosing and current_year >= self.params.systemic_diagnosis_start_year:
                if self._try_to_diagnose(current_year):
                    detection_success = True

            # Канал В: Случайная диагностика (находка при поиске других патологий)
            if not detection_success:
                chance = self.incidental_diagnosis_chance

                # Если скрининга нет, а форма легкая, пациент годами списывает боли на другие причины
                if not self.params.use_screening and self.disease_severity == "mild":
                    chance *= 0.1

                if random.random() < chance:
                    detection_success = True

            # Регистрация факта выявления и немедленный перевод на терапию
            if detection_success:
                self._start_basic_therapy()

    def _start_basic_therapy(self):
        # Внутренний метод: фиксирует диагноз и переводит пациента на постоянный прием лекарств
        self.is_diagnosed = True
        self.on_colchicine = True

        # Устанавливаем врожденную непереносимость/резистентность к препарату.
        # Средний популяционный показатель для Армении держится в районе 7%.
        if random.random() < 0.07:
            self.is_colchicine_resistant = True

    def _get_age_weight(self) -> float:
        # Возвращает весовой коэффициент для национальной таблицы смертности в зависимости от возраста.
        if self.age < 1:
            return 2.0  # Младенческая смертность (стандартный пик)
        if self.age < 15:
            return 0.15  # Самый безопасный жизненный период
        if self.age < 45:
            return 0.3
        if self.age < 65:
            return 1.2
        # Прогрессивный рост рисков для пожилого населения
        return 2.5 + ((self.age - 65) / (self.max_age_limit - 65)) * 4.0

    def _try_to_diagnose(self, current_year: int) -> bool:
        # Вычисляет вероятность обнаружения болезни врачом на основе параметров текущего сценария
        # Линейный тренд улучшения качества медицины с годами
        base_k = self.params.diagnosis_slope * current_year + self.params.diagnosis_intercept

        # Разделение доступности помощи для детей и взрослых
        if self.age <= 18:
            access_rate = base_k * self.params.diagnosis_child_multiplier
        else:
            access_rate = base_k * self.params.diagnosis_adult_multiplier

        # Коррекция для Сценария 1 (отсутствие системного скрининга):
        # При таком подходе тяжелые пациенты выявляются быстрее, а легкие — почти никогда
        if not self.params.use_screening:
            if self.disease_severity == "severe":
                severity_multiplier = 1.5
            elif self.disease_severity == "moderate":
                severity_multiplier = 0.6
            else:
                severity_multiplier = 0.05

            access_rate *= severity_multiplier

        # Жестко зажимаем итоговую вероятность в валидные математические границы [0, 1]
        access_rate = max(self.params.diagnosis_min_prob, min(access_rate, self.params.diagnosis_max_prob))

        return random.random() < access_rate

    def _calculate_annual_onset_probability(self) -> float:
        # Возвращает базовый риск манифестации симптомов за год на основании типа мутации
        if self.clinical_status != 'asymptomatic':
            return 0.0

        # Вероятности откалиброваны по клиническим данным распределения дебютов FMF
        if self.mutation_type == "M694V_homozygous":
            return 0.025
        elif self.mutation_type == "compound_heterozygous":
            return 0.051
        elif self.mutation_type == "heterozygous":
            return 0.00045
        elif self.mutation_type == "other_homozygous":
            return 0.005
        return 0.0

    def _determine_disease_severity(self):
        # Определяет тяжесть течения заболевания и рассчитывает риски резистентности
        # Распределение тяжести завязано на генетический бэкграунд
        if self.mutation_type == "M694V_homozygous":
            weights = [0.1, 0.3, 0.6]  # Сдвиг в сторону тяжелого течения
        elif self.mutation_type == "compound_heterozygous":
            # Наличие аллеля M694V усложняет картину даже в компаунд-состоянии
            if "M694V" in [self.mefv_allele_1, self.mefv_allele_2]:
                weights = [0.2, 0.4, 0.4]
            else:
                weights = [0.4, 0.4, 0.2]
        else:
            weights = [0.6, 0.3, 0.1]  # Менее агрессивные мутации чаще дают легкую форму

        self.disease_severity = random.choices(["mild", "moderate", "severe"], weights=weights)[0]

        # Вероятность вторичной резистентности к лечению коррелирует с тяжестью фенотипа
        if self.disease_severity == "severe":
            resistance_chance = 0.12
        elif self.disease_severity == "mild":
            resistance_chance = 0.01
        else:
            resistance_chance = 0.05

        if random.random() < resistance_chance:
            self.is_colchicine_resistant = True

    def set_partner(self, partner: 'Agent'):
        # Связывает двух агентов в брачную пару (взаимное присвоение ID)
        self.partner_id = partner.id
        partner.partner_id = self.id

    def add_child(self, child_id: str):
        # Добавляет идентификатор ребенка в историю родов текущего агента
        if child_id not in self.children_ids:
            self.children_ids.append(child_id)

    def get_children_count(self) -> int:
        # Возвращает общее число рожденных агентом детей.
        return len(self.children_ids)

    def can_get_pregnant(self, current_year: int, birth_cooldown: int) -> bool:

        # Проверяет, способна ли женщина зачать ребенка в текущем шаге симуляции.
        # Учитывает возраст, репродуктивный интервал и влияние некомпенсированной периодической болезни.

        # 1. Проверка биологического репродуктивного окна
        if self.age < 18 or self.age > 45:
            return False

        # 2. Проверка физиологического интервала восстановления между беременностями
        if current_year - self.last_birth_year < birth_cooldown:
            return False

        # 3. Оценка влияния FMF на фертильность (воспалительные процессы снижают вероятность зачатия)
        if self.clinical_status == 'symptomatic':
            base_penalty = 0.3  # Без лечения активный воспалительный процесс снижает фертильность на 30%

            if self.on_colchicine:
                # Пациент лечится. Проверяем, помогает ли терапия
                if not self.is_colchicine_resistant or self.on_antibodies:
                    # Терапия эффективна (колхицин работает или подключена биологическая терапия)
                    # Частично или полностью убираем штраф к фертильности согласно параметрам модели
                    penalty = base_penalty * (1.0 - self.params.fertility_recovery)
                else:
                    # Препарат принимает, но есть резистентность, а таргетной терапии нет. Штраф остается.
                    penalty = base_penalty
            else:
                # Болезнь запущена, терапии нет — накладывается максимальный негативный эффект
                penalty = base_penalty

            # Симулируем случайный сбой зачатия из-за воспалительного фактора
            if random.random() < penalty:
                return False

        return True