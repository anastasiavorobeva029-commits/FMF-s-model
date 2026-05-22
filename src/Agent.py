import random
import uuid
from typing import Optional

from ModelParams import ModelParams


class Agent:
    """
    Класс агента для симуляции популяции с семейной средиземноморской лихорадкой (FMF).
    Содержит демографические, генетические и клинические параметры, а также
    логику жизненного цикла: старение, заболеваемость, диагностика и репродукция.
    """

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
        self.incidental_diagnosis_chance = 0.02  # Базовая вероятность случайной находки
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

        # Метки происхождения аллелей (гарантируют отсутствие AttributeError в логгерах)
        self.passed_from_father = 'N'
        self.passed_from_mother = 'N'

    def set_genotype(self, allele_1: str, allele_2: str):
        """
        Принудительно устанавливает генотип (например, при рождении)
        и автоматически фиксирует, какой аллель пришел от какого родителя.
        """
        self.mefv_allele_1 = allele_1
        self.mefv_allele_2 = allele_2

        # Автоматический маркер происхождения: первый аргумент всегда отцовский, второй — материнский
        self.passed_from_father = allele_1
        self.passed_from_mother = allele_2

        self.update_genotype_status()

    def update_genotype_status(self):
        """
        Интерпретирует сочетание аллелей и классифицирует генетический статус агента.
        Сюда заложена базовая медицинская логика оценки риска манифестации.
        """
        alleles = [self.mefv_allele_1, self.mefv_allele_2]
        mutant_count = sum(1 for allele in alleles if allele != "N")

        if mutant_count == 0:
            self.genotype_status = "healthy"
            self.mutation_type = None

        elif mutant_count == 1:
            self.genotype_status = "carrier"
            self.mutation_type = "heterozygous"

        else:
            self.genotype_status = "at_risk"
            if len(set(alleles)) == 1:
                allele_type = alleles[0]
                if allele_type == "M694V":
                    self.mutation_type = "M694V_homozygous"
                else:
                    self.mutation_type = "other_homozygous"
            else:
                self.mutation_type = "compound_heterozygous"

    def age_year(self, annual_death_prob: float, current_year: int):
        """Основной шаг симуляции для агента за один календарный год."""
        if not self.alive:
            return

        self.age += 1

        if self.age > self.max_age_limit:
            self.alive = False
            return

        if random.random() < annual_death_prob * self._get_age_weight():
            self.alive = False
            return

        if self.clinical_status == 'asymptomatic' and self.mutation_type:
            prob = self._calculate_annual_onset_probability()
            age_modifier = 1.0 if self.age < 25 else (0.3 if self.age < 40 else 0.11)

            if random.random() < (prob * age_modifier):
                self.clinical_status = 'symptomatic'
                self.age_of_onset = self.age
                self._determine_disease_severity()

        if self.clinical_status == 'symptomatic' and not self.is_diagnosed:
            detection_success = False

            if current_year >= self.params.screening_start_year and self.params.use_screening:
                chance = self.params.screening_coverage * self.params.screening_efficiency
                if random.random() < chance:
                    detection_success = True

            if not detection_success and self.params.do_diagnosing and current_year >= self.params.systemic_diagnosis_start_year:
                if self._try_to_diagnose(current_year):
                    detection_success = True

            if not detection_success:
                chance = self.incidental_diagnosis_chance
                if not self.params.use_screening and self.disease_severity == "mild":
                    chance *= 0.1
                if random.random() < chance:
                    detection_success = True

            if detection_success:
                self._start_basic_therapy()

    def _start_basic_therapy(self):
        self.is_diagnosed = True
        self.on_colchicine = True
        if random.random() < 0.07:
            self.is_colchicine_resistant = True

    def _get_age_weight(self) -> float:
        if self.age < 1:
            return 2.0
        if self.age < 15:
            return 0.15
        if self.age < 45:
            return 0.3
        if self.age < 65:
            return 1.2
        return 2.5 + ((self.age - 65) / (self.max_age_limit - 65)) * 4.0

    def _try_to_diagnose(self, current_year: int) -> bool:
        base_k = self.params.diagnosis_slope * current_year + self.params.diagnosis_intercept

        if self.age <= 18:
            access_rate = base_k * self.params.diagnosis_child_multiplier
        else:
            access_rate = base_k * self.params.diagnosis_adult_multiplier

        if not self.params.use_screening:
            if self.disease_severity == "severe":
                severity_multiplier = 1.5
            elif self.disease_severity == "moderate":
                severity_multiplier = 0.6
            else:
                severity_multiplier = 0.05
            access_rate *= severity_multiplier

        return random.random() < max(self.params.diagnosis_min_prob, min(access_rate, self.params.diagnosis_max_prob))

    def _calculate_annual_onset_probability(self) -> float:
        if self.clinical_status != 'asymptomatic':
            return 0.0
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
        if self.mutation_type == "M694V_homozygous":
            weights = [0.1, 0.3, 0.6]
        elif self.mutation_type == "compound_heterozygous":
            if "M694V" in [self.mefv_allele_1, self.mefv_allele_2]:
                weights = [0.2, 0.4, 0.4]
            else:
                weights = [0.4, 0.4, 0.2]
        else:
            weights = [0.6, 0.3, 0.1]

        self.disease_severity = random.choices(["mild", "moderate", "severe"], weights=weights)[0]

        resistance_chance = 0.12 if self.disease_severity == "severe" else (
            0.01 if self.disease_severity == "mild" else 0.05)
        if random.random() < resistance_chance:
            self.is_colchicine_resistant = True

    def set_partner(self, partner: 'Agent'):
        self.partner_id = partner.id
        partner.partner_id = self.id

    def add_child(self, child_id: str):
        if child_id not in self.children_ids:
            self.children_ids.append(child_id)

    def get_children_count(self) -> int:
        return len(self.children_ids)

    def can_get_pregnant(self, current_year: int, birth_cooldown: int) -> bool:
        if self.age < 18 or self.age > 45:
            return False

        if current_year - self.last_birth_year < birth_cooldown:
            return False

        if self.clinical_status == 'symptomatic':
            base_penalty = 0.3
            if self.on_colchicine:
                if not self.is_colchicine_resistant or self.on_antibodies:
                    penalty = base_penalty * (1.0 - self.params.fertility_recovery)
                else:
                    penalty = base_penalty
            else:
                penalty = base_penalty

            if random.random() < penalty:
                return False

        return True