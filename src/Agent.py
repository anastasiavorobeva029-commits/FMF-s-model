import random
import uuid
from typing import Optional

from ModelParams import ModelParams


class Agent:

    def __init__(self, params: ModelParams, gender: str, age: int,
                 generation: int,
                 birth_year: int,
                 max_age_limit: int = 85,
                 ethnicity: str = 'Armenian',
                 mating_strategy: str = 'endogamy',
                 father_id: Optional[str] = None,
                 mother_id: Optional[str] = None):

        self.params = params
        self.id = str(uuid.uuid4())[:8]
        self.gender = gender
        self.age = age
        self.birth_year = birth_year
        self.max_age_limit = max_age_limit
        self.generation = generation
        self.ethnicity = ethnicity
        self.mating_strategy = mating_strategy
        self.alive = True

        # Клинические параметры
        self.clinical_status = 'asymptomatic'
        self.disease_severity = None
        self.age_of_onset = None
        self.on_colchicine = False
        self.is_diagnosed = False
        self.is_screened = False
        self.incidental_diagnosis_chance = 0.02
        self.is_colchicine_resistant = False
        self.on_antibodies = False

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
        if not self.alive:
            return

        self.age += 1
        if self.age > self.max_age_limit:
            self.alive = False
            return

        # Смертность
        if random.random() < annual_death_prob * self._get_age_weight():
            self.alive = False
            return

        # Манифестация FMF
        if self.clinical_status == 'asymptomatic' and self.mutation_type:
            prob = self._calculate_annual_onset_probability()
            age_modifier = 1.0 if self.age < 25 else (0.3 if self.age < 40 else 0.11)

            if random.random() < (prob * age_modifier):
                self.clinical_status = 'symptomatic'
                self.age_of_onset = self.age
                self._determine_disease_severity()

        # Диагностика и старт лечения
        if self.clinical_status == 'symptomatic' and not self.is_diagnosed:
            detection_success = False

            # А) Скрининг
            if current_year >= self.params.screening_start_year and self.params.use_screening:
                chance = self.params.screening_coverage * self.params.screening_efficiency
                if random.random() < chance:
                    detection_success = True

            # Б) Системная диагностика (врач)
            if not detection_success and self.params.do_diagnosing and current_year >= self.params.systemic_diagnosis_start_year:
                if self._try_to_diagnose(current_year):
                    detection_success = True

            # В) Случайный диагноз
            if not detection_success:
                chance = self.incidental_diagnosis_chance
                if not self.params.use_screening and self.disease_severity == "mild":
                    chance *= 0.1

                if random.random() < chance:
                    detection_success = True

            # ⚡ ИСПРАВЛЕНО: Если один из методов диагностики сработал,
            # мы обязаны физически назначить терапию (включить колхицин)
            if detection_success:
                self._start_basic_therapy()

    def _start_basic_therapy(self):
        self.is_diagnosed = True
        self.on_colchicine = True

        # 🎯 Определяем резистентность строго среди тех, кто начал принимать препарат.
        # В среднем 7% пациентов (попадаем в клинический коридор 5-10% для Армении).
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

    def _try_to_diagnose(self, current_year: int):
        base_k = self.params.diagnosis_slope * current_year + self.params.diagnosis_intercept

        if self.age <= 18:
            access_rate = base_k * self.params.diagnosis_child_multiplier
        else:
            access_rate = base_k * self.params.diagnosis_adult_multiplier

        # Модификатор тяжести для Сценария 1 (статус-кво)
        if not self.params.use_screening:
            if self.disease_severity == "severe":
                severity_multiplier = 1.5
            elif self.disease_severity == "moderate":
                severity_multiplier = 0.6
            else:
                severity_multiplier = 0.05

            access_rate *= severity_multiplier

        access_rate = max(self.params.diagnosis_min_prob, min(access_rate, self.params.diagnosis_max_prob))

        # ⚡ ИСПРАВЛЕНО: Этот метод возвращает True/False для логики в age_year.
        # Само назначение флага перенесено в блок инициации терапии.
        return random.random() < access_rate

    def _calculate_annual_onset_probability(self):
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

        if self.disease_severity == "severe":
            resistance_chance = 0.12
        elif self.disease_severity == "mild":
            resistance_chance = 0.01
        else:
            resistance_chance = 0.05

        if random.random() < resistance_chance:
            self.is_colchicine_resistant = True

    def set_partner(self, partner: 'Agent'):
        self.partner_id = partner.id
        partner.partner_id = self.id

    def add_child(self, child_id: str):
        if child_id not in self.children_ids:
            self.children_ids.append(child_id)

    def get_children_count(self):
        return len(self.children_ids)

    def can_get_pregnant(self, current_year: int, birth_cooldown: int) -> bool:
        # 1. Возрастной лимит
        if self.age < 18 or self.age > 45:
            return False

        # 2. Интервал между родами
        if current_year - self.last_birth_year < birth_cooldown:
            return False

        # 3. Влияние Периодической болезни и колхицина
        if self.clinical_status == 'symptomatic':
            base_penalty = 0.3  # Штраф к фертильности 30%

            # реальное лечение определяется ТОЛЬКО флагом нахождения на терапии.
            # Если диагноз поставлен и агент получает препарат, оцениваем эффективность.
            if self.on_colchicine:
                if not self.is_colchicine_resistant or self.on_antibodies:
                    # Терапия успешна -> восстанавливаем фертильность по параметрам сценария
                    penalty = base_penalty * (1.0 - self.params.fertility_recovery)
                else:
                    # Есть резистентность к колхицину, а биологическая терапия (антитела) недоступна
                    penalty = base_penalty
            else:
                # Диагноз не поставлен, колхицин не назначен -> штраф максимальный
                penalty = base_penalty

            if random.random() < penalty:
                return False

        return True