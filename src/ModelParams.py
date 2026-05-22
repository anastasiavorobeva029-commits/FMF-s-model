from dataclasses import dataclass
from typing import Dict


@dataclass
class ModelParams:

    ethnic_assortativity: float
    do_diagnosing: bool
    use_screening: bool
    use_pgt: bool
    pgt_efficiency: float
    screening_coverage: float
    screening_efficiency: float
    fertility_recovery: float
    ethnic_distribution: dict

    systemic_diagnosis_start_year: int = 1990
    screening_start_year: int = 2010
    pgt_start_year: int = 2018
    num_runs: int = 2
    max_age_limit: int = 85
    initial_population_size: int = 5_000
    diagnosis_slope: float = 0.012350
    diagnosis_intercept: float = -24.52535
    diagnosis_min_prob: float = 0.05
    diagnosis_max_prob: float = 0.98
    diagnosis_child_multiplier: float = 1.3
    diagnosis_adult_multiplier: float = 0.8
    bio_access_pgt_scenario = 0.15
    bio_access_screening_scenario = 0.05
    bio_access_baseline = 0.0


    @staticmethod
    def scenario_1():
        # Базовый сценарий: статус-кво.
        # do_diagnosing ставим True, чтобы гомозиготы с тяжелой клиникой могли получить диагноз.
        # Скрининга нет, решений о планировании нет, фертильность урезана (нет колхицина).
        return ModelParams(ethnic_assortativity=0.85,
                           do_diagnosing=True,
                           use_screening=False,
                           screening_coverage=0,
                           screening_efficiency=0,
                           fertility_recovery=0.5,
                           ethnic_distribution={'Armenian': 0.9, 'Other': 0.1},
                           use_pgt=False,
                           pgt_efficiency=0.0)

    @staticmethod
    def scenario_2():
        # Скрининг и доступный колхицин (фертильность 0.85).
        # Охват скрининга 30%. Из выявленных пар 70% принимают решение рожать (screening_efficiency=0.7).
        # PGT нет, рожают естественным путем.
        return ModelParams(
            ethnic_assortativity=0.75,
            do_diagnosing=True,
            use_screening=True,
            screening_coverage=0.6,
            screening_efficiency=0.3,
            fertility_recovery=0.85,
            ethnic_distribution={'Armenian': 0.8, 'Other': 0.20},
            use_pgt=True,
            pgt_efficiency=0.5
        )

    @staticmethod
    def scenario_3():
        # Массовый скрининг (80%). 80% пар принимают решение в пользу рождения (screening_efficiency=0.8),
        # но делают это через ЭКО + PGT (решение об ЭКО принимают 50%).
        return ModelParams(
            ethnic_assortativity=0.55,
            do_diagnosing=True,
            use_screening=True,
            screening_coverage=0.8,
            screening_efficiency=0.8,
            fertility_recovery= 0.95,
            ethnic_distribution={'Armenian': 0.6, 'Other': 0.4},
            use_pgt=True,
            pgt_efficiency=0.85
        )


    def get_bio_access_chance(self):
        if self.use_pgt: return self.bio_access_pgt_scenario
        if self.use_screening: return self.bio_access_screening_scenario
        return self.bio_access_baseline


    # недописанный конструктор для анализа чувствительности, чтобы он вышел с ошибкой для дальнейшей доработки
    @staticmethod
    def from_dictionary(d: Dict):
        raise NotImplementedError

# преобразование модели в список параметров.
    def items(self):
        return []

