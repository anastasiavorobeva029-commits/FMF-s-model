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
    initial_population_size: int = 1_000
    diagnosis_slope: float = 0.012350
    diagnosis_intercept: float = -24.52535
    diagnosis_min_prob: float = 0.05
    diagnosis_max_prob: float = 0.98
    diagnosis_child_multiplier: float = 1.3
    diagnosis_adult_multiplier: float = 0.8
    bio_access_pgt_scenario: float = 0.15
    bio_access_screening_scenario: float = 0.05
    bio_access_baseline: float = 0.0

    @staticmethod
    def scenario_1():
        return ModelParams(
            ethnic_assortativity=0.85,
            do_diagnosing=True,
            use_screening=False,
            screening_coverage=0,
            screening_efficiency=0,
            fertility_recovery=0.5,
            ethnic_distribution={'Armenian': 0.9, 'Other': 0.1},
            use_pgt=False,
            pgt_efficiency=0.0
        )

    @staticmethod
    def scenario_2():
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
        return ModelParams(
            ethnic_assortativity=0.55,
            do_diagnosing=True,
            use_screening=True,
            screening_coverage=0.8,
            screening_efficiency=0.8,
            fertility_recovery=0.95,
            ethnic_distribution={'Armenian': 0.6, 'Other': 0.4},
            use_pgt=True,
            pgt_efficiency=0.85
        )

    def get_bio_access_chance(self):
        if self.use_pgt:
            return self.bio_access_pgt_scenario
        if self.use_screening:
            return self.bio_access_screening_scenario
        return self.bio_access_baseline

    @staticmethod
    def from_dictionary(d: Dict):
        """Создает ModelParams из словаря (для анализа чувствительности)"""
        # Начинаем со сценария 1
        base = ModelParams.scenario_1()

        # Собираем параметры для конструктора
        params_dict = {
            'ethnic_assortativity': d.get('ethnic_assortativity', base.ethnic_assortativity),
            'do_diagnosing': d.get('do_diagnosing', base.do_diagnosing),
            'use_screening': d.get('use_screening', base.use_screening),
            'use_pgt': d.get('use_pgt', base.use_pgt),
            'pgt_efficiency': d.get('pgt_efficiency', base.pgt_efficiency),
            'screening_coverage': d.get('screening_coverage', base.screening_coverage),
            'screening_efficiency': d.get('screening_efficiency', base.screening_efficiency),
            'fertility_recovery': d.get('fertility_recovery', base.fertility_recovery),
            'ethnic_distribution': d.get('ethnic_distribution', base.ethnic_distribution),
            'systemic_diagnosis_start_year': d.get('systemic_diagnosis_start_year', base.systemic_diagnosis_start_year),
            'screening_start_year': d.get('screening_start_year', base.screening_start_year),
            'pgt_start_year': d.get('pgt_start_year', base.pgt_start_year),
            'num_runs': d.get('num_runs', base.num_runs),
            'max_age_limit': d.get('max_age_limit', base.max_age_limit),
            'initial_population_size': d.get('initial_population_size', base.initial_population_size),
            'diagnosis_slope': d.get('diagnosis_slope', base.diagnosis_slope),
            'diagnosis_intercept': d.get('diagnosis_intercept', base.diagnosis_intercept),
            'diagnosis_min_prob': d.get('diagnosis_min_prob', base.diagnosis_min_prob),
            'diagnosis_max_prob': d.get('diagnosis_max_prob', base.diagnosis_max_prob),
            'diagnosis_child_multiplier': d.get('diagnosis_child_multiplier', base.diagnosis_child_multiplier),
            'diagnosis_adult_multiplier': d.get('diagnosis_adult_multiplier', base.diagnosis_adult_multiplier),
            'bio_access_pgt_scenario': d.get('bio_access_pgt_scenario', base.bio_access_pgt_scenario),
            'bio_access_screening_scenario': d.get('bio_access_screening_scenario', base.bio_access_screening_scenario),
            'bio_access_baseline': d.get('bio_access_baseline', base.bio_access_baseline),
        }

        # Критическая логика: включаем флаги если соответствующие параметры > 0
        if params_dict['screening_coverage'] > 0.01:
            params_dict['use_screening'] = True

        if params_dict['pgt_efficiency'] > 0.01:
            params_dict['use_pgt'] = True

        return ModelParams(**params_dict)

    def to_dictionary(self) -> Dict:
        """Преобразует объект в словарь"""
        return {
            'ethnic_assortativity': self.ethnic_assortativity,
            'do_diagnosing': self.do_diagnosing,
            'use_screening': self.use_screening,
            'use_pgt': self.use_pgt,
            'pgt_efficiency': self.pgt_efficiency,
            'screening_coverage': self.screening_coverage,
            'screening_efficiency': self.screening_efficiency,
            'fertility_recovery': self.fertility_recovery,
            'ethnic_distribution': self.ethnic_distribution,
            'systemic_diagnosis_start_year': self.systemic_diagnosis_start_year,
            'screening_start_year': self.screening_start_year,
            'pgt_start_year': self.pgt_start_year,
            'num_runs': self.num_runs,
            'max_age_limit': self.max_age_limit,
            'initial_population_size': self.initial_population_size,
            'diagnosis_slope': self.diagnosis_slope,
            'diagnosis_intercept': self.diagnosis_intercept,
            'diagnosis_min_prob': self.diagnosis_min_prob,
            'diagnosis_max_prob': self.diagnosis_max_prob,
            'diagnosis_child_multiplier': self.diagnosis_child_multiplier,
            'diagnosis_adult_multiplier': self.diagnosis_adult_multiplier,
            'bio_access_pgt_scenario': self.bio_access_pgt_scenario,
            'bio_access_screening_scenario': self.bio_access_screening_scenario,
            'bio_access_baseline': self.bio_access_baseline
        }

    def items(self):
        """Возвращает параметры модели в виде списка (key, value)"""
        return list(self.to_dictionary().items())

