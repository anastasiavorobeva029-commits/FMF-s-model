import pandas as pd
import numpy as np
from typing import Optional, Any, Union
import random
import uuid
from collections import defaultdict
from tqdm import tqdm  # добавьте этот импорт в начало файла
import sys
from math import sqrt
from scipy import stats
from collections import Counter
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="main thread is not in main loop")
import matplotlib.pyplot as plt
import seaborn as sns
import time
import psutil
import hashlib
from rich.console import Console
import matplotlib
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from typing import Dict, List, Tuple
from threading import Lock


def load_demographic_data():

    files = {
        'birth_rate': 'birth_rate.csv',
        'death_rate': 'death_rate.csv',
        'fertility_rate': 'fertility_rate.csv',
        'age_structure_1950': 'age_structure_1950.csv',
        'age_fertility_dist': 'age_fertility_dist.csv'
    }

    data = {}
    for name, filename in files.items():
        try:
            # Проверяем существование файла
            if not os.path.exists(filename):
                raise FileNotFoundError(f"Файл {filename} не найден в текущей директории")

            df = pd.read_csv(filename, sep=';', skipinitialspace=True, decimal=',')

            if name in ['age_structure_1950', 'age_fertility_dist']:
                df.set_index(df.columns[0], inplace=True)
                df.index.name = 'Age_Group'
            else:
                df.set_index(df.columns[0], inplace=True)
                df.index.name = 'Year'
                df.columns = [name]

            data[name] = df
            print(f"  ✓ Загружен {filename}: {df.shape}")

        except Exception as e:
            print(f"  ❌ Ошибка загрузки {filename}: {e}")
            raise

    return (
        data['birth_rate'],
        data['death_rate'],
        data['fertility_rate'],
        data['age_structure_1950'],
        data['age_fertility_dist']
    )

def aggregate_multiple_runs(results_list: List[Dict], target_values: Dict = None) -> pd.DataFrame:
    """
    Агрегирует результаты множественных прогонов с доверительными интервалами
    """
    df = pd.DataFrame(results_list)

    # Метрики для агрегации
    metrics = [
        'm694v_homo_in_affected_pct',
        'compound_in_affected_pct',
        'hetero_in_affected_pct',
        'other_homo_in_affected_pct',
        'late_onset_pct',
        'prevalence_pct',
        'final_population',
        'avg_model_birth_rate',
        'avg_model_death_rate'
    ]

    # Фильтруем существующие метрики
    available_metrics = [m for m in metrics if m in df.columns]

    if not available_metrics:
        return pd.DataFrame()

    print(f"📊 Агрегация {len(results_list)} записей из {len(df['run_id'].unique())} прогонов")

    # Статистика
    summary = []
    for metric in available_metrics:
        values = df[metric].dropna()
        if len(values) == 0:
            continue

        mean_val = values.mean()
        std_val = values.std()
        ci_95 = 1.96 * std_val / np.sqrt(len(values))

        result = {
            'metric': metric,
            'mean': mean_val,
            'std': std_val,
            'ci_lower': mean_val - ci_95,
            'ci_upper': mean_val + ci_95,
            'min': values.min(),
            'max': values.max(),
            'n_runs': len(values)
        }

        # Добавляем целевое значение если есть
        if target_values and metric in target_values:
            result['target'] = target_values[metric]
            result['diff'] = mean_val - target_values[metric]
            result['within_target'] = abs(result['diff']) <= ci_95

        summary.append(result)

    # Диагностика прогонов (можно убрать или оставить)
    # for i, result in enumerate(results_list):
    #     print(f"Прогон {i + 1}:")
    #     print(f"  Всего больных: {result.get('total_affected', 0)}")
    #     print(f"  Компаунды абс: {result.get('compound_abs', 0)}")
    #     print(f"  Компаунды %: {result.get('compound_in_affected_pct', 0):.2f}%")

    return pd.DataFrame(summary)


def plot_convergence(results_list: List[Dict], metrics: List[str],
                     target_values: Dict = None, figsize=(15, 10)):
    """
    Показывает, как средние значения сходятся к цели с увеличением числа прогонов
    """
    df = pd.DataFrame(results_list)
    n_runs = len(df)

    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Убедимся, что axes всегда список
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        if idx >= len(axes) or metric not in df.columns:
            continue

        ax = axes[idx]
        values = df[metric].dropna()

        if len(values) < 2:
            ax.text(0.5, 0.5, f'Not enough data for {metric}',
                    ha='center', va='center')
            continue

        # Cumulative mean and CI
        cum_mean = values.expanding().mean()
        cum_std = values.expanding().std()
        cum_ci = 1.96 * cum_std / np.sqrt(np.arange(1, len(values) + 1))

        # Plot
        x = range(1, len(values) + 1)
        ax.plot(x, values, 'o', alpha=0.3, markersize=3, label='Individual runs')
        ax.plot(x, cum_mean, 'b-', linewidth=2, label='Cumulative mean')
        ax.fill_between(x, cum_mean - cum_ci, cum_mean + cum_ci,
                        alpha=0.2, color='blue', label='95% CI')

        # Target line
        if target_values and metric in target_values:
            target = target_values[metric]
            ax.axhline(y=target, color='r', linestyle='--',
                       linewidth=2, label=f'Target: {target}%')

        # Final value annotation
        final_mean = cum_mean.iloc[-1]
        final_ci = cum_ci.iloc[-1]
        ax.text(0.98, 0.98,
                f'Final: {final_mean:.2f}% ± {final_ci:.2f}%\nn={len(values)}',
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel('Run number')
        ax.set_ylabel('%')
        ax.set_title(f'Convergence: {metric}')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

    # Hide empty subplots
    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    return fig


def analyze_yearly_median_across_runs(results_list: List[Dict],
                                      years_range: range = range(2012, 2025),
                                      output_file: str = 'yearly_median_analysis.csv'):

    key_metrics = [
        'total_agents', 'total_affected', 'prevalence_pct',
        'age_0_14_abs', 'age_15_29_abs', 'age_30_49_abs',
        'age_0_14_pct', 'age_15_29_pct', 'age_30_49_pct',
        'm694v_homo_abs', 'compound_abs', 'other_homo_abs', 'hetero_abs',
        'm694v_homo_in_affected_pct', 'compound_in_affected_pct',
        'other_homo_in_affected_pct', 'hetero_in_affected_pct',
        'final_population', 'total_births', 'total_deaths',
        'allele_freq_M694V', 'allele_freq_R761H', 'allele_freq_V726A',
        'allele_freq_M680I', 'allele_freq_N'
    ]

    # Создаем DataFrame из всех результатов
    df_all = pd.DataFrame(results_list)

    # Фильтруем по годам
    df_filtered = df_all[df_all['year'].isin(years_range)].copy()

    if df_filtered.empty:
        print(f"❌ Нет данных за годы {min(years_range)}-{max(years_range)}")
        return pd.DataFrame()

    # Группируем по годам и считаем медиану, 25-й и 75-й перцентили
    yearly_stats = []

    for year in sorted(df_filtered['year'].unique()):
        year_data = df_filtered[df_filtered['year'] == year]

        stats = {'year': year, 'n_runs': len(year_data)}

        for metric in key_metrics:
            if metric in year_data.columns:
                values = year_data[metric].dropna()
                if len(values) > 0:
                    stats[f'{metric}_median'] = values.median()
                    stats[f'{metric}_q25'] = values.quantile(0.25)
                    stats[f'{metric}_q75'] = values.quantile(0.75)
                    stats[f'{metric}_min'] = values.min()
                    stats[f'{metric}_max'] = values.max()

        yearly_stats.append(stats)

    # Создаем итоговый DataFrame
    result_df = pd.DataFrame(yearly_stats)

    # Сохраняем в CSV
    result_df.to_csv(output_file, index=False)
    print(f"💾 Сохранено {len(result_df)} лет анализа в {output_file}")

    return result_df


def print_yearly_analysis_summary(yearly_df, metrics_to_show=None):
    """
    Красивый вывод анализа по годам
    """
    if metrics_to_show is None:
        metrics_to_show = [
            'total_agents', 'total_affected', 'prevalence_pct',
            'm694v_homo_in_affected_pct', 'compound_in_affected_pct',
            'hetero_in_affected_pct', 'other_homo_in_affected_pct'
        ]

    print("\n" + "=" * 100)
    print(f"📊 МЕДИАННЫЙ АНАЛИЗ ПО ГОДАМ (2012-2024)")
    print("=" * 100)

    for metric in metrics_to_show:
        metric_cols = [col for col in yearly_df.columns if col.startswith(metric)]
        if not metric_cols:
            continue

        print(f"\n📈 {metric.replace('_', ' ').title()}:")
        print("-" * 80)
        print(f"{'Год':<6} | {'Медиана':>12} | {'25%':>10} | {'75%':>10} | {'Min':>10} | {'Max':>10} | {'N runs':>6}")
        print("-" * 80)

        for _, row in yearly_df.iterrows():
            median = row.get(f'{metric}_median', float('nan'))
            q25 = row.get(f'{metric}_q25', float('nan'))
            q75 = row.get(f'{metric}_q75', float('nan'))
            min_val = row.get(f'{metric}_min', float('nan'))
            max_val = row.get(f'{metric}_max', float('nan'))

            # Форматируем в зависимости от типа метрики
            if 'pct' in metric or 'freq' in metric:
                print(
                    f"{int(row['year']):<6} | {median:>11.2f}% | {q25:>9.2f}% | {q75:>9.2f}% | {min_val:>9.2f}% | {max_val:>9.2f}% | {int(row['n_runs']):>6}")
            elif 'agents' in metric or 'population' in metric or 'abs' in metric:
                print(
                    f"{int(row['year']):<6} | {median:>12,.0f} | {q25:>10,.0f} | {q75:>10,.0f} | {min_val:>10,.0f} | {max_val:>10,.0f} | {int(row['n_runs']):>6}")
            else:
                print(
                    f"{int(row['year']):<6} | {median:>12.3f} | {q25:>10.3f} | {q75:>10.3f} | {min_val:>10.3f} | {max_val:>10.3f} | {int(row['n_runs']):>6}")

    print("\n" + "=" * 100)


def plot_yearly_trends(yearly_df, metrics, target_values=None, figsize=(15, 10)):
    """
    Визуализация трендов по годам
    """
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        if idx >= len(axes):
            break

        ax = axes[idx]

        median_col = f'{metric}_median'
        q25_col = f'{metric}_q25'
        q75_col = f'{metric}_q75'

        if median_col not in yearly_df.columns:
            continue

        years = yearly_df['year'].values
        medians = yearly_df[median_col].values
        q25 = yearly_df[q25_col].values
        q75 = yearly_df[q75_col].values

        # Основная линия
        ax.plot(years, medians, 'o-', linewidth=2, label='Медиана', color='blue')

        # Доверительный интервал
        ax.fill_between(years, q25, q75, alpha=0.2, color='blue', label='25-75 перцентили')

        # Целевая линия
        if target_values and metric in target_values:
            target = target_values[metric]
            ax.axhline(y=target, color='red', linestyle='--', linewidth=2,
                       label=f'Цель: {target}%')

        # Настройки графика
        ax.set_xlabel('Год')
        ax.set_ylabel('%' if 'pct' in metric else 'Значение')
        ax.set_title(metric.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

        # Добавляем значения на график
        for i, (year, val) in enumerate(zip(years, medians)):
            if i % 2 == 0:  # Каждый второй год для читаемости
                ax.annotate(f'{val:.1f}', (year, val),
                            textcoords="offset points", xytext=(0, 10),
                            ha='center', fontsize=8)

    # Скрываем пустые подграфики
    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig('yearly_trends_2012_2024.png', dpi=150, bbox_inches='tight')
    # plt.show()  # 🔴 ЗАКОММЕНТИРУЙ ЭТУ СТРОКУ

    return fig


def run_multiple_simulations(num_runs: int, params: Dict,
                             data_files: Tuple, parallel: bool = True,
                             show_progress: bool = True,
                             years_to_keep: Union[List[int], range, None] = None,
                             verbose: bool = False) -> List[Dict]:
    """
    Запускает multiple симуляций с оптимальным распределением ресурсов

    Args:
        num_runs: количество прогонов
        params: параметры симуляции
        data_files: кортеж с демографическими данными
        parallel: использовать ли параллельные вычисления
        show_progress: показывать ли прогресс-бар
        years_to_keep: список годов для сохранения (например, [2024] или range(2012, 2025))
                      Если None - сохраняются все годы (1950-2024)
    """
    birth_rate, death_rate, tfr_data, age_structure, fert_factors = data_files

    # 🔐 ПОТОКОБЕЗОПАСНАЯ ОЧИСТКА КЭША
    global _SIMULATION_CACHE
    with _CACHE_LOCK:
        _SIMULATION_CACHE.clear()
        print("🧹 Кэш очищен перед запуском симуляций")

    # Адаптивное количество воркеров
    if parallel:
        cpu_count = os.cpu_count() or 4
        if num_runs < cpu_count:
            max_workers = num_runs
        else:
            max_workers = max(1, cpu_count - 1) if num_runs > 20 else cpu_count
    else:
        max_workers = 1

    results = []

    # Преобразуем range в список для быстрой проверки
    if years_to_keep is not None:
        if isinstance(years_to_keep, range):
            years_set = set(years_to_keep)
        else:
            years_set = set(years_to_keep)
        print(f"📅 Будем сохранять только годы: {sorted(years_set)}")
    else:
        years_set = None
        print("📅 Сохраняем все годы (1950-2024)")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        run_func = partial(
            run_single_simulation_optimized,
            params=params,
            birth_rate_df=birth_rate,
            death_rate_df=death_rate,
            tfr_df=tfr_data,
            age_structure_df=age_structure,
            fertility_factors_df=fert_factors,
            verbose=False,
            use_cache=True
        )

        future_to_run = {
            executor.submit(run_func, run_id=i): i
            for i in range(num_runs)
        }

        iterator = as_completed(future_to_run)
        if show_progress:
            iterator = tqdm(iterator, total=num_runs,
                            desc="Running simulations",
                            unit="sim")

        for future in iterator:
            try:
                result = future.result(timeout=300)

                if result is None:
                    continue

                # Если результат - список (годовые данные)
                if isinstance(result, list):
                    for yearly_res in result:
                        if yearly_res and yearly_res.get('status') == 'success':
                            # 🔴 ФИЛЬТРАЦИЯ ПО ГОДАМ
                            if years_set is None:
                                # Сохраняем все годы
                                results.append(yearly_res)
                            else:
                                # Сохраняем только нужные годы
                                year = yearly_res.get('year')
                                if year in years_set:
                                    results.append(yearly_res)

                # Если результат - словарь (один прогон)
                elif isinstance(result, dict) and result.get('status') == 'success':
                    # Для обратной совместимости
                    if years_set is None:
                        results.append(result)
                    else:
                        year = result.get('year')
                        if year in years_set:
                            results.append(result)

            except Exception as e:
                print(f"Error in simulation: {e}")

    print(f"✅ Собрано {len(results)} записей")
    if years_set is not None:
        # Подсчитываем, сколько записей на каждый год
        years_count = {}
        for r in results:
            year = r.get('year')
            if year:
                years_count[year] = years_count.get(year, 0) + 1
        print(f"   По годам: {years_count}")

    return results

class Agent:
    def __init__(self, gender: str, age: int,
                 generation: int,
                 birth_year: int,  # Для связи с историческими данными
                 max_age_limit: int = 80, # Можно сделать общим параметром
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

        # Генетика (начальные значения)
        self.mefv_allele_1 = 'N'
        self.mefv_allele_2 = 'N'
        self.genotype_status = 'healthy'
        self.mutation_type = None

        # Семейные параметры
        self.father_id = father_id
        self.mother_id = mother_id
        self.partner_id = None
        self.children_ids = []
        self.last_birth_year = -100 #число, которое заведомо меньше самого первого года симуляции на величину

    def set_genotype(self, allele_1: str, allele_2: str):
        # Исключили E148Q согласно правилу
        valid_values = ['N', 'M694V', "V726A", "M680I", "R761H"]
        if allele_1 not in valid_values or allele_2 not in valid_values:
            raise ValueError("Invalid allele values. E148Q is excluded.")

        self.mefv_allele_1 = allele_1
        self.mefv_allele_2 = allele_2
        self.update_genotype_status()

    # теперь будем определять генетический статус на основе выбранных выше аллелей
    def update_genotype_status(self):
        alleles = [self.mefv_allele_1, self.mefv_allele_2] # смотрим какие аллели попались агенту
        mutant_count = sum(1 for allele in alleles if allele != "N") # считаем количество мутаций в (если они не N)

        if mutant_count == 0:
            self.genotype_status = "healthy"
            self.mutation_type = None # если нет мутаций N/N - то все ок и агент имеет статус здоров

        elif mutant_count == 1:
            self.genotype_status = "carrier"
            self.mutation_type = "heterozygous" # мутацией считается N/и любой другой мутантный аллель - тогда агент носитель

        else:
            self.genotype_status = "at_risk" # если оба аллея мутантные, то детализируем статус:

            if len(set(alleles)) == 1: # делаем из списка множество, чтобы понять, одинаковые аллели или нет.
                allele_type = alleles[0] # если два аллеля одинаковые достаточно взять только первую мутацию
                if allele_type == "M694V": # и если аллели одинаковые и M694V присвается статус гомозиготной мутации
                    self.mutation_type = "M694V_homozygous"
                else:
                    self.mutation_type = "other_homozygous" # если не M694V, то любая другая гомозиготная
            else:
                self.mutation_type = "compound_heterozygous" # если во множестве больше 1 аллели, то присваивается статус гетерозиготность

    def age_year(self, annual_death_prob: float):
        if not self.alive:
            return

        # 1. Увеличиваем возраст
        self.age += 1

        # 2. Проверяем биологический лимит
        if self.age > self.max_age_limit:
            self.alive = False
            return

        # 3. Расчет смертности для текущего возраста
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

        # 3. Манифестация FMF
        if self.clinical_status == 'asymptomatic' and self.mutation_type:
            prob = self._calculate_annual_onset_probability()

            # СНИЖАЕМ ВЕРОЯТНОСТЬ ПОСЛЕ 40 ЛЕТ
            if self.age >= 40:
                prob = prob * 0.1  # в 10 раз ниже

            if random.random() < prob:
                self.clinical_status = 'symptomatic'
                self.age_of_onset = self.age
                self._determine_disease_severity()

                # Доступ к лечению
                if self.age <= 18:
                    access_rate = 0.60
                else:
                    access_rate = 0.35

                if random.random() < access_rate:
                    self.on_colchicine = True

    def _calculate_annual_onset_probability(self):
        """Финальная версия"""
        if self.clinical_status != 'asymptomatic':
            return 0.0

        # Простые константы - только ОДИН параметр меняем!
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
        """Степень тяжести зависит от агрессивности мутаций."""
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

    # устанавливаем партнерские отношения
    def set_partner(self, partner: 'Agent'):
        self.partner_id = partner.id # присваиваем оригинальный номер идентификации, теперь агент(А) знает, кто его партнер
        partner.partner_id = self.id # присваиваем номер для парнера агента. теперь он (Б) тоже знает, кто его партнер (А)

    def add_child(self, child_id: str): # ссылаемся на родителей и даем уникальный номер детям
        if child_id not in self.children_ids: # проверяем нет ли номера ребенка в списке
            self.children_ids.append(child_id)

    def get_children_count(self): # возвращаем количество детей
        return len(self.children_ids)

    def can_get_pregnant(self, current_year: int, birth_cooldown: int) -> bool:
        """Проверяет возможность беременности."""
        if not self.alive or self.gender != 'female' or self.partner_id is None:
            return False

        # Биологический лимит (можно менять в зависимости от данных)
        if self.age < 18 or self.age > 45:
            return False

        # Проверка интервала между родами
        if current_year - self.last_birth_year < birth_cooldown:
            return False

        return True

    def get_mutation_count(self) -> int:
        """Возвращает количество мутантных аллелей (0, 1 или 2)."""
        return sum(1 for allele in [self.mefv_allele_1, self.mefv_allele_2] if allele != "N")

    def is_genetically_healthy(self) -> bool:
        return self.get_mutation_count() == 0

    def is_genetically_carrier(self) -> bool:
        return self.get_mutation_count() == 1

    def is_genetically_affected(self) -> bool:
        return self.get_mutation_count() == 2


    def get_short_info(self) -> str:
        return f"{self.id} ({self.gender[0]}{self.age}) Gen{self.generation} [{self.mefv_allele_1},{self.mefv_allele_2}]"

    def __str__(self):
        treatment_status = "лечится" if self.on_colchicine else "без лечения"
        return (f"Agent {self.id} ({self.gender}, {self.age} лет, Gen{self.generation}) - "
                f"{self.genotype_status} ({self.mefv_allele_1},{self.mefv_allele_2}) "
                f"{self.clinical_status} {treatment_status}")


def calculate_ci(count, total):
    """95% доверительный интервал для пропорции"""
    if total == 0:
        return 0
    p = count / total
    se = (p * (1 - p) / total) ** 0.5
    return 1.96 * se * 100

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
                 ethnic_assortativity: float = 0.85):

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

        # 6. Инициализация хранилищ и статистики
        self.agents: Dict[str, Agent] = {}
        self.population_history = []
        self.children_born = 0
        self.total_deaths = 0

        self.family_children_count = defaultdict(int)
        self.last_birth_year = defaultdict(lambda: -100)

        self.generation_stats = defaultdict(lambda: {
            'total': 0, 'alive': 0, 'healthy': 0, 'carrier': 0, 'affected': 0
        })

        self.birth_cooldown = 2

        # ВАЖНО: Создаем начальное население на основе структуры 1950 года
        self.initialize_founders_with_structure()

        self.inheritance_stats = {
            'allele_transmission': defaultdict(int),
            'parent_combinations': defaultdict(int),
            'mutation_pairs': defaultdict(int),
            'child_genotypes': defaultdict(int),
            'children_genotype_by_parent_combo': defaultdict(lambda: defaultdict(int)),
            'combo_children_genotypes': defaultdict(lambda: defaultdict(int))
        }

        # self.debug_stats = defaultdict(int)
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

            # Создаем агента с учетом года рождения
            agent = Agent(
                gender=gender,
                age=age,
                birth_year=1950 - age,  # Наш научный якорь
                generation=0,
                max_age_limit=self.max_age_limit
            )

            # Генетика (Харди-Вайнберг для основателей)
            a1, a2 = self._get_random_allele(), self._get_random_allele()
            agent.set_genotype(a1, a2)

            # "Проживаем" жизнь до 1950 года, чтобы проявились болезни
            real_age = agent.age
            agent.age = 0
            for year_tick in range(real_age):
                # Важно: здесь мы не учитываем смертность (она уже учтена структурой CSV),
                # только проявление болезни
                agent.age_year(annual_death_prob=0)  # 0, т.к. агент УЖЕ дожил до 1950

            agent.age = real_age  # Возвращаем возраст
            self.agents[agent.id] = agent

        # 3. Формирование семей (логика связей)
        self._form_initial_social_structure()

    def _get_family_key(self, p1, p2):
        # Гарантируем, что ключ будет одинаковым независимо от того, кто p1, а кто p2
        ids = sorted([p1.id, p2.id])
        return f"{ids[0]}_{ids[1]}"

    def _form_initial_social_structure(self):
        """Создает семейные связи для стартового населения 1950 года"""

        # 1. Подготовка списков
        males = [a for a in self.agents.values() if a.gender == 'male' and 18 <= a.age <= 50]
        females = [a for a in self.agents.values() if a.gender == 'female' and 18 <= a.age <= 45]
        children = [a for a in self.agents.values() if a.age < 18]

        random.shuffle(males)
        random.shuffle(females)

        # 2. Формируем пары
        n_pairs = min(len(males), len(females))
        active_couples = []

        for i in range(n_pairs):
            m, f = males[i], females[i]
            m.set_partner(f)
            active_couples.append((m, f))

            family_key = f"{m.id}_{f.id}"
            self.family_children_count[family_key] = 0

        # 3. Распределяем детей (с научной точки зрения - привязываем к биологическим матерям)
        # Сортируем детей от старших к младшим, чтобы легче подбирать родителей
        children.sort(key=lambda x: x.age, reverse=True)

        for child in children:
            # Ищем подходящую пару (где мать старше ребенка хотя бы на 18 лет)
            # и где разница в возрасте позволяет иметь такого ребенка
            suitable_couples = [
                (m, f) for m, f in active_couples
                if (f.age - child.age) >= 18 and (f.age - child.age) <= 45
            ]

            if suitable_couples:
                father, mother = random.choice(suitable_couples)

                # Устанавливаем связи
                child.father_id = father.id
                child.mother_id = mother.id
                father.add_child(child.id)
                mother.add_child(child.id)

                # Обновляем год последних родов матери, чтобы сработал birth_cooldown
                # Если сейчас 1950, а ребенку 5 лет, значит роды были в 1945
                birth_year_of_child = 1950 - child.age
                if birth_year_of_child > mother.last_birth_year:
                    mother.last_birth_year = birth_year_of_child

                family_key = f"{father.id}_{mother.id}"
                self.family_children_count[family_key] += 1

        # Удаляем временный атрибут, если он был создан в initialize_founders
        if hasattr(self, 'temp_adults'):
            del self.temp_adults

    # запускаем генерацию одного поколения
    def run_generation_with_calibration(self, verbose=False, run_id='unknown'):
        """Запуск с калибровкой по TFR, Birth Rate и Death Rate (1950-2024)"""

        yearly_results = []

        # 1. Инициализация (используем уже загруженные в __init__ данные)
        if not self.agents:
            self.initialize_founders_with_structure(verbose=verbose)

        # Инициализация результатов (оставляем твой словарь)
        self.calibration_results = {k: [] for k in [
            'year', 'target_tfr', 'model_births', 'model_deaths',
            'model_population', 'model_birth_rate', 'target_birth_rate',
            'model_death_rate', 'target_death_rate'
        ]}

        sum_of_asfr = sum(self.age_fertility_factors.values())
        calibration_factor = 0.23  # подобранное значение

        # 🔴 ОПТИМИЗАЦИЯ 1: Преобразуем DataFrame в словари для O(1) доступа
        if not hasattr(self, '_tfr_dict'):
            # Создаем словари только один раз
            self._tfr_dict = {}
            for idx in self.tfr_data.index:
                self._tfr_dict[idx] = self.tfr_data.loc[idx].iloc[0]

            self._death_dict = {}
            for idx in self.death_rate_data.index:
                self._death_dict[idx] = self.death_rate_data.loc[idx].iloc[0]

            self._birth_dict = {}
            for idx in self.birth_rate_data.index:
                self._birth_dict[idx] = self.birth_rate_data.loc[idx].iloc[0]

            # Значения по умолчанию (последние доступные)
            self._default_tfr = list(self._tfr_dict.values())[-1] if self._tfr_dict else 0
            self._default_dr = list(self._death_dict.values())[-1] if self._death_dict else 0
            self._default_br = list(self._birth_dict.values())[-1] if self._birth_dict else 0

        # Основной цикл
        for year in range(1950, 2025):
            self.year = year

            # 🔴 ОПТИМИЗАЦИЯ 2: Быстрый доступ через словари вместо .loc
            target_tfr = self._tfr_dict.get(year, self._default_tfr)
            target_dr = self._death_dict.get(year, self._default_dr)
            target_br = self._birth_dict.get(year, self._default_br)

            # УСТАНАВЛИВАЕМ ПАРАМЕТРЫ ДЛЯ ТЕКУЩЕГО ГОДА
            self.target_fertility_rate = target_tfr
            # Базовая вероятность рождения
            self.base_birth_prob = (target_tfr / sum_of_asfr) * calibration_factor
            # Вероятность смерти для каждого агента в этом году
            self.annual_death_prob = target_dr / 1000.0

            # 🔴 ОПТИМИЗАЦИЯ 3: Быстрый подсчет живых агентов
            alive_agents_ids = []
            for aid, a in self.agents.items():
                if a.alive:
                    alive_agents_ids.append(aid)

            start_population = len(alive_agents_ids)
            births_before = self.children_born
            deaths_before = self.total_deaths

            # ЗАПУСКАЕМ ГОД
            self._run_single_year_with_tracking(self.annual_death_prob)

            # 🔴 ОПТИМИЗАЦИЯ 4: Вызываем калибровку реже (уже есть, оставляем)
            # if year % 10 == 0 or year in [1975, 2000, 2024]:
            #     self.log_calibration_status()

            # ПОДСЧИТЫВАЕМ ИТОГИ
            births_this_year = self.children_born - births_before
            deaths_this_year = self.total_deaths - deaths_before

            # 🔴 ОПТИМИЗАЦИЯ 5: Быстрый подсчет живых после шага
            end_population = 0
            for a in self.agents.values():
                if a.alive:
                    end_population += 1

            # СЧИТАЕМ ПОКАЗАТЕЛИ (на 1000 человек)
            avg_pop = (start_population + end_population) / 2
            if avg_pop > 0:
                model_br = births_this_year / avg_pop * 1000
                model_dr = deaths_this_year / avg_pop * 1000
            else:
                model_br = 0
                model_dr = 0

            self._record_calibration(year, target_tfr, births_this_year, deaths_this_year,
                                     end_population, model_br, target_br, model_dr, target_dr)

            # if verbose and year % 10 == 0:
            #     print(f"Year {year}: Pop {end_population}, TFR {target_tfr:.2f}, "
            #           f"BR {model_br:.1f}({target_br:.1f}), DR {model_dr:.1f}({target_dr:.1f})")

            # Создаем запись для этого года
            year_result = collect_age_group_results_optimized(
                self, run_id, age_min=0, age_max=49
            )
            year_result['year'] = year

            self.method_name2(year_result, run_id, verbose)

            yearly_results.append(year_result)

        return yearly_results

    def method_name2(self, result, run_id, verbose):
        # Демографические показатели
        alive_agents = [a for a in self.agents.values() if a.alive]
        final_population = len(alive_agents)

        result.update({
            'status': 'success',
            'total_births': self.children_born,
            'total_deaths': self.total_deaths,
            'final_population': final_population,
            'total_agents_ever': len(self.agents),
            'simulation_years': 2024 - 1950 + 1,
            'run_id': run_id
        })
        # Статистика наследования
        child_genotypes = self.inheritance_stats.get('child_genotypes', {})
        if isinstance(child_genotypes, dict):
            result.update({
                'healthy_births': child_genotypes.get('healthy', 0),
                'carrier_births': child_genotypes.get('carrier', 0),
                'affected_births': child_genotypes.get('affected', 0),
            })
        else:
            result.update({
                'healthy_births': 0,
                'carrier_births': 0,
                'affected_births': 0,
            })
        # result.update({
        #     'total_births_checked': self.debug_stats.get('total_births_checked', 0),
        #     'genetic_anomalies': self.debug_stats.get('impossible_healthy', 0),
        # })
        # Распределение аллелей
        allele_analysis = self.get_allele_frequency_analysis()
        for k, v in allele_analysis['current'].items():
            result[f'allele_freq_{k}'] = v
        for k, v in allele_analysis['drift'].items():
            result[f'allele_drift_{k}'] = v
        # Передача аллелей от родителей
        trans = self.inheritance_stats.get('allele_transmission', {})
        if isinstance(trans, dict):
            father_items = {k: v for k, v in trans.items() if isinstance(k, str) and k.startswith('father_')}
            mother_items = {k: v for k, v in trans.items() if isinstance(k, str) and k.startswith('mother_')}

            for k, v in father_items.items():
                result[f'transmission_{k}'] = v
            for k, v in mother_items.items():
                result[f'transmission_{k}'] = v

            result['father_alleles_total'] = sum(father_items.values()) if father_items else 0
            result['mother_alleles_total'] = sum(mother_items.values()) if mother_items else 0
        else:
            result['father_alleles_total'] = 0
            result['mother_alleles_total'] = 0
        # Топ комбинации родителей
        parent_stats = self.inheritance_stats.get('parent_combinations', {})
        if isinstance(parent_stats, dict):
            result['unique_parent_combinations'] = len(parent_stats)
            if parent_stats:
                result['most_common_combo_count'] = max(parent_stats.values())
            else:
                result['most_common_combo_count'] = 0
        else:
            result['unique_parent_combinations'] = 0
            result['most_common_combo_count'] = 0
        # Менделевское расщепление
        combo_stats = self.inheritance_stats.get('combo_children_genotypes', {})
        if combo_stats and isinstance(combo_stats, dict):
            total_mendel_healthy = 0
            total_mendel_carrier = 0
            total_mendel_affected = 0

            for genotypes in combo_stats.values():
                if isinstance(genotypes, dict):
                    total_mendel_healthy += genotypes.get('healthy', 0)
                    total_mendel_carrier += genotypes.get('carrier', 0)
                    total_mendel_affected += genotypes.get('affected', 0)

            total_mendel_children = total_mendel_healthy + total_mendel_carrier + total_mendel_affected

            if total_mendel_children > 0:
                result.update({
                    'mendel_total_healthy': total_mendel_healthy,
                    'mendel_total_carrier': total_mendel_carrier,
                    'mendel_total_affected': total_mendel_affected,
                    'mendel_healthy_pct': (total_mendel_healthy / total_mendel_children * 100),
                    'mendel_carrier_pct': (total_mendel_carrier / total_mendel_children * 100),
                    'mendel_affected_pct': (total_mendel_affected / total_mendel_children * 100),
                })
        # Калибровочные метрики
        if hasattr(self, 'calibration_results') and self.calibration_results.get('year'):
            cal = self.calibration_results
            model_birth = np.array(cal.get('model_birth_rate', []))
            target_birth = np.array(cal.get('target_birth_rate', []))
            model_death = np.array(cal.get('model_death_rate', []))
            target_death = np.array(cal.get('target_death_rate', []))

            result.update({
                'avg_model_birth_rate': np.mean(model_birth) if len(model_birth) > 0 else 0,
                'avg_target_birth_rate': np.mean(target_birth) if len(target_birth) > 0 else 0,
                'avg_model_death_rate': np.mean(model_death) if len(model_death) > 0 else 0,
                'avg_target_death_rate': np.mean(target_death) if len(target_death) > 0 else 0,
            })

            if len(model_birth) > 0 and len(target_birth) > 0:
                result['rmse_birth_rate'] = np.sqrt(np.mean((model_birth - target_birth) ** 2))
            else:
                result['rmse_birth_rate'] = 0

            if len(model_death) > 0 and len(target_death) > 0:
                result['rmse_death_rate'] = np.sqrt(np.mean((model_death - target_death) ** 2))
            else:
                result['rmse_death_rate'] = 0

            model_pop = cal.get('model_population', [])
            result['final_population_calibration'] = model_pop[-1] if model_pop else 0
        # Клиническая статистика за последний год
        if hasattr(self, 'population_history') and self.population_history and len(self.population_history) > 0:
            last_stats = self.population_history[-1]

            if isinstance(last_stats, dict):
                clinical_stats = last_stats.get('clinical_stats', {})
                if isinstance(clinical_stats, dict):
                    result.update({
                        'final_symptomatic': clinical_stats.get('symptomatic', 0),
                        'final_asymptomatic': clinical_stats.get('asymptomatic', 0),
                        'final_severe': clinical_stats.get('severe', 0),
                        'final_moderate': clinical_stats.get('moderate', 0),
                        'final_mild': clinical_stats.get('mild', 0),
                    })

                treatment_stats = last_stats.get('treatment_stats', {})
                if isinstance(treatment_stats, dict):
                    result.update({
                        'final_on_colchicine': treatment_stats.get('on_colchicine', 0),
                        'final_no_colchicine': treatment_stats.get('no_colchicine', 0),
                    })

                # Распределение мутаций среди больных
                mutation_dist = last_stats.get('mutation_distribution', {})
                if isinstance(mutation_dist, dict):
                    for k, v in mutation_dist.items():
                        result[f'final_mutation_{k}'] = v
        # Статистика по поколениям
        if hasattr(self, 'generation_stats') and self.generation_stats:
            total_alive_all_gens = 0
            total_symptomatic_all_gens = 0

            for gen, stats in self.generation_stats.items():
                if isinstance(stats, dict):
                    alive = stats.get('alive', 0)
                    symptomatic = stats.get('symptomatic', 0)

                    if alive > 0:
                        result[f'gen_{gen}_alive'] = alive
                        result[f'gen_{gen}_symptomatic'] = symptomatic
                        total_alive_all_gens += alive
                        total_symptomatic_all_gens += symptomatic

            result['total_alive_all_generations'] = total_alive_all_gens
            result['total_symptomatic_all_generations'] = total_symptomatic_all_gens

            if alive_agents:
                all_affected = [a for a in self.agents.values()
                                if a.clinical_status == 'symptomatic']
                if all_affected:
                    late_onset = sum(1 for a in all_affected
                                     if getattr(a, 'age_of_onset', 0) >= 40)
                    result['late_onset_abs'] = int(late_onset)

                    late_onset_pct = (late_onset / len(all_affected) * 100)
                    result['late_onset_pct'] = float(late_onset_pct)
        # Возрастная структура всей популяции
        if alive_agents:
            ages = np.array([a.age for a in alive_agents])
            genders = np.array([a.gender for a in alive_agents])

            result.update({
                'mean_age': float(np.mean(ages)),
                'median_age': float(np.median(ages)),
                'max_age': int(np.max(ages)),
                'male_count': int(np.sum(genders == 'male')),
                'female_count': int(np.sum(genders == 'female')),
            })
        if hasattr(self, 'debug_genotype_penetrance') and verbose:
            self.debug_genotype_penetrance()
        # Метрики пенетрантности
        alive_agents = [a for a in self.agents.values() if a.alive]
        for mtype in ['M694V_homozygous', 'compound_heterozygous', 'heterozygous', 'other_homozygous']:
            type_agents = [a for a in alive_agents if a.mutation_type == mtype]
            if type_agents:
                affected = sum(1 for a in type_agents if a.clinical_status == 'symptomatic')
                result[f'penetrance_{mtype}'] = affected / len(type_agents) * 100

    def _record_calibration(self, year, target_tfr, births, deaths, pop,
                            model_br, target_br, model_dr, target_dr):
        """
        Записывает результаты симуляции за год для последующего анализа и графиков.
        """
        self.calibration_results['year'].append(year)
        self.calibration_results['target_tfr'].append(target_tfr)
        self.calibration_results['model_births'].append(births)
        self.calibration_results['model_deaths'].append(deaths)
        self.calibration_results['model_population'].append(pop)
        self.calibration_results['model_birth_rate'].append(model_br)
        self.calibration_results['target_birth_rate'].append(target_br)
        self.calibration_results['model_death_rate'].append(model_dr)
        self.calibration_results['target_death_rate'].append(target_dr)

    def _run_single_year_with_tracking(self, death_prob: float):
        """
        Ежегодный цикл.
        death_prob — вероятность смерти из CSV для текущего года.
        """
        # 1. Получаем список ID всех живых на начало года
        # (Работа по ID безопаснее, если список будет меняться)
        living_ids = [aid for aid, a in self.agents.items() if a.alive]

        # 2. Старение и Смертность
        for aid in living_ids:
            agent = self.agents[aid]

            # Передаем вероятность смерти в метод старения
            # (Нам нужно будет чуть подправить Agent.age_year)
            was_alive = agent.alive
            agent.age_year(annual_death_prob=death_prob)

            if was_alive and not agent.alive:
                self.total_deaths += 1

        # 3. Рождаемость
        # Сначала создаем новые пары из тех, кто повзрослел или овдовел
        self._form_new_partnerships()

        # Затем запускаем процесс рождения детей
        # Этот метод будет использовать self.base_birth_prob и self.age_fertility_factors
        self._birth_process()

        # 4. Обновление общей статистики модели
        self._record_population_stats()

    def _form_new_partnerships(self):
        """Формирует новые пары среди одиноких агентов."""

        # 1. Сначала фильтруем всех кандидатов один раз
        single_males = [a for a in self.agents.values()
                        if a.alive and a.gender == 'male' and a.partner_id is None
                        and 18 <= a.age <= 60]

        single_females = [a for a in self.agents.values()
                          if a.alive and a.gender == 'female' and a.partner_id is None
                          and 18 <= a.age <= 45]

        if not single_males or not single_females:
            return

        # Оптимизация 1: Создаем множества для быстрой проверки занятости
        paired_females = set()

        # Оптимизация 2: Предварительно группируем женщин по этничности
        females_by_ethnicity = {}
        females_by_id = {}  # для быстрого доступа по id

        for f in single_females:
            ethnicity = getattr(f, 'ethnicity', 'Armenian')
            if ethnicity not in females_by_ethnicity:
                females_by_ethnicity[ethnicity] = []
            females_by_ethnicity[ethnicity].append(f)
            females_by_id[f.id] = f

        random.shuffle(single_males)

        # Оптимизация 3: Предварительно вычисляем вероятности для всех мужчин
        for male in single_males:
            # Если мужчина уже нашел пару в этом цикле
            if male.partner_id is not None:
                continue

            # Проверка вероятности создания пары
            partnership_prob = max(0.1, 0.3 - (male.age - 25) * 0.01)
            if random.random() > partnership_prob:
                continue

            male_ethnicity = getattr(male, 'ethnicity', 'Armenian')

            # Оптимизация 4: Быстрое получение кандидаток по этничности
            same_ethnicity_females = []
            other_ethnicity_females = []

            # Используем предварительно сгруппированные списки
            if male_ethnicity in females_by_ethnicity:
                same_ethnicity_females = [f for f in females_by_ethnicity[male_ethnicity]
                                          if f.id not in paired_females]

            # Собираем женщин других этничностей
            for eth, females in females_by_ethnicity.items():
                if eth != male_ethnicity:
                    other_ethnicity_females.extend([f for f in females if f.id not in paired_females])

            # Выбор пула кандидаток
            if random.random() < self.ethnic_assortativity:
                candidate_females = same_ethnicity_females if same_ethnicity_females else other_ethnicity_females
            else:
                candidate_females = other_ethnicity_females if other_ethnicity_females else same_ethnicity_females

            if not candidate_females:
                continue

            # Оптимизация 5: Ограничиваем размер выборки для производительности
            sample_size = min(10, len(candidate_females))
            potential_brides = random.sample(candidate_females, sample_size)

            # Цикл по невестам
            for female in potential_brides:
                # Проверка на родство (быстрая)
                is_sibling = False
                has_common_father = (male.father_id is not None and male.father_id == female.father_id)
                has_common_mother = (male.mother_id is not None and male.mother_id == female.mother_id)

                if has_common_father or has_common_mother:
                    is_sibling = True

                if not is_sibling:
                    male.set_partner(female)
                    paired_females.add(female.id)  # Отмечаем как занятую

                    family_key = f"{male.id}_{female.id}"
                    self.family_children_count[family_key] = 0

                    # Оптимизация 6: Выходим из цикла, но продолжаем с другими мужчинами
                    break

    def _get_fertility_factor(self, age: int) -> float:
        if age < 18 or age > 45:
            return 0.0

        # Оптимизация: Создаем numpy array только один раз
        if not hasattr(self, '_fertility_factors_array'):
            import numpy as np
            # Сортируем для бинарного поиска
            sorted_factors = sorted(
                [(min_age, max_age, factor) for (min_age, max_age), factor in self.age_fertility_factors.items()]
            )
            self._fertility_factors_array = np.array(sorted_factors,
                                                     dtype=[('min', int), ('max', int), ('factor', float)])

        # Бинарный поиск (уже есть)
        factors = self._fertility_factors_array
        left, right = 0, len(factors) - 1

        while left <= right:
            mid = (left + right) // 2
            if age < factors[mid]['min']:
                right = mid - 1
            elif age > factors[mid]['max']:
                left = mid + 1
            else:
                return factors[mid]['factor']
        return 0.0

    def _birth_process(self):
        # Оптимизация 1: Предварительно отфильтровываем потенциальных матерей ОДИН раз
        # и создаем словарь для быстрого доступа к отцам
        potential_mothers = []
        fathers_cache = {}  # Кэш для отцов, чтобы не делать self.agents.get() многократно

        for agent in self.agents.values():
            if agent.alive and agent.gender == 'female' and agent.can_get_pregnant(self.year, self.birth_cooldown):
                potential_mothers.append(agent)
                # Кэшируем отца, если есть
                if agent.partner_id and agent.partner_id in self.agents:
                    fathers_cache[agent.id] = self.agents[agent.partner_id]

        # Оптимизация 2: Предварительно вычисляем children_penalty для всех возможных значений
        children_penalty_map = {0: 1.0, 1: 1.0, 2: 0.8, 3: 0.4}

        # Оптимизация 3: Предварительно вычисляем медицинские факторы
        untreated_penalties = {
            "M694V_homozygous": 0.15,
            "compound_heterozygous": 0.35,
            "heterozygous": 0.60,
            "other_homozygous": 0.25
        }

        asymptomatic_penalties = {
            "M694V_homozygous": 0.70,
            "compound_heterozygous": 0.80,
            "other_homozygous": 0.85
        }

        # Оптимизация 4: Основной цикл с минимальными проверками
        for mother in potential_mothers:
            # Получаем отца из кэша
            father = fathers_cache.get(mother.id)
            if not father or not father.alive:
                continue

            family_key = f"{father.id}_{mother.id}"
            current_children_count = self.family_children_count.get(family_key, 0)

            # --- БАЗОВАЯ ВЕРОЯТНОСТЬ ---
            age_factor = self._get_fertility_factor(mother.age)
            if age_factor <= 0:
                continue

            # Социально-биологический лимит (используем map вместо dict.get каждый раз)
            children_penalty = children_penalty_map.get(current_children_count, 0.1)

            # --- МЕДИЦИНСКИЙ ФАКТОР (оптимизированный) ---
            health_factor = 1.0

            clinical_status = mother.clinical_status
            mutation_type = mother.mutation_type

            if clinical_status == 'symptomatic':
                if mother.on_colchicine:
                    health_factor = 0.95
                else:
                    # Используем get с дефолтом 1.0
                    health_factor = untreated_penalties.get(mutation_type, 1.0)
            elif clinical_status == 'asymptomatic' and mutation_type in asymptomatic_penalties:
                # Проверяем наличие в словаре перед обращением
                health_factor = asymptomatic_penalties[mutation_type]

            # --- ОГРАНИЧЕНИЯ (clip через min/max) ---
            if health_factor < 0.1:
                health_factor = 0.1
            elif health_factor > 1.0:
                health_factor = 1.0

            # --- ИТОГОВАЯ ВЕРОЯТНОСТЬ ---
            birth_prob = self.base_birth_prob * age_factor * children_penalty * health_factor

            # Clip probability
            if birth_prob < 0.0:
                birth_prob = 0.0
            elif birth_prob > 1.0:
                birth_prob = 1.0

            # Оптимизация 5: Быстрое сравнение с random
            if random.random() < birth_prob:
                self._create_child_with_detailed_tracking(mother, father)
                self.family_children_count[family_key] = current_children_count + 1
                mother.last_birth_year = self.year
                self.children_born += 1
    def print_fertility_report(self):
        """Выводит отчет о рождаемости в симуляции"""
        print("\n" + "=" * 60)
        print("📊 ОТЧЕТ О РОЖДАЕМОСТИ")
        print("=" * 60)

        # Текущий TFR
        actual_tfr = self.calculate_actual_fertility_rate()
        print(f"Целевой TFR:           {self.target_fertility_rate:.2f}")
        print(f"Фактический TFR:        {actual_tfr:.2f}")
        print(f"Отклонение:             {(actual_tfr - self.target_fertility_rate):+.2f}")

        # Доля детей в популяции
        living_agents = [a for a in self.agents.values() if a.alive]
        if living_agents:
            total = len(living_agents)
            children_0_14 = len([a for a in living_agents if 0 <= a.age <= 14])
            youth_15_29 = len([a for a in living_agents if 15 <= a.age <= 29])
            adults_30_49 = len([a for a in living_agents if 30 <= a.age <= 49])
            seniors_50_plus = len([a for a in living_agents if a.age >= 50])

            print(f"\nВозрастная структура:")
            print(f"  Дети 0-14:      {children_0_14:>6} ({children_0_14 / total * 100:>5.1f}%)")
            print(f"  Молодежь 15-29: {youth_15_29:>6} ({youth_15_29 / total * 100:>5.1f}%)")
            print(f"  Взрослые 30-49: {adults_30_49:>6} ({adults_30_49 / total * 100:>5.1f}%)")
            print(f"  Пожилые 50+:    {seniors_50_plus:>6} ({seniors_50_plus / total * 100:>5.1f}%)")
            print(
                f"  Всего 0-49:     {children_0_14 + youth_15_29 + adults_30_49:>6} ({(children_0_14 + youth_15_29 + adults_30_49) / total * 100:>5.1f}%)")

        # Распределение рождений по возрастам матерей (если есть данные)
        if hasattr(self, 'last_birth_year') and self.children_born > 0:
            print(f"\nСтатистика рождений:")
            print(f"  Всего рождений: {self.children_born}")
            print(
                f"  Среднее число детей на семью: {self.children_born / len(self.family_children_count) if self.family_children_count else 0:.2f}")

    def calculate_actual_fertility_rate(self, birth_year_range: tuple = None) -> float:
        """
        Рассчитывает TFR для женщин, завершивших репродуктивный период.
        birth_year_range: можно указать (1960, 1970), чтобы проверить TFR только этого поколения.
        """
        completed_women = [
            a for a in self.agents.values()
            if a.gender == 'female' and a.age >= 45
        ]

        # Если указан диапазон, фильтруем женщин по году их рождения
        if birth_year_range:
            start_y, end_y = birth_year_range
            completed_women = [w for w in completed_women if start_y <= w.birth_year <= end_y]

        if not completed_women:
            return 0.0

        total_children = sum(len(w.children_ids) for w in completed_women)

        return total_children / len(completed_women)

# определяем генотип ребенка
    def _create_child_with_detailed_tracking(self, parent1: Agent, parent2: Agent):
        # 0. Безопасная проверка вместо агрессивного assert
        if parent1.gender == parent2.gender:
            return None

        # 1. Определяем роли
        if parent1.gender == 'male':
            father, mother = parent1, parent2
        else:
            father, mother = parent2, parent1

        # 2. МЕНДЕЛЕВСКОЕ НАСЛЕДОВАНИЕ
        father_allele = random.choice([father.mefv_allele_1, father.mefv_allele_2])
        mother_allele = random.choice([mother.mefv_allele_1, mother.mefv_allele_2])

        alleles_sorted = sorted([father_allele, mother_allele])
        child_genotype_str = f"{alleles_sorted[0]}/{alleles_sorted[1]}"

        # 3. ОБНОВЛЕНИЕ СТАТИСТИКИ
        # Убедись, что этот метод инициализирован!
        self._update_inheritance_stats(father, mother, father_allele, mother_allele, child_genotype_str)

        # 4. СОЗДАНИЕ НОВОГО АГЕНТА
        child = Agent(
            gender=random.choice(['male', 'female']),
            age=0,
            generation=max(father.generation, mother.generation) + 1,
            birth_year=self.year,
            max_age_limit=self.max_age_limit,
            father_id=father.id,
            mother_id=mother.id
        )

        # 5. Установка генотипа и связей
        child.set_genotype(alleles_sorted[0], alleles_sorted[1])

        self.agents[child.id] = child
        father.add_child(child.id)
        mother.add_child(child.id)

        return child  # Возвращаем ребенка для удобства
 # для статистики

    def _update_inheritance_stats(self, father, mother, f_allele, m_allele, child_genotype_str):
        """
        Обновляет детальную генетическую статистику и проверяет соблюдение законов Менделя.
        """
        # 0. ГАРАНТИРУЕМ, ЧТО СТРУКТУРА СУЩЕСТВУЕТ (если забыли прописать в __init__)
        if not hasattr(self, 'inheritance_stats') or not self.inheritance_stats:
            self.inheritance_stats = {
                'allele_transmission': defaultdict(int),  # подсчет того, как часто передается каждый конкретный аллель
                'parent_combinations': defaultdict(int),  # подсчет частоты комбинаций генотипов родителей
                'child_genotypes': defaultdict(int),  # подсчет частоты генотипов детей
                'mutation_pairs': defaultdict(int),
                # подсчет специфических пар мутаций, которые были унаследованы детьми
                'expected_vs_actual': defaultdict(int),
                # сколько раз ожидался определенный результат наследования и сколько раз он фактически произошел в симуляции
                'children_genotype_by_parent_combo': defaultdict(lambda: defaultdict(int)),
                # отслеживаем какой именно генотип наследуется от родителей
                'combo_children_genotypes': defaultdict(lambda: defaultdict(int))
                # новая структура для детального анализа менделевского наследования
            }

        # if not hasattr(self, 'debug_stats'):
        #     self.debug_stats = defaultdict(int)
        #
        # # Увеличиваем счетчик проверенных рождений
        # self.debug_stats['total_births_checked'] += 1

        # 1. Фиксируем передачу аллелей
        self.inheritance_stats['allele_transmission'][f"father_{f_allele}"] += 1
        self.inheritance_stats['allele_transmission'][f"mother_{m_allele}"] += 1

        # 2. Создаем ключ комбинации родителей
        p1_gen = sorted([father.mefv_allele_1, father.mefv_allele_2])
        p2_gen = sorted([mother.mefv_allele_1, mother.mefv_allele_2])
        parent_combo = f"{p1_gen[0]}/{p1_gen[1]} x {p2_gen[0]}/{p2_gen[1]}"
        self.inheritance_stats['parent_combinations'][parent_combo] += 1

        # 3. Фиксируем генотип ребенка
        self.inheritance_stats['mutation_pairs'][child_genotype_str] += 1

        # Генетический статус ребенка
        child_mutations = sum(1 for a in [f_allele, m_allele] if a != "N")
        child_status = "affected" if child_mutations == 2 else ("carrier" if child_mutations == 1 else "healthy")

        self.inheritance_stats['child_genotypes'][child_status] += 1
        self.inheritance_stats['children_genotype_by_parent_combo'][parent_combo][child_status] += 1

        # 4. Сравнение с теоретическим ожиданием
        theory_key = self._get_combo_key_for_theory(father, mother)
        self.inheritance_stats['combo_children_genotypes'][theory_key][child_status] += 1

        # ========== ЛОГИКА КОНТРОЛЯ КАЧЕСТВА (Генетическая полиция) ==========

        f_mut_count = sum(1 for a in [father.mefv_allele_1, father.mefv_allele_2] if a != "N")
        m_mut_count = sum(1 for a in [mother.mefv_allele_1, mother.mefv_allele_2] if a != "N")

        parent_is_homozygote = (f_mut_count == 2 or m_mut_count == 2)

        # Проверка на биологическую невозможность (Закон расщепления Менделя)
        # if parent_is_homozygote and child_status == "healthy":
        #     self.debug_stats['impossible_healthy'] += 1
        #     print(f"\n🔴 ГЕНЕТИЧЕСКАЯ АНОМАЛИЯ #{self.debug_stats['impossible_healthy']} (Год: {self.year})")
        #     print(
        #         f"  Отец: {father.mefv_allele_1}/{father.mefv_allele_2} | Мать: {mother.mefv_allele_1}/{mother.mefv_allele_2}")
        #     print(f"  Результат: {child_genotype_str} (N/N невозможен, если есть родитель гомозигота M/M)")
        #
        # # Специальные счетчики ошибок для отчета
        # if (f_mut_count == 0 and m_mut_count == 2) or (f_mut_count == 2 and m_mut_count == 0):
        #     if child_status == "healthy":
        #         self.debug_stats['err_affected_x_healthy'] += 1
        #
        # if (f_mut_count == 1 and m_mut_count == 2) or (f_mut_count == 2 and m_mut_count == 1):
        #     if child_status == "healthy":
        #         self.debug_stats['err_affected_x_carrier'] += 1

    def _get_combo_key_for_theory(self, father: Agent, mother: Agent) -> str:
        """
        Определяет ключ теоретического наследования (Мендель).
        Использует количество мутаций (0, 1, 2) для классификации пары.
        """

        def get_genetic_type(agent):
            # Считаем мутации (любой аллель, не равный "N")
            mut_count = sum(1 for a in [agent.mefv_allele_1, agent.mefv_allele_2] if a != "N")

            if mut_count == 0: return 'healthy'
            if mut_count == 1: return 'carrier'
            return 'affected'  # 2 мутации

        f_type = get_genetic_type(father)
        m_type = get_genetic_type(mother)

        # Сортировка гарантирует, что 'carrier_healthy' и 'healthy_carrier'
        # попадут в одну корзину для статистики
        statuses = sorted([f_type, m_type])
        return f"{statuses[0]}_{statuses[1]}"

    def _calculate_expected_inheritance(self, father, mother) -> str:
        """
        Анализирует генотипы родителей и предсказывает исход по законам Менделя.
        """

        f_alleles = [father.mefv_allele_1, father.mefv_allele_2]
        m_alleles = [mother.mefv_allele_1, mother.mefv_allele_2]

        # 1. Строим решетку Пеннета (все 4 комбинации)
        outcomes = []
        for fa in f_alleles:
            for ma in m_alleles:
                gen = sorted([fa, ma])
                outcomes.append(f"{gen[0]}/{gen[1]}")

        # 2. Считаем частоту каждой комбинации
        counts = Counter(outcomes)

        # 3. Формируем красивую строку ожидания
        # Сортируем ключи, чтобы отчет всегда выглядел одинаково
        parts = []
        for genotype in sorted(counts.keys()):
            prob = (counts[genotype] / 4) * 100
            parts.append(f"{genotype}({int(prob)}%)")

        return "Expected: " + ", ".join(parts)
    # собираем ключевые данные о текущем состоянии живой популяции
    def _record_population_stats(self):
        """Снимает метрики популяции в текущем году и сохраняет в историю."""

        # Получаем всех агентов
        agents_list = list(self.agents.values())

        if not agents_list:
            stats = {
                'year': self.year,
                'total_population': 0,
                'genotype_stats': {},
                'clinical_stats': {},
                'treatment_stats': {},
                'mutation_distribution': {},
                'generation_counts': {}
            }
            if not hasattr(self, 'population_history'):
                self.population_history = []
            self.population_history.append(stats)
            return

        # Создаем массивы numpy для всех нужных атрибутов
        n_agents = len(agents_list)

        # Массивы для всех агентов
        generations = np.array([a.generation for a in agents_list])
        alive_mask = np.array([a.alive for a in agents_list])

        # Для живых агентов создаем отдельные массивы
        alive_indices = np.where(alive_mask)[0]
        n_alive = len(alive_indices)

        if n_alive == 0:
            stats = {
                'year': self.year,
                'total_population': 0,
                'genotype_stats': {},
                'clinical_stats': {},
                'treatment_stats': {},
                'mutation_distribution': {},
                'generation_counts': {}
            }
            if not hasattr(self, 'population_history'):
                self.population_history = []
            self.population_history.append(stats)
            return

        # Массивы только для живых агентов
        alive_generations = generations[alive_indices]
        alive_genotypes = np.array([agents_list[i].genotype_status for i in alive_indices])
        alive_clinical = np.array([agents_list[i].clinical_status for i in alive_indices])
        alive_treatment = np.array(['on_colchicine' if agents_list[i].on_colchicine else 'no_colchicine'
                                    for i in alive_indices])
        alive_mutation = np.array([agents_list[i].mutation_type if agents_list[i].mutation_type else 'none'
                                   for i in alive_indices])
        alive_severity = np.array([agents_list[i].disease_severity if agents_list[i].disease_severity else 'none'
                                   for i in alive_indices])

        # 1. total_alive
        total_alive = n_alive

        # 2. genotype_stats - через unique
        unique_genotypes, genotype_counts = np.unique(alive_genotypes, return_counts=True)
        genotype_stats = dict(zip(unique_genotypes, genotype_counts))

        # 3. clinical_stats - сначала symptomatic/asymptomatic, потом severity
        unique_clinical, clinical_counts = np.unique(alive_clinical, return_counts=True)
        clinical_stats = dict(zip(unique_clinical, clinical_counts))

        # Добавляем severity для symptomatic
        symptomatic_mask = alive_clinical == 'symptomatic'
        if np.any(symptomatic_mask):
            severity_values = alive_severity[symptomatic_mask]
            # Убираем 'none'
            valid_severity = severity_values[severity_values != 'none']
            if len(valid_severity) > 0:
                unique_severity, severity_counts = np.unique(valid_severity, return_counts=True)
                for sev, count in zip(unique_severity, severity_counts):
                    clinical_stats[sev] = count

        # 4. treatment_stats
        unique_treatment, treatment_counts = np.unique(alive_treatment, return_counts=True)
        treatment_stats = dict(zip(unique_treatment, treatment_counts))

        # 5. mutation_distribution (только для symptomatic)
        mutation_distribution = {}
        if np.any(symptomatic_mask):
            mutation_values = alive_mutation[symptomatic_mask]
            # Убираем 'none'
            valid_mutations = mutation_values[mutation_values != 'none']
            if len(valid_mutations) > 0:
                unique_mutations, mutation_counts = np.unique(valid_mutations, return_counts=True)
                mutation_distribution = dict(zip(unique_mutations, mutation_counts))

        # 6. generation_counts
        current_gen_counts = {}
        unique_generations = np.unique(alive_generations)

        for gen in unique_generations:
            gen_mask = alive_generations == gen
            gen_alive = np.sum(gen_mask)

            # Total для поколения (считаем всех агентов этого поколения, включая мертвых)
            total_mask = generations == gen
            gen_total = np.sum(total_mask)

            # Symptomatic для этого поколения
            gen_symptomatic = np.sum(gen_mask & symptomatic_mask)

            current_gen_counts[int(gen)] = {
                'total': int(gen_total),
                'alive': int(gen_alive),
                'symptomatic': int(gen_symptomatic)
            }

        # Формируем stats точно как в оригинале
        stats = {
            'year': self.year,
            'total_population': total_alive,
            'genotype_stats': genotype_stats,
            'clinical_stats': clinical_stats,
            'treatment_stats': treatment_stats,
            'mutation_distribution': mutation_distribution,
            'generation_counts': current_gen_counts
        }

        # Сохраняем в историю
        if not hasattr(self, 'population_history'):
            self.population_history = []

        self.population_history.append(stats)
        self.generation_stats = current_gen_counts


    def _print_initial_stats(self):

        print("\n" + "=" * 70)
        print("ИСХОДНЫЕ ДАННЫЕ (ПОКОЛЕНИЕ 0)")
        print("=" * 70)

        initial = self.population_history[0]
        total = initial['total_population']
        g_stats = initial['genotype_stats']

        if total == 0 or not g_stats:
            print("Нет данных")
            return

        # Создаем DataFrame
        df = pd.DataFrame({
            'Генотип': list(g_stats.keys()),
            'N': list(g_stats.values())
        })

        # Векторные вычисления
        df['%'] = df['N'] / total * 100
        df['p'] = df['N'] / total
        df['se'] = np.sqrt(df['p'] * (1 - df['p']) / total)

        # 95% ДИ
        z = 1.96
        df['ДИ_ниж'] = np.maximum(0, df['p'] - z * df['se']) * 100
        df['ДИ_верх'] = np.minimum(1, df['p'] + z * df['se']) * 100

        # p-value
        expected = total / len(df)
        df['chi2'] = ((df['N'] - expected) ** 2) / expected
        df['p_val'] = 1 - stats.chi2.cdf(df['chi2'], df=1)

        # Форматирование
        print(f"\n{'Генотип':<10} {'N':>6} {'%':>6} {'95% ДИ':>14} {'p-value':>10}")
        print("-" * 48)

        for _, row in df.iterrows():
            p_str = ("<0.001***" if row['p_val'] < 0.001 else
                     f"{row['p_val']:.3f}**" if row['p_val'] < 0.01 else
                     f"{row['p_val']:.3f}*" if row['p_val'] < 0.05 else
                     f"{row['p_val']:.3f}")

            print(f"{row['Генотип'].capitalize():<10} "
                  f"{row['N']:>6} "
                  f"{row['%']:>5.1f}% "
                  f"[{row['ДИ_ниж']:>4.1f}-{row['ДИ_верх']:<4.1f}] "
                  f"{p_str:>10}")

        print("-" * 48)
        print(f"{'ИТОГО':<10} {total:>6} {'100%':>6} {'-' * 12} {'-' * 10}")

        # Общий тест
        if len(df) > 1:
            chi2, p = stats.chisquare(df['N'])
            print(f"\nχ²={chi2:.2f}, p={p:.4f}")

        print("=" * 70)

    def _print_generation_breakdown(self):
         # Создаем DataFrame из данных
        df = pd.DataFrame([
            {'gen': g, **self.generation_stats[g]}
            for g in sorted(self.generation_stats.keys())
        ])

        # Фильтруем и рассчитываем
        df = df[df['alive'] > 0].copy()
        if df.empty:
            print("Нет активных поколений")
            return

        df['healthy'] = df['alive'] - df['symptomatic']
        df['sick_%'] = (df['symptomatic'] / df['alive'] * 100).round(1)

        print(f"\n{' ПОКОЛЕНИЯ ':=^50}")
        print(df[['gen', 'alive', 'symptomatic', 'healthy', 'sick_%']]
              .to_string(index=False,
                         header=['Покол', 'Всего', 'Больных', 'Здоровых', '%'],
                         formatters={'%': '{:.1f}%'.format}))

        print(f"{'=' * 50}")
        print(f"ИТОГО: {df['alive'].sum():>3} чел. | "
              f"Больных: {df['symptomatic'].sum():>3} "
              f"({(df['symptomatic'].sum() / df['alive'].sum() * 100):.1f}%)")

    def _print_calibration_report(self):
        if not hasattr(self, 'calibration_results') or not self.calibration_results.get('year'):
            print("\n🛑 Ошибка: Нет данных для калибровочного отчета.")
            return

        # Создаем DataFrame из результатов
        df = pd.DataFrame({
            'year': self.calibration_results['year'],
            'br_model': self.calibration_results['model_birth_rate'],
            'br_target': self.calibration_results['target_birth_rate'],
            'dr_model': self.calibration_results['model_death_rate'],
            'dr_target': self.calibration_results['target_death_rate'],
            'population': self.calibration_results['model_population']
        })

        print("\n" + "═" * 70)
        print(f"{'📊 ВАЛИДАЦИЯ ДЕМОГРАФИЧЕСКИХ ПОКАЗАТЕЛЕЙ (1950-2024)':^70}")
        print("═" * 70)

        # Контрольные точки
        check_points = [1960, 1975, 1990, 2005, 2020, 2024]
        df_check = df[df['year'].isin(check_points)].copy()

        # Рассчитываем отклонения и статус
        df_check['diff'] = np.abs(df_check['br_model'] - df_check['br_target'])
        df_check['status'] = '✅ OK'
        df_check.loc[(df_check['diff'] >= 3.0) & (df_check['year'] < 1980), 'status'] = '🔄 Адаптация'
        df_check.loc[(df_check['diff'] >= 3.0) & (df_check['year'] >= 1980), 'status'] = '⚠️ Отклонение'

        # Форматируем вывод
        print(df_check[['year', 'br_model', 'br_target', 'dr_model', 'dr_target', 'status']]
              .to_string(index=False,
                         header=['Год', 'BR (Мод)', 'BR (Цель)', 'DR (Мод)', 'DR (Цель)', 'Статус'],
                         formatters={
                             'BR (Мод)': '{:.2f}'.format,
                             'BR (Цель)': '{:.2f}'.format,
                             'DR (Мод)': '{:.2f}'.format,
                             'DR (Цель)': '{:.2f}'.format
                         }))

        print("-" * 70)

        # Статистика точности
        modern_df = df[df['year'] >= 1990]
        if not modern_df.empty:
            rmse = np.sqrt(np.mean((modern_df['br_model'] - modern_df['br_target']) ** 2))
            mae = np.mean(np.abs(modern_df['br_model'] - modern_df['br_target']))
            max_error = np.max(np.abs(modern_df['br_model'] - modern_df['br_target']))

            print(f"📊 Статистика точности (с 1990 г.):")
            print(f"   RMSE = {rmse:.2f}  |  MAE = {mae:.2f}  |  Max error = {max_error:.2f}")
            print(f"   {'✅ Норма' if rmse < 5.0 else '⚠️ Выше нормы'}")

        # Итоговая популяция
        final_pop = df['population'].iloc[-1]
        pop_growth = ((df['population'].iloc[-1] / df['population'].iloc[0] - 1) * 100)

        print(f"\n👥 ИТОГОВАЯ ПОПУЛЯЦИЯ (2024): {int(final_pop):,} чел.")
        print(f"📈 Рост с 1950 г.: {pop_growth:.1f}%")
        print("═" * 70)

    def _print_age_structure(self):

        ages = [a.age for a in self.agents.values() if a.alive]

        if not ages:
            print(f"\n📊 Год {self.year}: Нет живых агентов")
            return

        df = pd.DataFrame(ages, columns=['age'])
        n = len(ages)

        print(f"\n{'=' * 60}")
        print(f"📊 ВОЗРАСТНАЯ СТРУКТУРА (Год: {self.year})")
        print(f"{'=' * 60}")

        # Основные метрики
        stats_dict = {
            'Средний': df['age'].mean(),
            'Медиана': df['age'].median(),
            'Стд откл': df['age'].std(),
            'Мин/Макс': f"{df['age'].min()}/{df['age'].max()}"
        }

        print("Основные показатели:")
        for k, v in stats_dict.items():
            print(f"  {k:>10}: {v:>8.1f}" if isinstance(v, float) else f"  {k:>10}: {v:>8}")

        # Возрастные группы
        bins = [0, 18, 46, 120]
        labels = ['Дети (0-17)', 'Репродуктивный (18-45)', 'Старшее (46+)']

        df['group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
        groups = df['group'].value_counts().reset_index()
        groups.columns = ['Группа', 'Кол-во']
        groups['%'] = groups['Кол-во'] / n * 100

        # Доверительные интервалы
        z = 1.96
        p = groups['Кол-во'] / n
        se = np.sqrt(p * (1 - p) / n)
        groups['ДИ_ниж'] = (p - z * se).clip(0, 1) * 100
        groups['ДИ_верх'] = (p + z * se).clip(0, 1) * 100

        # p-value
        expected = n / len(groups)
        groups['p'] = 1 - stats.chi2.cdf(((groups['Кол-во'] - expected) ** 2) / expected, 1)

        print(f"\n{'Группа':<20} {'N':>6} {'%':>6} {'95% ДИ':>16} {'p':>8}")
        print('-' * 60)

        for _, row in groups.iterrows():
            p_str = "<0.001" if row['p'] < 0.001 else f"{row['p']:.3f}"
            print(f"{row['Группа']:<20} "
                  f"{row['Кол-во']:>6} "
                  f"{row['%']:>5.1f}% "
                  f"[{row['ДИ_ниж']:>4.0f}-{row['ДИ_верх']:<4.0f}] "
                  f"{p_str:>8}")

        # Итоговый тест
        chi2, p_global = stats.chisquare(groups['Кол-во'])
        print(f"\nХи-квадрат тест: χ²={chi2:.2f}, p={p_global:.4f}")
        print('=' * 60)

    def _print_final_summary(self):
        print("\n" + "═" * 65)
        print(" 🏁 ИТОГОВЫЙ ОТЧЕТ СИМУЛЯЦИИ (ВАЛИДАЦИЯ ГЕНЕТИКИ)")
        print("═" * 65)

        # 1. Общие цифры
        all_affected = [a for a in self.agents.values() if a.clinical_status == 'symptomatic']
        total_aff = len(all_affected)
        total_agents = len(self.agents)

        print(f"🔹 Создано агентов за всю историю: {total_agents:,}")
        print(f"🔹 Всего случаев FMF (симптоматических): {total_aff:,}")

        if total_agents > 0:
            prevalence = (total_aff / total_agents) * 100
            print(f"🔹 Общая историческая распространенность: {prevalence:.2f}%")

        if total_aff > 0:
            print("\n📊 Распределение по мутациям (сравнение с данными статьи):")
            m_stats = defaultdict(int)
            late_onset = 0

            # Добавим детальную диагностику
            genotype_details = defaultdict(int)
            compound_details = defaultdict(int)

            for a in all_affected:
                m_stats[a.mutation_type] += 1

                # ДИАГНОСТИКА: смотрим реальные аллели
                alleles = sorted([a.mefv_allele_1, a.mefv_allele_2])
                genotype = f"{alleles[0]}/{alleles[1]}"
                genotype_details[genotype] += 1

                # Для компаундов - конкретная комбинация
                if a.mutation_type == "compound_heterozygous":
                    compound_details[genotype] += 1

                # Считаем возраст начала болезни
                if hasattr(a, 'age_of_onset') and a.age_of_onset and a.age_of_onset >= 40:
                    late_onset += 1

            targets = {
                "M694V_homozygous": 11.12,
                "compound_heterozygous": 58.26,
                "heterozygous": 25.33,
                "other_homozygous": 2.0,
                "late_onset_pct": 3.40  # Добавить, если используешь
            }

            print(f"{'-' * 85}")
            print(f"{'Тип мутации (генотипа)':<25} | {'Модель %':<20} | {'Статья %':<10}")
            print(f"{'-' * 85}")

            # ДИАГНОСТИКА: печатаем реальное распределение генотипов
            # print("\n🔍 ДЕТАЛЬНЫЙ РАЗБОР ГЕНОТИПОВ:")
            total = len(all_affected)
            for genotype, count in sorted(genotype_details.items(), key=lambda x: -x[1]):
                pct = (count / total) * 100
            #     print(f"  {genotype}: {count} ({pct:.2f}%)")

            # print("\n🔍 КОМПАУНД-ГЕТЕРОЗИГОТЫ ДЕТАЛЬНО:")
            for genotype, count in sorted(compound_details.items(), key=lambda x: -x[1]):
                pct = (count / total) * 100
            #     print(f"  {genotype}: {count} ({pct:.2f}%)")

            # Основная таблица
            for mtype in ["M694V_homozygous", "compound_heterozygous", "other_homozygous", "heterozygous"]:
                count = m_stats.get(mtype, 0)
                pct = (count / total) * 100
                target = targets.get(mtype, 0)
                status = "✅" if abs(pct - target) <= 5 else "⚠️" if abs(pct - target) <= 10 else "❌"
                print(f"{mtype:<25} | {pct:>6.2f}% ± {calculate_ci(count, total):>4.2f}% | {target:>6.2f}% {status:>3}")


            # for m_type, target_val in targets.items():
            #     current_val = (m_stats[m_type] / total_aff * 100)
            #     p = current_val / 100  # преобразуем в пропорцию
            #     se = sqrt(p * (1 - p) / total_aff)  # стандартная ошибка
            #     ci = 1.96 * se * 100  # 95% доверительный интервал в процентах
            #
            #     diff = current_val - target_val
            #     # Добавим маркер отклонения (если разница > 5%, ставим предупреждение)
            #     marker = "⚠️" if abs(diff) > 5 else "✅"
            #
            #     print(f"{m_type:<25} | {current_val:>9.2f}% ± {ci:>5.2f}% | {target_val:>9.2f}% {marker}")
            #
            # print(f"{'-' * 85}")

            late_onset_pct = (late_onset / total_aff * 100)
            # Расчет ДИ для позднего начала
            p_late = late_onset_pct / 100
            se_late = sqrt(p_late * (1 - p_late) / total_aff)
            ci_late = 1.96 * se_late * 100

            print(f"{'Позднее начало (>=40 лет)':<25} | {late_onset_pct:>9.2f}% ± {ci_late:>5.2f}% | 3.40%")

        # 2. ОТЛАДОЧНАЯ СТАТИСТИКА (Генетическая чистота)
        # if hasattr(self, 'debug_stats') and self.debug_stats.get('total_births_checked', 0) > 0:
        #     print(f"\n" + "⚙️  ОТЛАДОЧНАЯ СТАТИСТИКА НАСЛЕДОВАНИЯ")
        #     print(f"{'-' * 65}")
        #     total_births = self.debug_stats.get('total_births_checked', 0)
        #     impossible = self.debug_stats.get('impossible_healthy', 0)
        #
        #     print(f"Всего рождений проверено: {total_births:,}")
        #     print(f"Генетических аномалий:    {impossible}")
        #
        #     if total_births > 0:
        #         anomaly_percent = (impossible / total_births) * 100
        #         if impossible == 0:
        #             print(f"🧪 Чистота наследования: 100% (Законы Менделя соблюдены)")
        #         else:
        #             print(f"🧪 Ошибки наследования: {anomaly_percent:.4f}%")
        #
        # print("\n" + "=" * 65)
        # # self.debug_genotype_penetrance()

        print("\n" + "═" * 65)
        self.log_calibration_status()

        print("═" * 65 + "\n")

    def get_allele_frequency_analysis(self) -> Dict:
        """
        Детальный анализ стабильности частот аллелей в популяции.
        Сравнивает текущее состояние со стартовыми параметрами 1950 года.
        """
        allele_count = defaultdict(int)
        living_agents = [a for a in self.agents.values() if a.alive]
        total_living = len(living_agents)
        total_alleles = total_living * 2

        # 1. Считаем все аллели у живых
        for agent in living_agents:
            allele_count[agent.mefv_allele_1] += 1
            allele_count[agent.mefv_allele_2] += 1

        current_frequencies = {}
        drift = {}

        if total_alleles > 0:
            # Берем уникальный набор всех аллелей (и начальных, и тех, что есть сейчас)
            all_known_alleles = set(self.mutation_frequencies.keys()) | set(allele_count.keys())

            for allele in all_known_alleles:
                count = allele_count.get(allele, 0)
                freq = count / total_alleles
                current_frequencies[allele] = freq

                # Считаем отклонение от начальных частот (если аллеля не было, считаем за 0)
                initial_freq = self.mutation_frequencies.get(allele, 0)
                drift[allele] = freq - initial_freq

        return {
            'year': self.year,
            'total_living': total_living,
            'initial': self.mutation_frequencies,
            'current': current_frequencies,
            'drift': drift,
            'counts': dict(allele_count)
        }

    def print_allele_report(self):
        """Красивый вывод анализа частот аллелей с индикацией дрейфа."""
        analysis = self.get_allele_frequency_analysis()

        print("\n" + "═" * 60)
        print(f"🧬 АНАЛИЗ ЧАСТОТ АЛЛЕЛЕЙ (Год {analysis['year']})")
        print(f"   Живых агентов: {analysis['total_living']:,} (Всего аллелей: {analysis['total_living'] * 2:,})")
        print("═" * 60)
        print(f"{'Аллель':<10} | {'Старт':<10} | {'Сейчас':<10} | {'Дрейф':<10} | {'Тренд'}")
        print("-" * 60)

        # Берем все уникальные аллели из обоих наборов (старт и текущие)
        all_alleles = sorted(set(analysis['initial'].keys()) | set(analysis['current'].keys()))

        for allele in all_alleles:
            start = analysis['initial'].get(allele, 0.0)
            now = analysis['current'].get(allele, 0.0)
            drift = analysis['drift'].get(allele, 0.0)

            # Определяем иконку тренда
            if abs(drift) < 0.0001:
                trend = " стабильно"
            elif drift > 0:
                trend = "↑ рост"
            else:
                trend = "↓ убыль"

            print(f"{allele:<10} | {start:>10.4f} | {now:>10.4f} | {drift:>+10.4f} | {trend}")

        print("-" * 60)

    def log_calibration_status(self):
        """
        Логирует статус калибровки распределения мутаций среди больных
        Сравнивает с целевыми значениями из статьи
        """
        # Данные из статьи (цель)
        targets = {
            "M694V_homozygous": 11.12,
            "compound_heterozygous": 58.26,
            "heterozygous": 25.33,
            "other_homozygous": 2.0
        }

        # Считаем только тех, кто заболел (наша виртуальная больница)
        sick_agents = [a for a in self.agents.values()
                       if a.clinical_status == 'symptomatic' and a.alive]
        total_sick = len(sick_agents)

        if total_sick < 50:
            # Было: print(f"⚠️ Год {self.year}: Недостаточно больных...")
            # Стало: ничего не печатаем или только при verbose=True
            if hasattr(self, 'verbose') and self.verbose:
                print(f"⚠️ Год {self.year}: Недостаточно больных (n={total_sick})")
            return

        print(f"\n{'=' * 70}")
        print(f"📊 КАЛИБРОВКА ГЕНОТИПОВ (Год {self.year}, Больных: {total_sick})")
        print(f"{'-' * 70}")
        print(f"{'Тип мутации':<25} | {'Модель %':>10} | {'Цель %':>8} | {'Разница':>10} | {'Статус':>8}")
        print(f"{'-' * 70}")

        for m_type, target_val in targets.items():
            count = len([a for a in sick_agents if a.mutation_type == m_type])
            current_pct = (count / total_sick) * 100
            diff = current_pct - target_val

            # Визуальный индикатор
            if abs(diff) < 2:
                status = "✅ OK"
            elif abs(diff) < 5:
                status = "⚠️ ADJUST"
            else:
                status = "❌ NEED FIX"

            # Доверительный интервал для пропорции
            ci = 1.96 * ((current_pct / 100 * (1 - current_pct / 100) / total_sick) ** 0.5) * 100

            print(f"{m_type:<25} | {current_pct:>6.2f}% ±{ci:>4.1f}% | "
                  f"{target_val:>6.2f}% | {diff:>+6.2f}% | {status:>8}")

        print(f"{'=' * 70}")

        # Дополнительная диагностика для компаундов
        compound_agents = [a for a in sick_agents if a.mutation_type == "compound_heterozygous"]
        if compound_agents:
            # print("\n🔬 Детальный анализ компаунд-гетерозигот:")
            compound_compositions = defaultdict(int)
            for a in compound_agents:
                alleles = sorted([a.mefv_allele_1, a.mefv_allele_2])
                comp_key = f"{alleles[0]}/{alleles[1]}"
                compound_compositions[comp_key] += 1

            for comp, count in sorted(compound_compositions.items(), key=lambda x: -x[1]):
                pct = (count / len(compound_agents)) * 100
                # print(f"  {comp}: {count} ({pct:.1f}%)")

    def _print_detailed_inheritance_stats(self):
        print(f"\n{'=' * 70}")
        print(f" ДЕТАЛЬНАЯ СТАТИСТИКА НАСЛЕДОВАНИЯ (ПОКОЛЕНИЯ 1+)")
        print(f"{'=' * 70}")

        # self.children_born должен быть int
        total_children = int(self.children_born)
        print(f"Всего детей рождено в симуляции: {total_children:,}")

        if total_children == 0:
            print("Данные отсутствуют: дети еще не рождались.")
            return

        # 1. Генетический статус новорожденных (Используем 'child_genotypes')
        print(f"\nГенетический статус новорожденных (Генотипы):")
        # Важно: берем 'child_genotypes', так как там лежат простые числа (int)
        child_genotypes = self.inheritance_stats['mutation_pairs']
        for gen, count in sorted(child_genotypes.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / self.children_born) * 100
            print(f"  {gen:<15}: {count:>5} ({percentage:>5.2f}%)")

        # 2. Топ комбинаций родителей
        print(f"\n Самые частые союзы родителей:")
        parent_stats = self.inheritance_stats.get('parent_combinations', {})
        for combo, count in sorted(parent_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
            percentage = (count / self.children_born) * 100
            print(f"  {combo:<25}: {count:>5} детей ({percentage:>5.2f}%)")

        # 3. Баланс передачи аллелей
        print(f"\n Распределение передачи аллелей:")
        trans = self.inheritance_stats.get('allele_transmission', {})

        # Фильтруем ключи, приводим значения к int для линтера
        fathers = {k: int(v) for k, v in trans.items() if k.startswith('father_')}
        mothers = {k: int(v) for k, v in trans.items() if k.startswith('mother_')}

        total_f = sum(fathers.values())
        total_m = sum(mothers.values())

        print(f"  ОТЦЫ (передано {total_f} аллелей):")
        for k, v in sorted(fathers.items()):
            perc = (v / total_f * 100) if total_f > 0 else 0
            label = k.replace('father_', '')
            print(f"    {label:<10}: {v:>5} ({perc:>5.1f}%)")

        print(f"  МАТЕРИ (передано {total_m} аллелей):")
        for k, v in sorted(mothers.items()):
            perc = (v / total_m * 100) if total_m > 0 else 0
            label = k.replace('mother_', '')
            print(f"    {label:<10}: {v:>5} ({perc:>5.1f}%)")

        # 4. Дополнительно: Анализ по конкретным парам
        print(f"\n🧬 Расщепление признаков по типам союзов (Пример):")
        combo_stats = self.inheritance_stats.get('combo_children_genotypes', {})

        # Берем только первую пару для примера
        for combo, genotypes_dict in list(combo_stats.items())[:3]:
            print(f"  Для пары {combo}:")
            # Преобразуем defaultdict в обычный dict для вывода
            for g, c in genotypes_dict.items():
                # Убеждаемся, что c - это число
                count_val = int(c) if hasattr(c, '__int__') else c
                print(f"    -> {g}: {count_val}")

        # 5. Вызов теоретического анализа
        print(f"\n{'=' * 70}")
        if hasattr(self, '_print_theoretical_analysis'):
            # Исправляем вызов - убираем дефис, добавляем скобки
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

        print(f"\n{'🧬 ТИП СКРЕЩИВАНИЯ':<40} | {'СТАТУС':<10} | {'ТЕОРИЯ':<8} | {'ФАКТ':<8} | {'ОТКЛОН.'}")
        print("═" * 85)

        def _get_abbr(s):
            return {'healthy': 'N/N', 'carrier': 'N/M', 'affected': 'M/M'}.get(s, s)

        for combo_key, data in theoretical_expectations.items():
            title = data['title']
            expected_ratios = data['ratios']

            actual_results = self.inheritance_stats['combo_children_genotypes'].get(combo_key, {})
            total_children = sum(actual_results.values())

            if total_children == 0:
                print(f"{title:<40} | {'НЕТ ДАННЫХ (дети не рождались)':<41}")
                print("-" * 85)
                continue

            first_line = True
            for status in ['healthy', 'carrier', 'affected']:
                exp = expected_ratios[status]
                count = actual_results.get(status, 0)
                act = count / total_children

                # Считаем разницу (абсолютное отклонение)
                diff = act - exp
                # Маркер точности: если отклонение > 3% при большой выборке, это повод проверить код
                marker = "!" if (abs(diff) > 0.03 and total_children > 100) else ""

                display_title = f"{title} (n={total_children})" if first_line else ""

                print(
                    f"{display_title:<40} | {_get_abbr(status):<10} | {exp:>7.1%} | {act:>7.1%} | {diff:>+7.1f}% {marker}")
                first_line = False
            print("-" * 85)


def get_cache_key(params: Dict[str, Any], birth_rate_df, death_rate_df, run_id=None) -> int:
    """Генерирует уникальный ключ кэша"""
    param_str = str(sorted(params.items()))

    if run_id is not None:
        param_str += f"_run_{run_id}"

    data_str = (
            str(birth_rate_df.head(10).values.tolist()) +
            str(death_rate_df.head(10).values.tolist())
    )

    combined = param_str + data_str
    return int(hashlib.md5(combined.encode()).hexdigest(), 16) % (2 ** 32)

_SIMULATION_CACHE: Dict[int, Dict[str, Any]] = {}
_MAX_CACHE_SIZE = 128
_CACHE_LOCK = Lock()


def run_single_simulation_optimized(run_id: Union[int, str],
                                    params: Dict[str, Any],
                                    birth_rate_df,
                                    death_rate_df,
                                    tfr_df,
                                    age_structure_df,
                                    fertility_factors_df,
                                    verbose: bool = False,
                                    use_cache: bool = True) -> List[Dict[str, Any]]:  # 🔴 ВОЗВРАЩАЕТ СПИСОК!

    global _SIMULATION_CACHE, _CACHE_LOCK

    # Устанавливаем уникальный seed
    if isinstance(run_id, int):
        random.seed(run_id)
        np.random.seed(run_id)
    else:
        seed_val = int(hashlib.md5(str(run_id).encode()).hexdigest(), 16) % (2**32)
        random.seed(seed_val)
        np.random.seed(seed_val)

    cache_key = None
    if use_cache:
        cache_key = get_cache_key(params, birth_rate_df, death_rate_df, run_id)
        with _CACHE_LOCK:
            if cache_key in _SIMULATION_CACHE:
                if verbose:
                    print(f"🔄 Использую кэшированный результат для {run_id}")
                # Возвращаем копию кэшированного списка
                cached_results = _SIMULATION_CACHE[cache_key].copy()
                for res in cached_results:
                    res['run_id'] = run_id
                    res['cached'] = True
                return cached_results

    try:
        # 1. СОЗДАЕМ СИМУЛЯЦИЮ ТОЛЬКО ОДИН РАЗ!
        sim = GenerationSimulation(
            birth_rate_df=birth_rate_df,
            death_rate_df=death_rate_df,
            tfr_df=tfr_df,
            age_structure_df=age_structure_df,
            fertility_factors_df=fertility_factors_df,
            initial_population_size=params.get('initial_population_size', 10000),
            max_age_limit=params.get('max_age_limit', 80),
            mutation_frequencies=params.get('mutation_frequencies', None),
            ethnic_assortativity=params.get('ethnic_assortativity', 0.85)
        )

        # 2. ЗАПУСКАЕМ СИМУЛЯЦИЮ
        yearly_results = sim.run_generation_with_calibration(verbose=False, run_id=run_id)

        # 5. СОХРАНЯЕМ ВЕСЬ СПИСОК В КЭШ
        if use_cache and cache_key is not None:
            with _CACHE_LOCK:
                if len(_SIMULATION_CACHE) < _MAX_CACHE_SIZE:
                    _SIMULATION_CACHE[cache_key] = yearly_results.copy()

        if verbose:
            print(f"✅ Симуляция {run_id} завершена, собрано {len(yearly_results)} годовых записей")

        return yearly_results  # 🔴 ВОЗВРАЩАЕМ СПИСОК!

    except Exception as e:
        print(f"❌ Ошибка в прогоне {run_id}: {e}")
        import traceback
        traceback.print_exc()
        return [{'run_id': run_id, 'status': 'error', 'error_message': str(e)}]




def collect_age_group_results_optimized(simulation, run_id: str, age_min: int = 0, age_max: int = 49):
    """
    Собирает статистику по возрастной группе для валидации.
    """
    # 1. Фильтруем живых
    agents_dict = getattr(simulation, 'agents', {})
    current_year = getattr(simulation, 'year', 2024)

    target_agents = [
        a for a in agents_dict.values()
        if a.alive and age_min <= a.age <= age_max
    ]

    total = len(target_agents)
    if total == 0:
        return {
            'run_id': run_id,
            'year': current_year,  # Добавляем год!
            'total_agents': 0,
            'prevalence': 0
        }

    # 2. Используем numpy для быстрых вычислений
    ages = np.array([a.age for a in target_agents])
    genotypes = np.array([a.genotype_status for a in target_agents])
    mutation_types = np.array([getattr(a, 'mutation_type', None) for a in target_agents])

    # 3. Возрастные группы
    age_counts = {
        'age_0_14_abs': np.sum(ages <= 14),
        'age_15_29_abs': np.sum((ages > 14) & (ages <= 29)),
        'age_30_49_abs': np.sum((ages > 29) & (ages <= 49))
    }

    # 4. Генетика
    affected_mask = np.array([a.clinical_status == 'symptomatic' for a in target_agents])
    total_aff = np.sum(affected_mask)

    if total_aff > 0:
        affected_mutations = mutation_types[affected_mask]

        mutation_counts = {
            'm694v_homo_abs': np.sum(affected_mutations == 'M694V_homozygous'),
            'compound_abs': np.sum(affected_mutations == 'compound_heterozygous'),
            'other_homo_abs': np.sum(affected_mutations == 'other_homozygous'),
            'hetero_abs': np.sum(affected_mutations == 'heterozygous')
        }
    else:
        mutation_counts = {'m694v_homo_abs': 0, 'compound_abs': 0,
                           'other_homo_abs': 0, 'hetero_abs': 0}

    # 5. Формирование результата
    return {
        'run_id': run_id,
        'year': current_year,  # Добавляем год!
        'total_agents': total,
        'total_affected': total_aff,
        'prevalence_pct': (total_aff / total * 100) if total > 0 else 0,
        **age_counts,
        'age_0_14_pct': (age_counts['age_0_14_abs'] / total * 100) if total > 0 else 0,
        'age_15_29_pct': (age_counts['age_15_29_abs'] / total * 100) if total > 0 else 0,
        'age_30_49_pct': (age_counts['age_30_49_abs'] / total * 100) if total > 0 else 0,
        **mutation_counts,
        'm694v_homo_in_affected_pct': (mutation_counts['m694v_homo_abs'] / total_aff * 100) if total_aff > 0 else 0,
        'compound_in_affected_pct': (mutation_counts['compound_abs'] / total_aff * 100) if total_aff > 0 else 0,
        'other_homo_in_affected_pct': (mutation_counts['other_homo_abs'] / total_aff * 100) if total_aff > 0 else 0,
        'hetero_in_affected_pct': (mutation_counts['hetero_abs'] / total_aff * 100) if total_aff > 0 else 0
    }


class SensitivityAnalysis:
    """
    Класс для проведения анализа чувствительности агент-ориентированной модели FMF
    Оптимизированная версия
    """

    def __init__(self, base_params, birth_rate_df, death_rate_df, tfr_df,
                 age_structure_df, fertility_factors_df):
        self.base_params = base_params
        self.birth_rate = birth_rate_df
        self.death_rate = death_rate_df
        self.tfr_data = tfr_df
        self.age_structure = age_structure_df
        self.fert_factors = fertility_factors_df

        # Кэш для результатов
        self._simulation_cache = {}
        self._baseline_cache = None

        # Ключевые метрики
        self.key_metrics = [
            'prevalence_pct',
            'm694v_homo_in_affected_pct',
            'compound_in_affected_pct',
            'hetero_in_affected_pct',
            'final_population',
            'avg_model_birth_rate',
            'avg_model_death_rate',
        ]

    def run_one_factor_at_a_time(self, param_ranges, num_runs_per_scenario=3,
                                 max_workers=None, use_caching=True):
        """
        OFAT (One Factor At Time) анализ - оптимизированная версия
        """
        scenarios = []

        # Базовый сценарий
        if use_caching and self._baseline_cache is not None:
            base_result = self._baseline_cache
        else:
            base_result = self._run_multiple_scenarios({}, num_runs_per_scenario,
                                                       scenario_name='baseline')
            if use_caching:
                self._baseline_cache = base_result

        # Собираем все сценарии
        all_scenario_params = []

        for param_name, values in param_ranges.items():
            base_value = self.base_params.get(param_name, 1.0)
            for value in values:
                if value == base_value:
                    continue
                params = {param_name: value}
                scenario_name = f"{param_name}={value}"

                for run in range(num_runs_per_scenario):
                    all_scenario_params.append({
                        'params': params,
                        'run': run,
                        'scenario_name': scenario_name,
                        'param_name': param_name,
                        'param_value': value
                    })

        # Параллельный запуск
        results = base_result.copy() if base_result else []

        if all_scenario_params:
            if max_workers is None:
                max_workers = min(32, len(all_scenario_params))

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_params = {
                    executor.submit(
                        self._run_single_simulation_wrapper,
                        params_data['params'],
                        f"{params_data['scenario_name']}_{params_data['run']}"
                    ): params_data
                    for params_data in all_scenario_params
                }

                for future in tqdm(as_completed(future_to_params),
                                   total=len(all_scenario_params),
                                   desc="OFAT прогоны"):
                    params_data = future_to_params[future]
                    try:
                        yearly_results = future.result(timeout=120)  # 🔴 Получаем СПИСОК!

                        if yearly_results and isinstance(yearly_results, list):
                            for year_result in yearly_results:
                                if year_result and year_result.get('status') == 'success':
                                    year_result['scenario'] = params_data['scenario_name']
                                    year_result['run'] = params_data['run']
                                    year_result[f'param_{params_data["param_name"]}'] = params_data['param_value']
                                    results.append(year_result)
                        elif yearly_results and yearly_results.get('status') == 'success':
                            # Для обратной совместимости
                            yearly_results['scenario'] = params_data['scenario_name']
                            yearly_results['run'] = params_data['run']
                            yearly_results[f'param_{params_data["param_name"]}'] = params_data['param_value']
                            results.append(yearly_results)

                    except Exception as e:
                        print(f"Ошибка в сценарии {params_data['scenario_name']}: {e}")
                        continue

        return results

    def _run_single_simulation_wrapper(self, params, run_id):
        """Обертка для запуска симуляции с кэшированием"""
        sim_params = self.base_params.copy()
        sim_params.update(params)

        # Теперь возвращает список, берем только последний год для анализа чувствительности
        yearly_results = run_single_simulation_optimized(
            run_id=run_id,
            params=sim_params,
            birth_rate_df=self.birth_rate,
            death_rate_df=self.death_rate,
            tfr_df=self.tfr_data,
            age_structure_df=self.age_structure,
            fertility_factors_df=self.fert_factors,
            verbose=False,
            use_cache=True
        )

        # Для анализа чувствительности берем только 2024 год
        if yearly_results and isinstance(yearly_results, list):
            # Ищем запись за 2024 год
            for result in yearly_results:
                if result.get('year') == 2024 and result.get('status') == 'success':
                    return result
            # Если нет 2024, берем последний
            return yearly_results[-1] if yearly_results else None

        return yearly_results

    def _run_multiple_scenarios(self, params, num_runs, scenario_name):
        """Запускает несколько прогонов для одного сценария"""
        results = []
        sim_params = self.base_params.copy()
        sim_params.update(params)

        run_func = partial(
            run_single_simulation_optimized,
            params=sim_params,
            birth_rate_df=self.birth_rate,
            death_rate_df=self.death_rate,
            tfr_df=self.tfr_data,
            age_structure_df=self.age_structure,
            fertility_factors_df=self.fert_factors,
            verbose=False,
            use_cache=True
        )

        for run in range(num_runs):
            yearly_results = run_func(run_id=f"{scenario_name}_{run}")  # 🔴 Теперь получаем СПИСОК!

            # Обрабатываем каждый год отдельно
            if yearly_results and isinstance(yearly_results, list):
                for year_result in yearly_results:
                    if year_result and year_result.get('status') == 'success':
                        year_result['scenario'] = scenario_name
                        year_result['run'] = run

                        for param_name, param_value in params.items():
                            year_result[f'param_{param_name}'] = param_value

                        results.append(year_result)
            elif yearly_results and yearly_results.get('status') == 'success':
                # Для обратной совместимости (если вдруг вернулся словарь)
                yearly_results['scenario'] = scenario_name
                yearly_results['run'] = run
                for param_name, param_value in params.items():
                    yearly_results[f'param_{param_name}'] = param_value
                results.append(yearly_results)

        return results

    def calculate_sensitivity_indices(self, results_df, metrics=None):
        """
        Рассчитывает индексы чувствительности - векторизованная версия
        """
        if metrics is None:
            metrics = self.key_metrics

        param_cols = [col for col in results_df.columns if col.startswith('param_')]

        if not param_cols:
            return pd.DataFrame()

        param_names = [col.replace('param_', '') for col in param_cols]

        # Базовое значение
        baseline_mask = results_df['scenario'] == 'baseline'
        baseline_values = {}
        if baseline_mask.any():
            baseline_data = results_df[baseline_mask]
            for metric in metrics:
                if metric in results_df.columns:
                    baseline_values[metric] = baseline_data[metric].mean()

        sensitivity_indices = []

        for metric in metrics:
            if metric not in results_df.columns:
                continue

            baseline = baseline_values.get(metric, 0)
            if baseline == 0:
                continue

            for param, param_col in zip(param_names, param_cols):
                if param_col not in results_df.columns:
                    continue

                mask = results_df[param_col].notna()
                if not mask.any():
                    continue

                grouped = results_df[mask].groupby(param_col)[metric].agg(['mean', 'std', 'count'])

                param_base = self.base_params.get(param, 1.0)
                if param_base == 0:
                    continue

                rel_changes = (grouped['mean'] - baseline) / baseline
                sensitivities = rel_changes / ((grouped.index - param_base) / param_base)

                for value, (mean_val, std_val, count) in grouped.iterrows():
                    sensitivity_indices.append({
                        'metric': metric,
                        'parameter': param,
                        'param_value': value,
                        'mean_value': mean_val,
                        'std_value': std_val,
                        'relative_change': rel_changes[value],
                        'sensitivity_index': sensitivities[value],
                        'baseline': baseline,
                        'n_runs': count
                    })

        return pd.DataFrame(sensitivity_indices)

    def plot_tornado_diagram(self, sensitivity_df, metric, top_n=10, figsize=(10, 8)):
        """Строит торнадо-диаграмму"""
        metric_df = sensitivity_df[sensitivity_df['metric'] == metric]

        if metric_df.empty:
            print(f"Нет данных для метрики {metric}")
            return None

        param_effects = (metric_df.groupby('parameter')
                         .agg(
            max_effect=('relative_change', lambda x: x.abs().max()),
            direction=('relative_change', lambda x: x.loc[x.abs().idxmax()] if not x.empty else 0)
        )
                         .reset_index())

        effects_df = param_effects.nlargest(top_n, 'max_effect')

        fig, ax = plt.subplots(figsize=figsize)

        colors = ['green' if x >= 0 else 'red' for x in effects_df['direction']]

        ax.barh(effects_df['parameter'], effects_df['max_effect'],
                color=colors, alpha=0.7)

        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Относительное изменение метрики', fontsize=12)
        ax.set_title(f'Торнадо-диаграмма чувствительности для {metric}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_parameter_response(self, results_df, parameter, metrics=None, figsize=(12, 8)):
        """Строит графики зависимости метрик от параметра"""
        if metrics is None:
            metrics = self.key_metrics[:4]

        param_col = f'param_{parameter}'
        if param_col not in results_df.columns:
            print(f"Параметр {parameter} не найден")
            return None

        param_data = results_df[results_df[param_col].notna()]

        if param_data.empty:
            return None

        n_metrics = len(metrics)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_metrics == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        # Предварительная группировка
        grouped_data = {}
        for metric in metrics:
            if metric in param_data.columns:
                grouped = (param_data.groupby(param_col)[metric]
                           .agg(['mean', 'std', 'count'])
                           .reset_index())
                grouped_data[metric] = grouped

        for i, metric in enumerate(metrics):
            if i >= len(axes):
                break

            ax = axes[i]

            if metric in grouped_data:
                grouped = grouped_data[metric]
                ci = 1.96 * grouped['std'] / np.sqrt(grouped['count'])

                ax.errorbar(grouped[param_col], grouped['mean'],
                            yerr=ci, marker='o', capsize=5,
                            capthick=1, elinewidth=1)

                ax.set_xlabel(parameter, fontsize=11)
                ax.set_ylabel(metric, fontsize=11)
                ax.set_title(f'Зависимость {metric} от {parameter}', fontsize=12)
                ax.grid(True, alpha=0.3)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        return fig

    def calculate_global_sensitivity(self, param_distributions, num_samples=100,
                                     num_runs_per_sample=2, max_workers=None):
        """Глобальный анализ чувствительности методом Монте-Карло"""
        results = []

        # Генерация выборок
        samples = {}
        rng = np.random.RandomState(42)

        for param, dist_info in param_distributions.items():
            if dist_info['dist'] == 'uniform':
                samples[param] = rng.uniform(
                    dist_info['low'], dist_info['high'], num_samples
                )
            elif dist_info['dist'] == 'normal':
                samples[param] = rng.normal(
                    dist_info['mean'], dist_info['std'], num_samples
                )

        # Подготовка задач
        all_tasks = []
        for i in range(num_samples):
            params = {param: samples[param][i] for param in param_distributions.keys()}
            for run in range(num_runs_per_sample):
                all_tasks.append({
                    'params': params,
                    'sample_id': i,
                    'run': run,
                    'run_id': f"global_{i}_{run}"
                })

        # Параллельный запуск
        if max_workers is None:
            max_workers = min(32, len(all_tasks))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(
                    self._run_single_simulation_wrapper,
                    task['params'],
                    task['run_id']
                ): task
                for task in all_tasks
            }

            for future in tqdm(as_completed(future_to_task),
                               total=len(all_tasks),
                               desc="Глобальный анализ"):
                task = future_to_task[future]
                try:
                    result = future.result(timeout=120)
                    if result and result.get('status') == 'success':
                        for param, value in task['params'].items():
                            result[f'param_{param}'] = value

                        result['sample_id'] = task['sample_id']
                        result['run'] = task['run']
                        results.append(result)
                except Exception as e:
                    print(f"Ошибка в sample {task['sample_id']}: {e}")
                    continue

        return pd.DataFrame(results)

    def analyze_parameter_interactions(self, global_results, param_pairs, metric):
        """Анализирует взаимодействия между парами параметров"""
        if global_results.empty or metric not in global_results.columns:
            print(f"Нет данных для метрики {metric}")
            return None

        n_pairs = len(param_pairs)
        fig, axes = plt.subplots(1, n_pairs, figsize=(6 * n_pairs, 5))
        if n_pairs == 1:
            axes = [axes]

        for idx, (param1, param2) in enumerate(param_pairs):
            ax = axes[idx]

            param1_col = f'param_{param1}'
            param2_col = f'param_{param2}'

            if param1_col not in global_results.columns or param2_col not in global_results.columns:
                ax.text(0.5, 0.5, f"Параметры не найдены", ha='center', va='center')
                continue

            data_subset = global_results[[param1_col, param2_col, metric]].dropna()

            if len(data_subset) < 10:
                ax.text(0.5, 0.5, "Недостаточно данных", ha='center', va='center')
                continue

            try:
                data_subset['param1_bin'] = pd.qcut(data_subset[param1_col], q=5, duplicates='drop')
                data_subset['param2_bin'] = pd.qcut(data_subset[param2_col], q=5, duplicates='drop')
            except:
                data_subset['param1_bin'] = pd.cut(data_subset[param1_col], bins=5)
                data_subset['param2_bin'] = pd.cut(data_subset[param2_col], bins=5)

            pivot = data_subset.pivot_table(
                values=metric,
                index='param1_bin',
                columns='param2_bin',
                aggfunc='mean'
            )

            sns.heatmap(pivot, annot=True, fmt='.2f', cmap='coolwarm',
                        center=pivot.values.mean() if not pivot.empty else 0,
                        ax=ax)

            ax.set_title(f'Взаимодействие {param1} и {param2}\n({metric})')
            ax.set_xlabel(param2)
            ax.set_ylabel(param1)

        plt.tight_layout()
        return fig

    def clear_cache(self):
        """Очищает кэш"""
        self._simulation_cache.clear()
        self._baseline_cache = None
        global _SIMULATION_CACHE
        _SIMULATION_CACHE.clear()


def run_sensitivity_analysis(use_parallel=True, max_workers=None, quick_mode=False):
    """
    Запускает полный анализ чувствительности - оптимизированная версия
    """
    # Загружаем данные
    birth_rate, death_rate, tfr_data, age_structure, fert_factors = load_demographic_data()

    # Базовые параметры
    base_params = {
        'initial_population_size': DEFAULT_POP_SIZE,
        'max_age_limit': 80,
        'birth_cooldown': 2,
        'calibration_factor': 0.33,
        'access_rate_children': 0.60,
        'access_rate_adults': 0.35,
        'ethnic_assortativity': 0.85,  # НОВЫЙ ПАРАМЕТР
        'mutation_frequencies': {
            'N': 0.90427, 'M694V': 0.0437, 'V726A': 0.0292,
            'M680I': 0.0192, 'R761H': 0.00363
        }
    }

    # Создаем объект для анализа
    sa = SensitivityAnalysis(
        base_params=base_params,
        birth_rate_df=birth_rate,
        death_rate_df=death_rate,
        tfr_df=tfr_data,
        age_structure_df=age_structure,
        fertility_factors_df=fert_factors
    )

    # Настройка параметров
    if quick_mode:
        num_runs = 2
        num_samples = 20
        param_ranges = {
            'calibration_factor': [0.2, 0.33, 0.5],
            'access_rate_children': [0.3, 0.6, 0.9],
            'ethnic_assortativity': [0.5, 0.85, 0.95]  # НОВЫЙ ПАРАМЕТР
        }
    else:
        num_runs = 3
        num_samples = 50
        param_ranges = {
            'calibration_factor': [0.2, 0.33, 0.5],
            'access_rate_children': [0.3, 0.6, 0.9],
            'access_rate_adults': [0.1, 0.35, 0.7],
            'birth_cooldown': [1, 2, 3, 4],
            'ethnic_assortativity': [0.5, 0.7, 0.85, 0.95]  # НОВЫЙ ПАРАМЕТР
        }

    # 1. OFAT анализ
    print("\n" + "=" * 80)
    print("1. OFAT АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ")
    print("=" * 80)

    ofat_results = sa.run_one_factor_at_a_time(
        param_ranges,
        num_runs_per_scenario=num_runs,
        max_workers=max_workers
    )

    ofat_df = pd.DataFrame(ofat_results)
    ofat_df.to_csv("sensitivity_ofat_results.csv", index=False)
    print(f"✓ OFAT результаты сохранены")

    # 2. Индексы чувствительности
    sensitivity_indices = sa.calculate_sensitivity_indices(ofat_df)
    if not sensitivity_indices.empty:
        sensitivity_indices.to_csv("sensitivity_indices.csv", index=False)

    if not sensitivity_indices.empty:
        # print("\nТОП-5 наиболее чувствительных комбинаций:")
        # top_sensitive = sensitivity_indices.nlargest(5, 'sensitivity_index')
        # for _, row in top_sensitive.iterrows():
        #     print(f"  {row['metric']} <- {row['parameter']}: {row['sensitivity_index']:.3f}")

        # Визуализация - добавлены plt.close(fig) после каждого сохранения
        for metric in sa.key_metrics:
            if metric in ofat_df.columns:
                fig = sa.plot_tornado_diagram(sensitivity_indices, metric, top_n=8)
                if fig:
                    fig.savefig(f"tornado_{metric}.png", dpi=150, bbox_inches='tight')
                    plt.close(fig)  # Закрываем фигуру

        for param in param_ranges.keys():
            fig = sa.plot_parameter_response(ofat_df, param)
            if fig:
                fig.savefig(f"response_{param}.png", dpi=150, bbox_inches='tight')
                plt.close(fig)  # Закрываем фигуру

    # 3. Глобальный анализ
    global_results = None
    if not quick_mode:
        print("\n" + "=" * 80)
        print("2. ГЛОБАЛЬНЫЙ АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ")
        print("=" * 80)

        param_distributions = {
            'calibration_factor': {'dist': 'uniform', 'low': 0.2, 'high': 0.5},
            'access_rate_children': {'dist': 'uniform', 'low': 0.3, 'high': 0.9},
            'access_rate_adults': {'dist': 'uniform', 'low': 0.1, 'high': 0.7},
        }

        global_results = sa.calculate_global_sensitivity(
            param_distributions,
            num_samples=num_samples,
            num_runs_per_sample=num_runs,
            max_workers=max_workers
        )

        global_results.to_csv("sensitivity_global_results.csv", index=False)
        print(f"✓ Глобальные результаты сохранены")

        # Анализ взаимодействий
        if not global_results.empty:
            param_pairs = [
                ('calibration_factor', 'access_rate_children'),
                ('access_rate_children', 'access_rate_adults')
            ]

            for metric in sa.key_metrics[:2]:
                if metric in global_results.columns:
                    fig = sa.analyze_parameter_interactions(global_results, param_pairs, metric)
                    if fig:
                        fig.savefig(f"interactions_{metric}.png", dpi=150, bbox_inches='tight')
                        plt.close(fig)  # Закрываем фигуру

    # Сводный отчет
    print("\n" + "=" * 80)
    print("СВОДНЫЙ ОТЧЕТ")
    print("=" * 80)

    critical_params = []
    if not sensitivity_indices.empty:
        summary = (sensitivity_indices.groupby('parameter')
                   .agg({
            'sensitivity_index': ['mean', 'max', 'std'],
        })
                   .round(3))

        print("\nИндексы чувствительности по параметрам:")
        print(summary)

        critical_params = (sensitivity_indices.groupby('parameter')['sensitivity_index']
                           .max()[lambda x: x.abs() > 0.5]
                           .index.tolist())

        if critical_params:
            print(f"\n⚠️ КРИТИЧЕСКИЕ ПАРАМЕТРЫ: {', '.join(critical_params)}")

    sa.clear_cache()

    # Финальная очистка всех фигур
    plt.close('all')

    return {
        'ofat_results': ofat_df,
        'sensitivity_indices': sensitivity_indices,
        'global_results': global_results if not quick_mode else None,
        'critical_params': critical_params
    }

# =============================================================================
# ОСНОВНОЙ БЛОК ЗАПУСКА
# =============================================================================

if __name__ == "__main__":
    # В САМОМ НАЧАЛЕ: настройка matplotlib для работы без GUI
    DEFAULT_POP_SIZE = 50_000

    matplotlib.use('Agg')  # Используем бэкенд без GUI
    console = Console()
    start_time = time.time()

    # Загрузка данных
    console.print("📂 [bold]Загрузка демографических данных...[/bold]")
    try:
        data_files = load_demographic_data()
        birth_rate, death_rate, tfr_data, age_structure, fert_factors = data_files
        console.print("   ✓ Все данные загружены успешно")
    except Exception as e:
        console.print(f"❌ Ошибка загрузки данных: {e}", style="bold red")
        sys.exit(1)

    # =========================================================================
    # ЧАСТЬ 1: МНОЖЕСТВЕННЫЕ ПРОГОНЫ (MONTE CARLO)
    # =========================================================================

    NUM_RUNS = 25

    SIM_PARAMS = {
        'initial_population_size': DEFAULT_POP_SIZE,
        'max_age_limit': 80,
        'ethnic_assortativity': 0.85
    }

    console.print(f"\n{'=' * 70}", style="bold blue")
    console.print(f" 🧬 ЧАСТЬ 1: MONTE CARLO СИМУЛЯЦИИ ({NUM_RUNS} ПРОГОНОВ)", style="bold blue")
    console.print(f" {'=' * 70}", style="bold blue")

    # Запуск множественных прогонов - ТОЛЬКО ПРОГРЕСС-БАР
    all_results = run_multiple_simulations(
        num_runs=NUM_RUNS,
        params=SIM_PARAMS,
        data_files=data_files,
        parallel=True,
        show_progress=True,  # 🔴 ПРОГРЕСС-БАР БУДЕТ!
        years_to_keep=list(range(1950, 2025)),
        verbose=False  # 🔴 БЕЗ ДЕТАЛЬНОГО ВЫВОДА
    )

    # Сохраняем результаты
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv("monte_carlo_all_runs.csv", index=False)
        console.print(f"💾 Сохранено {len(all_results)} результатов в monte_carlo_all_runs.csv")

        # Целевые значения из статьи
        target_values = {
            'm694v_homo_in_affected_pct': 11.12,
            'compound_in_affected_pct': 58.26,
            'hetero_in_affected_pct': 25.33,
            'other_homo_in_affected_pct': 2.0,
            'late_onset_pct': 3.40
        }

        # Агрегация результатов
        summary_df = aggregate_multiple_runs(all_results, target_values)

        if not summary_df.empty:
            console.print(f"\n{'=' * 70}", style="bold cyan")
            console.print("📊 АГРЕГИРОВАННЫЕ РЕЗУЛЬТАТЫ (СРЕДНЕЕ ПО ПРОГОНАМ)", style="bold cyan")
            console.print(f"{'=' * 70}", style="bold cyan")

            # Форматированный вывод
            for _, row in summary_df.iterrows():
                metric_name = row['metric'].replace('_', ' ').title()
                ci_width = row['ci_upper'] - row['mean']

                status = ""
                if 'target' in row:
                    if row['within_target']:
                        status = "✅"
                    else:
                        status = "⚠️"

                if row['metric'] == 'final_population':
                    console.print(f"{metric_name:<35} | "
                                  f"{row['mean']:>8.0f} ± {ci_width:>4.0f} | "
                                  f"Target: {row.get('target', 'N/A'):>6} {status}")
                else:
                    console.print(f"{metric_name:<35} | "
                                  f"{row['mean']:>6.2f}% ± {ci_width:>4.2f}% | "
                                  f"Target: {row.get('target', 'N/A'):>6} {status}")

            # Сохраняем сводку
            summary_df.to_csv("monte_carlo_summary.csv", index=False)

            # ВАЖНО: Делаем небольшую паузу для завершения всех потоков
            time.sleep(1)

            # Визуализация сходимости
            metrics_to_plot = [
                'm694v_homo_in_affected_pct',
                'compound_in_affected_pct',
                'hetero_in_affected_pct',
                'late_onset_pct'
            ]

            fig = plot_convergence(all_results, metrics_to_plot, target_values)
            fig.savefig('convergence_analysis.png', dpi=150, bbox_inches='tight')
            console.print(f"\n📈 График сходимости сохранен: convergence_analysis.png")
            plt.close(fig)
            plt.close('all')  # Дополнительная очистка

    # =========================================================================
    # ЧАСТЬ 2: ДЕТАЛЬНАЯ СИМУЛЯЦИЯ (ОДИН ПРОГОН С ОТЧЕТАМИ)
    # =========================================================================

    console.print(f"\n{'=' * 80}", style="bold magenta")
    console.print(" 🔬 ЧАСТЬ 2: ДЕТАЛЬНАЯ СИМУЛЯЦИЯ (ОДИН ПРОГОН)", style="bold magenta")
    console.print(f"{'=' * 80}", style="bold magenta")

    sim = GenerationSimulation(
        birth_rate_df=birth_rate,
        death_rate_df=death_rate,
        tfr_df=tfr_data,
        age_structure_df=age_structure,
        fertility_factors_df=fert_factors,
        initial_population_size=DEFAULT_POP_SIZE ,
        ethnic_assortativity=0.85,# Меньше для быстроты
        max_age_limit=80
    )

    sim.run_generation_with_calibration(verbose=False)

    # Отчеты
    console.print(f"\n{'=' * 80}", style="bold cyan")
    console.print(" 🚀 ПОЛНЫЕ ОТЧЕТЫ", style="bold cyan")
    console.print(f"{'=' * 80}", style="bold cyan")

    report_methods = [
        ('print_fertility_report', "📊 Отчет по фертильности"),
        ('_print_initial_stats', "📈 Начальная статистика"),
        ('_print_calibration_report', "⚖️ Отчет калибровки"),
        ('_print_final_summary', "📋 Итоговое резюме"),
        ('print_allele_report', "🧬 Отчет по аллелям"),
    ]

    available_methods = {name for name in dir(sim)}

    for method_name, description in report_methods:
        if method_name in available_methods:
            console.print(f"\n[bold cyan]{description}[/bold cyan]")
            try:
                getattr(sim, method_name)()
            except Exception as e:
                console.print(f"   [red]Ошибка: {e}[/red]")

    console.print(f"\n{'=' * 80}", style="bold magenta")
    console.print(" 🔬 ДЕТАЛЬНАЯ ПРОВЕРКА КАЛИБРОВКИ", style="bold magenta")
    console.print(f"{'=' * 80}", style="bold magenta")

    # Создаем симуляцию с теми же параметрами
    calibration_sim = GenerationSimulation(
        birth_rate_df=birth_rate,
        death_rate_df=death_rate,
        tfr_df=tfr_data,
        age_structure_df=age_structure,
        fertility_factors_df=fert_factors,
        initial_population_size=DEFAULT_POP_SIZE,
        max_age_limit=80,
        ethnic_assortativity=0.85
    )

    # Запускаем симуляцию
    console.print("🔄 Запуск симуляции для проверки калибровки...")
    calibration_sim.run_generation_with_calibration(verbose=False)

    # 1. Сохраняем проверку калибровки генотипов в файл
    with open('calibration_check.txt', 'w', encoding='utf-8') as f:
        # Перенаправляем stdout в файл временно
        import sys

        original_stdout = sys.stdout
        sys.stdout = f

        print("\n" + "=" * 70)
        print("📊 ПРОВЕРКА КАЛИБРОВКИ ГЕНОТИПОВ")
        print("=" * 70)
        calibration_sim.log_calibration_status()

        # Возвращаем stdout обратно
        sys.stdout = original_stdout

    console.print(f"   ✅ Результаты сохранены в [bold]calibration_check.txt[/bold]")

    # 2. Сохраняем отчет по аллелям в отдельный файл
    with open('allele_report_calibration.txt', 'w', encoding='utf-8') as f:
        original_stdout = sys.stdout
        sys.stdout = f

        print("\n" + "=" * 70)
        print("📈 ОТЧЕТ ПО АЛЛЕЛЯМ (КАЛИБРОВКА)")
        print("=" * 70)
        calibration_sim.print_allele_report()

        sys.stdout = original_stdout

    console.print(f"   ✅ Отчет по аллелям сохранен в [bold]allele_report_calibration.txt[/bold]")

    # 3. (Опционально) Сохраняем полный финальный отчет
    with open('final_report_calibration.txt', 'w', encoding='utf-8') as f:
        original_stdout = sys.stdout
        sys.stdout = f

        print("\n" + "=" * 70)
        print("📋 ПОЛНЫЙ ФИНАЛЬНЫЙ ОТЧЕТ")
        print("=" * 70)
        calibration_sim._print_final_summary()

        sys.stdout = original_stdout

    console.print(f"   ✅ Полный отчет сохранен в [bold]final_report_calibration.txt[/bold]")

    console.print(f"\n[green]✓ Все отчеты сохранены в текстовые файлы[/green]")
    # =========================================================================
    # ЧАСТЬ 3: АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ
    # =========================================================================

    console.print(f"\n{'=' * 80}", style="bold yellow")
    console.print(" 📊 ЧАСТЬ 3: АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ", style="bold yellow")
    console.print(f"{'=' * 80}", style="bold yellow")

    # Определяем параметры системы
    cpu_count = psutil.cpu_count(logical=True)
    available_memory = psutil.virtual_memory().available / (1024 ** 3)

    # Автоматический выбор режима
    quick_mode = available_memory < 4 or NUM_RUNS > 20
    max_workers = min(16, cpu_count) if cpu_count else 4

    # console.print(f"[dim]Система: {cpu_count} ядер, {available_memory:.1f} GB RAM[/dim]")
    # console.print(f"[dim]Режим анализа: {'быстрый' if quick_mode else 'полный'}[/dim]")

    try:
        results = run_sensitivity_analysis(
            use_parallel=True,
            max_workers=max_workers,
            quick_mode=quick_mode
        )

        console.print(f"\n{'=' * 80}", style="bold green")
        console.print("✅ АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ ЗАВЕРШЕН", style="bold green")
        console.print(f"{'=' * 80}", style="bold green")

        # Показываем файлы
        files = [
            "sensitivity_ofat_results.csv",
            "sensitivity_indices.csv",
            "sensitivity_global_results.csv",
        ]

        import glob

        png_files = glob.glob("*.png")
        relevant_png = [f for f in png_files if 'tornado' in f or 'response' in f or 'interactions' in f]

        console.print("\n📁 Сгенерированные файлы:")
        for file in files + relevant_png[:5]:
            if os.path.exists(file):
                size = os.path.getsize(file) / 1024
                console.print(f"   [green]✓[/green] {file} [dim]({size:.1f} KB)[/dim]")

    except Exception as e:
        console.print(f"❌ Ошибка в анализе чувствительности: {e}")
        import traceback

        traceback.print_exc()

    # =============================================================================
    # ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ: МЕДИАНЫ ПО ГОДАМ (2012-2024)
    # =============================================================================

    if all_results:
        console.print(f"\n{'=' * 80}", style="bold cyan")
        console.print(" 📊 МЕДИАННЫЙ АНАЛИЗ ПО ГОДАМ 2012-2024", style="bold cyan")
        console.print(f"{'=' * 80}", style="bold cyan")

        # Анализ по годам
        yearly_median_df = analyze_yearly_median_across_runs(
            results_list=all_results,
            years_range=range(2012, 2025),
            output_file='yearly_median_2012_2024.csv'
        )

        if not yearly_median_df.empty:
            # Печатаем сводку по ключевым метрикам
            print_yearly_analysis_summary(yearly_median_df)

            # Визуализируем тренды
            target_values = {
                'm694v_homo_in_affected_pct': 11.12,
                'compound_in_affected_pct': 58.26,
                'hetero_in_affected_pct': 25.33,
                'other_homo_in_affected_pct': 2.0,
                'prevalence_pct': 0.5  # Примерное значение
            }

            metrics_to_plot = [
                'm694v_homo_in_affected_pct',
                'compound_in_affected_pct',
                'hetero_in_affected_pct',
                'other_homo_in_affected_pct',
                'prevalence_pct',
                'total_agents'
            ]

            fig = plot_yearly_trends(yearly_median_df, metrics_to_plot, target_values)
            plt.savefig('yearly_trends_detailed.png', dpi=150, bbox_inches='tight')
            console.print(f"\n📈 Графики трендов сохранены: yearly_trends_detailed.png")

            # Дополнительно: таблица с полными данными
            console.print(f"\n📋 Полная таблица сохранена в yearly_median_2012_2024.csv")

    # =========================================================================
    # ИТОГОВЫЙ ОТЧЕТ
    # =========================================================================

    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = total_time % 60

    console.print(f"\n{'=' * 80}", style="bold blue")
    console.print(f"🏁 ВСЕ АНАЛИЗЫ ЗАВЕРШЕНЫ", style="bold blue")
    console.print(f"{'=' * 80}", style="bold blue")

    # Сводка по Monte Carlo
    if all_results:
        console.print(f"\n📊 Monte Carlo ({len(all_results)} прогонов):")
        console.print(f"   - Результаты: monte_carlo_all_runs.csv")
        console.print(f"   - Сводка: monte_carlo_summary.csv")
        console.print(f"   - Сходимость: convergence_analysis.png")

    console.print(f"\n⏱ Общее время: {minutes} мин {seconds:.1f} сек")
    console.print(f"{'=' * 80}", style="bold blue")

    # Финальная очистка
    plt.close('all')

    # Завершаем программу чисто
    sys.exit(0)

