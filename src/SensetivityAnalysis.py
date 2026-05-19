from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import Dict

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

from ModelParams import ModelParams
from run_optimized import run_single_simulation_optimized


class SensitivityAnalysis:
    """
    Класс для проведения анализа чувствительности агент-ориентированной модели FMF
    Оптимизированная версия
    """

    def __init__(self, base_params: Dict, birth_rate_df, death_rate_df, tfr_df,
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

    def _run_single_simulation_wrapper(self, params: Dict, run_id):
        """Обертка для запуска симуляции с кэшированием"""
        sim_params = self.base_params.copy()
        sim_params.update(params)

        # Теперь возвращает список, берем только последний год для анализа чувствительности
        yearly_results = run_single_simulation_optimized(
            run_id=run_id,
            params=ModelParams.from_dictionary(sim_params),
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
            params=ModelParams.from_dictionary(sim_params),
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
        from caches import _SIMULATION_CACHE
        _SIMULATION_CACHE.clear()
