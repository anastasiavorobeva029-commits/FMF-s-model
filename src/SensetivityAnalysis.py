from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import Dict, List, Optional, Tuple, cast
from matplotlib.axes import Axes
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

    Parameters
    ----------
    base_params : Dict
        Базовые параметры модели
    birth_rate_df : pd.DataFrame
        Данные по рождаемости
    death_rate_df : pd.DataFrame
        Данные по смертности
    tfr_df : pd.DataFrame
        Данные по коэффициенту рождаемости
    age_structure_df : pd.DataFrame
        Структура населения по возрастам
    fertility_factors_df : pd.DataFrame
        Факторы фертильности
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

        # Ключевые метрики (унифицированные названия)
        self.key_metrics = [
            'prevalence_total_pct',
            'm694v_homo_in_affected_pct',
            'compound_in_affected_pct',
            'hetero_in_affected_pct',
            'total_population',
            'avg_model_birth_rate',
            'avg_model_death_rate',
        ]

    def run_one_factor_at_a_time(self, param_ranges: Dict[str, List[float]],
                                 num_runs_per_scenario: int = 3,
                                 max_workers: Optional[int] = None,
                                 use_caching: bool = True) -> List[Dict]:
        """
        Одномерный анализ чувствительности (OFAT)

        Parameters
        ----------
        param_ranges : Dict[str, List[float]]
            Словарь с диапазонами значений для каждого параметра
        num_runs_per_scenario : int
            Количество прогонов для каждого сценария
        max_workers : Optional[int]
            Максимальное количество параллельных потоков
        use_caching : bool
            Использовать ли кэширование результатов

        Returns
        -------
        List[Dict]
            Список результатов симуляций
        """
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

        if not all_scenario_params:
            return results

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
                    yearly_results = future.result(timeout=120)

                    if yearly_results is None:
                        continue

                    # Обработка списка результатов
                    if isinstance(yearly_results, list):
                        for year_result in yearly_results:
                            if year_result and year_result.get('status') == 'success':
                                result_copy = year_result.copy()
                                result_copy['scenario'] = params_data['scenario_name']
                                result_copy['run'] = params_data['run']
                                result_copy[f'param_{params_data["param_name"]}'] = params_data['param_value']
                                results.append(result_copy)

                    # Обработка словаря
                    elif isinstance(yearly_results, dict) and yearly_results.get('status') == 'success':
                        result_copy = yearly_results.copy()
                        result_copy['scenario'] = params_data['scenario_name']
                        result_copy['run'] = params_data['run']
                        result_copy[f'param_{params_data["param_name"]}'] = params_data['param_value']
                        results.append(result_copy)

                except Exception as e:
                    print(f"Ошибка при выполнении {params_data['scenario_name']}: {e}")
                    continue

        return results

    def _run_single_simulation_wrapper(self, params: Dict, run_id: str) -> Optional[Dict]:
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

    def _run_multiple_scenarios(self, params: Dict, num_runs: int, scenario_name: str) -> List[Dict]:
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
            yearly_results = run_func(run_id=f"{scenario_name}_{run}")

            # Пропускаем None результаты
            if yearly_results is None:
                continue

            # Обработка списка результатов (по годам)
            if isinstance(yearly_results, list):
                for year_result in yearly_results:
                    if year_result and year_result.get('status') == 'success':
                        # Создаем КОПИЮ словаря, чтобы не изменять оригинал
                        result_copy = year_result.copy()
                        result_copy['scenario'] = scenario_name
                        result_copy['run'] = run

                        for param_name, param_value in params.items():
                            result_copy[f'param_{param_name}'] = param_value

                        results.append(result_copy)

            # Обработка словаря (один результат) - для обратной совместимости
            elif isinstance(yearly_results, dict) and yearly_results.get('status') == 'success':
                # Создаем КОПИЮ словаря
                result_copy = yearly_results.copy()
                result_copy['scenario'] = scenario_name
                result_copy['run'] = run

                for param_name, param_value in params.items():
                    result_copy[f'param_{param_name}'] = param_value

                results.append(result_copy)

            # Логирование неожиданного формата
            else:
                print(
                    f"Предупреждение: Неожиданный формат результата для {scenario_name}_{run}: {type(yearly_results)}")

        return results

    def calculate_sensitivity_indices(self, results_df: pd.DataFrame,
                                      metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Рассчитывает индексы чувствительности - векторизованная версия

        Parameters
        ----------
        results_df : pd.DataFrame
            DataFrame с результатами симуляций
        metrics : Optional[List[str]]
            Список метрик для анализа

        Returns
        -------
        pd.DataFrame
            DataFrame с индексами чувствительности
        """
        if results_df.empty:
            print("DataFrame результатов пуст")
            return pd.DataFrame()

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

    def plot_tornado_diagram(self, sensitivity_df: pd.DataFrame, metric: str,
                             top_n: int = 10, figsize: Tuple[int, int] = (10, 8)) -> Optional[plt.Figure]:
        """
        Строит торнадо-диаграмму

        Parameters
        ----------
        sensitivity_df : pd.DataFrame
            DataFrame с индексами чувствительности
        metric : str
            Метрика для построения диаграммы
        top_n : int
            Количество отображаемых параметров
        figsize : Tuple[int, int]
            Размер фигуры

        Returns
        -------
        Optional[plt.Figure]
            Объект фигуры или None при ошибке
        """
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

    def plot_parameter_response(self, results_df: pd.DataFrame, parameter: str,
                                metrics: Optional[List[str]] = None,
                                figsize: Tuple[int, int] = (12, 8)) -> Optional[plt.Figure]:
        """
        Строит графики зависимости метрик от параметра

        Parameters
        ----------
        results_df : pd.DataFrame
            DataFrame с результатами симуляций
        parameter : str
            Имя параметра для анализа
        metrics : Optional[List[str]]
            Список метрик для построения
        figsize : Tuple[int, int]
            Размер фигуры

        Returns
        -------
        Optional[plt.Figure]
            Объект фигуры или None при ошибке
        """
        if metrics is None:
            metrics = self.key_metrics[:4]

        param_col = f'param_{parameter}'
        if param_col not in results_df.columns:
            print(f"Параметр {parameter} не найден в колонках: {results_df.columns.tolist()}")
            return None

        param_data = results_df[results_df[param_col].notna()].copy()

        # Конвертация в числовой тип
        try:
            param_data[param_col] = pd.to_numeric(param_data[param_col])
        except (ValueError, TypeError):
            # Оставляем как есть для строковых параметров
            pass

        if param_data.empty:
            print(f"Нет данных для параметра {parameter}")
            return None

        metrics_found = [m for m in metrics if m in param_data.columns]
        if not metrics_found:
            print(f"Ни одна из метрик {metrics} не найдена в данных")
            return None

        n_metrics = len(metrics_found)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

        # Обработка случая с одним графиком
        if n_metrics == 1:
            axes = np.array([axes])
        axes_flat = axes.flatten()

        for i, metric in enumerate(metrics_found):
            ax = cast(Axes, axes_flat[i])

            grouped = (param_data.groupby(param_col, sort=True)[metric]
                       .agg(['mean', 'std', 'count'])
                       .reset_index())

            # Сортировка для числовых параметров
            try:
                grouped = grouped.sort_values(by=param_col)
            except:
                pass

            # Расчет доверительных интервалов
            ci = np.zeros(len(grouped))
            mask = grouped['count'] > 1
            ci[mask] = 1.96 * grouped.loc[mask, 'std'] / np.sqrt(grouped.loc[mask, 'count'])

            x_values = grouped[param_col].to_numpy()
            y_values = grouped['mean'].to_numpy()

            # Проверка на числовой тип
            is_numeric = pd.api.types.is_numeric_dtype(grouped[param_col])

            if is_numeric:
                ax.errorbar(x_values, y_values, yerr=ci,
                            marker='o', capsize=5, capthick=1, elinewidth=1,
                            linestyle='-', linewidth=1.5)
            else:
                x_indices = list(range(len(x_values)))
                ax.errorbar(x_indices, y_values, yerr=ci,
                            marker='o', capsize=5, capthick=1, elinewidth=1,
                            linestyle='-', linewidth=1.5)
                ax.set_xticks(x_indices)
                ax.set_xticklabels(x_values, rotation=45, ha='right')

            ax.set_xlabel(parameter, fontsize=11)
            y_label = metric.replace('_pct', ' (%)').replace('_', ' ').title()
            ax.set_ylabel(y_label, fontsize=11)
            ax.set_title(f'Зависимость {y_label} от {parameter}', fontsize=12)
            ax.grid(True, alpha=0.3)

            # Добавление baseline
            if 'scenario' in results_df.columns:
                baseline_mask = results_df['scenario'] == 'baseline'
                if baseline_mask.any() and metric in results_df.columns:
                    baseline_value = results_df.loc[baseline_mask, metric].mean()
                    if pd.notna(baseline_value):
                        ax.axhline(y=baseline_value, color='red', linestyle='--',
                                   alpha=0.5, label=f'Baseline: {baseline_value:.2f}')
                        ax.legend(fontsize=9)

        # Скрытие неиспользуемых подграфиков
        for j in range(len(metrics_found), len(axes_flat)):
            cast(Axes, axes_flat[j]).set_visible(False)

        plt.tight_layout()
        return fig

    def calculate_global_sensitivity(self, param_distributions: Dict,
                                     num_samples: int = 100,
                                     num_runs_per_sample: int = 2,
                                     max_workers: Optional[int] = None,
                                     random_seed: int = 42) -> pd.DataFrame:
        """
        Глобальный анализ чувствительности методом Монте-Карло

        Parameters
        ----------
        param_distributions : Dict
            Словарь с распределениями параметров
        num_samples : int
            Количество семплов
        num_runs_per_sample : int
            Количество прогонов на семпл
        max_workers : Optional[int]
            Максимальное количество параллельных потоков
        random_seed : int
            Seed для генератора случайных чисел

        Returns
        -------
        pd.DataFrame
            DataFrame с результатами глобального анализа
        """
        results = []

        # Генерация выборок
        samples = {}
        rng = np.random.RandomState(random_seed)

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

    def analyze_parameter_interactions(self, global_results: pd.DataFrame,
                                       param_pairs: List[Tuple[str, str]],
                                       metric: str) -> Optional[plt.Figure]:
        """
        Анализирует взаимодействия между парами параметров

        Parameters
        ----------
        global_results : pd.DataFrame
            DataFrame с результатами глобального анализа
        param_pairs : List[Tuple[str, str]]
            Список пар параметров для анализа
        metric : str
            Метрика для анализа

        Returns
        -------
        Optional[plt.Figure]
            Объект фигуры или None при ошибке
        """
        if global_results.empty or metric not in global_results.columns:
            print(f"Нет данных для метрики {metric}")
            return None

        n_pairs = len(param_pairs)
        if n_pairs == 0:
            print("Нет пар параметров для анализа")
            return None

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

    def save_results(self, results_df: pd.DataFrame, output_dir: str = "sensitivity_results") -> None:
        """
        Сохраняет результаты анализа чувствительности

        Parameters
        ----------
        results_df : pd.DataFrame
            DataFrame с результатами для сохранения
        output_dir : str
            Директория для сохранения результатов
        """
        if results_df.empty:
            print("Нет результатов для сохранения")
            return

        import os
        import json

        os.makedirs(output_dir, exist_ok=True)

        # Сохраняем DataFrame
        results_df.to_csv(f"{output_dir}/sensitivity_results.csv", index=False)

        # Сохраняем метаданные
        metadata = {
            'base_params': self.base_params,
            'key_metrics': self.key_metrics,
            'num_results': len(results_df),
            'columns': list(results_df.columns)
        }

        with open(f"{output_dir}/sensitivity_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"Результаты сохранены в {output_dir}/")

    @classmethod
    def load_results(cls, results_dir: str = "sensitivity_results") -> Tuple[pd.DataFrame, Dict]:
        """
        Загружает ранее сохраненные результаты

        Parameters
        ----------
        results_dir : str
            Директория с сохраненными результатами

        Returns
        -------
        Tuple[pd.DataFrame, Dict]
            DataFrame с результатами и словарь с метаданными
        """
        import json
        import os

        results_df = pd.read_csv(f"{results_dir}/sensitivity_results.csv")

        with open(f"{results_dir}/sensitivity_metadata.json", 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        print(f"Загружено {len(results_df)} результатов")
        print(f"Базовые параметры: {metadata['base_params']}")

        return results_df, metadata

    def generate_summary_report(self, sensitivity_df: pd.DataFrame,
                                results_df: pd.DataFrame,
                                output_dir: str = "sensitivity_results") -> None:
        """
        Генерирует сводный отчет по анализу чувствительности

        Parameters
        ----------
        sensitivity_df : pd.DataFrame
            DataFrame с индексами чувствительности
        results_df : pd.DataFrame
            DataFrame с результатами симуляций
        output_dir : str
            Директория для сохранения отчета
        """
        import os

        os.makedirs(output_dir, exist_ok=True)

        with open(f"{output_dir}/sensitivity_report.txt", 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ОТЧЕТ ПО АНАЛИЗУ ЧУВСТВИТЕЛЬНОСТИ\n")
            f.write("=" * 80 + "\n\n")

            # Основная статистика
            f.write(f"Базовых параметров: {len(self.base_params)}\n")
            f.write(f"Всего результатов: {len(results_df)}\n")
            if 'scenario' in results_df.columns:
                f.write(f"Уникальных сценариев: {results_df['scenario'].nunique()}\n")
            f.write("\n")

            # Самые чувствительные параметры для каждой метрики
            f.write("НАИБОЛЕЕ ЧУВСТВИТЕЛЬНЫЕ ПАРАМЕТРЫ\n")
            f.write("-" * 80 + "\n")

            for metric in self.key_metrics:
                metric_df = sensitivity_df[sensitivity_df['metric'] == metric]
                if not metric_df.empty:
                    top_params = metric_df.nlargest(3, 'sensitivity_index')[['parameter', 'sensitivity_index']]
                    f.write(f"\n{metric}:\n")
                    for _, row in top_params.iterrows():
                        f.write(f"  - {row['parameter']}: {row['sensitivity_index']:.3f}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Отчет сгенерирован автоматически\n")

        print(f"Отчет сохранен в {output_dir}/sensitivity_report.txt")

    def clear_cache(self) -> None:
        """Очищает кэш"""
        self._simulation_cache.clear()
        self._baseline_cache = None
        try:
            from caches import _SIMULATION_CACHE
            _SIMULATION_CACHE.clear()
        except ImportError:
            pass  # caches module not available