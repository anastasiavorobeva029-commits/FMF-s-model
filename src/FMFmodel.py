from contextlib import redirect_stdout
import pandas as pd
import numpy as np
from typing import Any, Union
from rich.panel import Panel
from tqdm import tqdm
import warnings
from GenerationSimulation import GenerationSimulation
from ModelParams import ModelParams
from run_optimized import run_single_simulation_optimized
import glob
import shutil

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="main thread is not in main loop")
import matplotlib.pyplot as plt
import time
import psutil
from rich.console import Console
import matplotlib
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from typing import Dict, List, Tuple


def load_demographic_data():
    files = {
        'birth_rate': '../birth_rate_full_1950_2125.csv',
        'death_rate': '../death_rate_full_1950_2125.csv',
        'fertility_rate': '../fertility_rate_full_1950_2125.csv',
        'age_structure_1950': '../age_structure_1950.csv',
        'age_fertility_dist': '../age_fertility_dist.csv'
    }

    data = {}
    for name, filename in files.items():

        # ВАЖНО: decimal='.' так как в файлах точки (30.2)
        # skipinitialspace=True уберет лишние пробелы после разделителя
        df = pd.read_csv(filename, sep=';', skipinitialspace=True, decimal='.')

        # Принудительная конвертация числовых колонок (на всякий случай)
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

        if name in ['age_structure_1950', 'age_fertility_dist']:
            df.set_index(df.columns[0], inplace=True)
            df.index.name = 'Age_Group'
        else:
            df.set_index(df.columns[0], inplace=True)
            df.index.name = 'Year'
            # Переименовываем колонку с данными в название ключа (напр. 'birth_rate')
            df.columns = [name]

        data[name] = df
        print(f"  ✓ Загружен {filename}: {df.shape}, тип данных: {df[df.columns[0]].dtype}")

    return (
        data['birth_rate'],
        data['death_rate'],
        data['fertility_rate'],
        data['age_structure_1950'],
        data['age_fertility_dist']
    )

def print_final_validation_average(all_startup_stats, target_metrics):
    """
    Валидирует стартовые параметры популяции (1950 г.) по нескольким итерациям.
    """

    # Расширенные метки
    labels = {
        'arm_count': 'Армяне (N)',
        'oth_count': 'Другие (N)',
        'endo_count': 'Эндогамия',
        'exo_count': 'Экзогамия',
        'carrier_count': 'Носители FMF',
        'm694v_freq': 'Частота M694V'
    }

    # 1. Рассчитываем проценты для КАЖДОГО прогона отдельно (Vectorized way)
    df_startup = pd.DataFrame(all_startup_stats)
    df_startup['total_pop'] = df_startup['arm_count'] + df_startup['oth_count']

    avg_total = df_startup['total_pop'].mean()

    print("\n" + "═" * 85)
    print(f" Валидация стартовой популяции (1950 г.) | N = {len(df_startup)} итераций")
    print(f"    Средний размер выборки: {avg_total:.1f} чел.")
    print("═" * 85)
    print(f"{'Параметр':<18} | {'Среднее % (±std)':<18} | {'Цель %':>8} | {'Ошибка':>10} | {'Статус'}")
    print("-" * 85)

    for key, target_pct in target_metrics.items():
        if key not in df_startup.columns:
            continue

        # Считаем процент в каждом прогоне
        per_run_pct = (df_startup[key] / df_startup['total_pop']) * 100

        avg_pct = per_run_pct.mean()
        std_pct = per_run_pct.std()
        diff = avg_pct - target_pct

        # Статус на основе отклонения
        abs_diff = abs(diff)
        if abs_diff < 1.5:
            status = "✅ MATCH"
        elif abs_diff < 4.0:
            status = "TOLERANT"
        else:
            status = "RE-CALIBRATE"

        label = labels.get(key, key)
        # Выводим среднее значение процентов и отклонение
        print(f"{label:<18} | {avg_pct:>7.2f}% (±{std_pct:>4.2f}) | {target_pct:>7.2f}% | {diff:>+7.2f}% | {status}")



def aggregate_multiple_runs(results_list: List[Dict], target_values: Dict = None) -> pd.DataFrame:
    """
    Агрегирует финальные результаты множественных прогонов.
    Рассчитывает среднее, 95% ДИ и отклонение от эталонных значений (target).
    """


    df = pd.DataFrame(results_list)

    # Список ключевых метрик для итогового отчета
    metrics = [
        'prevalence_total_pct',  # Общая превалентность
        'm694v_homo_in_affected_pct', 'compound_in_affected_pct',  # Генетика
        'hetero_in_affected_pct', 'other_homo_in_affected_pct',
        'prevented_births_total',  # Эффективность (Сценарий 2)
        'diagnosed_pct',  # Качество диагностики
        'avg_model_birth_rate', 'avg_model_death_rate'  # Демография
    ]

    available_metrics = [m for m in metrics if m in df.columns]

    n_runs = len(df['run_id'].unique()) if 'run_id' in df.columns else len(results_list)
    print(f" Итоговая агрегация {n_runs} прогонов Монте-Карло...")

    summary = []
    for metric in available_metrics:
        values = df[metric].dropna()
        n = len(values)
        if n == 0: continue

        mean_val = values.mean()
        std_val = values.std()
        # Стандартная ошибка среднего (SEM)
        sem = std_val / np.sqrt(n) if n > 1 else 0
        ci_95 = 1.96 * sem

        # Дополнительно: перцентили (показывают реальный разброс модели)
        p025 = values.quantile(0.025)
        p975 = values.quantile(0.975)

        result = {
            'Метрика': metric,
            'Среднее': round(mean_val, 3),
            '95% ДИ (±)': round(ci_95, 3),
            'Диапазон (2.5-97.5%)': f"[{p025:.2f} - {p975:.2f}]",
            'Std_Dev': round(std_val, 3),
            'N_прогонов': n
        }

        # Сверка с эталоном (Валидация для диплома)
        if target_values and metric in target_values:
            target = target_values[metric]
            diff = mean_val - target
            result['Эталон'] = target
            result['Отклонение'] = round(diff, 3)
            # Статистический вердикт: попадает ли эталон в наш ДИ?
            result['Валидность'] = " OK" if abs(diff) <= ci_95 else " Bias"

        summary.append(result)

    summary_df = pd.DataFrame(summary)

    # Красивый вывод в консоль для контроля
    print("\n" + "─" * 95)
    print(f"{'Финальная валидация модели':^95}")
    print("─" * 95)
    print(summary_df.to_string(index=False))
    print("─" * 95 + "\n")

    return summary_df


def prepare_results_for_analysis(results_list):

    flattened_data = []

    for run_data in results_list:
        # Если run_data — это один словарь (а не список лет), оборачиваем его
        current_run = run_data if isinstance(run_data, list) else [run_data]

        for entry in current_run:
            if not isinstance(entry, dict):
                continue

            # Создаем строку данных
            row = {}


            year_val = entry.get('year')
            if year_val is None:
                continue  # Пропускаем записи без года

            row['year'] = int(year_val)

            # Переименовываем 'total_agents' в 'total_population' для совместимости
            row['total_population'] = entry.get('total_agents', entry.get('total_population', 0))

            # Подхватываем все остальные ключи (превалентность, мутации и т.д.)
            for key, value in entry.items():
                if key not in ['year', 'total_agents']:  # избегаем дублей
                    row[key] = value

            flattened_data.append(row)

    df = pd.DataFrame(flattened_data)

    return df


def analyze_yearly_median_across_runs(results_list: List[Dict],
                                      years_range: range = range(1950, 2126),
                                      output_file: str = 'yearly_median_analysis.csv'):
    # 1. Сначала "выпрямляем" вложенные словари (используем твою новую функцию)
    # Это гарантирует, что 'on_colchicine', 'diagnosed' и т.д. станут колонками
    df_all = prepare_results_for_analysis(results_list)

    # 2. Расширяем список метрик, чтобы включить всё самое важное для Master's Thesis
    key_metrics = [
        'total_population', 'total_affected', 'prevalence_total_pct',
        # Медицина и скрининг
        'on_colchicine', 'diagnosed', 'undiagnosed_symptomatic',
        'prevented_births_total',
        'total_screened',
        # Генетика - исправленные имена!
        'm694v_homo_in_affected_pct',  # вместо 'M694V_homozygous'
        'compound_in_affected_pct',  # вместо 'compound_heterozygous'
        'hetero_in_affected_pct',  # вместо 'heterozygous'
        'other_homo_in_affected_pct','m694v_homo_absolute',
        'm694v_homo_prevalence_pct',
        'total_carriers_absolute',
        # Аллели
        'allele_freq_M694V', 'allele_freq_N', 'total_births', 'total_deaths'
    ]
    # Фильтруем по калибровочному окну (2012-2024)
    df_filtered = df_all[df_all['year'].isin(years_range)].copy()

    if df_filtered.empty:
        print(f" Нет данных за годы {min(years_range)}-{max(years_range)}")
        return pd.DataFrame()

    yearly_stats = []

    for year in sorted(df_filtered['year'].unique()):
        year_data = df_filtered[df_filtered['year'] == year]
        stats = {'year': year, 'n_runs': len(year_data)}

        for metric in key_metrics:
            # Проверяем, есть ли колонка (некоторые мутации могут не появиться в конкретном году)
            if metric in year_data.columns:
                values = year_data[metric].dropna()
                if len(values) > 0:
                    stats[f'{metric}_median'] = values.median()
                    # 95% Доверительный интервал (CI) — золотой стандарт для публикаций
                    stats[f'{metric}_ci_low'] = values.quantile(0.025)
                    stats[f'{metric}_ci_high'] = values.quantile(0.975)
                    # Квартили для "ящиков с усами" (boxplots)
                    stats[f'{metric}_q25'] = values.quantile(0.25)
                    stats[f'{metric}_q75'] = values.quantile(0.75)

        yearly_stats.append(stats)

    result_df = pd.DataFrame(yearly_stats)
    result_df.to_csv(output_file, index=False)

    print(f" Обработано показателей: {len(key_metrics)} для каждого из {len(result_df)} лет.")

    return result_df


def print_yearly_analysis_summary(yearly_df, metrics_to_show=None, targets=None,
                                  verbose=True, save_to_file=None, scenario_name=None):
    """
    Выводит детальный медианный анализ по годам.

    Parameters:
    -----------
    yearly_df : pd.DataFrame
        Данные с медианами по годам
    metrics_to_show : list, optional
        Список метрик для отображения
    targets : dict, optional
        Целевые значения для сравнения
    verbose : bool, default=True
        Выводить ли в консоль
    save_to_file : str, optional
        Путь к файлу для сохранения (если None, не сохраняет)
    scenario_name : str, optional
        Название сценария для заголовка файла
    """
    if metrics_to_show is None:
        metrics_to_show = [
            'total_agents', 'prevalence_total_pct',
            'on_colchicine', 'diagnosed_pct',
            'm694v_homo_in_affected_pct', 'compound_in_affected_pct'
        ]

    # Функция, которая генерирует строку с отчетом
    def generate_report_lines():
        lines = []

        lines.append("═" * 110)
        lines.append(
            f"Динамика медианных показателей (Калибровка {int(yearly_df['year'].min())}-{int(yearly_df['year'].max())})")
        lines.append("═" * 110)

        for metric in metrics_to_show:
            if f'{metric}_median' not in yearly_df.columns:
                continue

            lines.append(f"\n🔹 Метрика: {metric.replace('_', ' ').upper()}")
            lines.append("-" * 110)
            lines.append(
                f"{'Год':<6} | {'Медиана':>12} | {'Δ к прошл.':>12} | {'IQR (75-25)':>15} | {'[Min - Max]':>22} | {'N runs':>6}")
            lines.append("-" * 110)

            prev_median = None
            for i, row in yearly_df.iterrows():
                curr_median = row[f'{metric}_median']
                q25 = row.get(f'{metric}_q25', 0)
                q75 = row.get(f'{metric}_q75', 0)
                iqr = q75 - q25

                if prev_median is not None and prev_median != 0:
                    trend = curr_median - prev_median
                    trend_str = f"{trend:>+11.2f}"
                    if 'pct' in metric:
                        trend_str += "%"
                else:
                    trend_str = f"{'-':>12}"

                is_pct = 'pct' in metric or 'freq' in metric
                fmt = ">11.2f%" if is_pct else ">12,.0f" if ('agents' in metric or 'abs' in metric) else ">12.3f"

                if '%' in fmt:
                    clean_fmt = fmt.replace('%', '')
                    val_str = f"{curr_median:{clean_fmt}}%"
                    iqr_str = f"{iqr:{clean_fmt}}%"
                else:
                    val_str = f"{curr_median:{fmt}}"
                    iqr_str = f"{iqr:{fmt}}"

                range_str = f"[{row.get(f'{metric}_min', 0):.1f} - {row.get(f'{metric}_max', 0):.1f}]"

                lines.append(
                    f"{int(row['year']):<6} | {val_str} | {trend_str} | {iqr_str} | {range_str:>22} | {int(row['n_runs']):>6}")
                prev_median = curr_median

        lines.append("\n" + "═" * 110)
        return lines

    # Генерируем отчет один раз
    report_lines = generate_report_lines()

    # Выводим в консоль если нужно
    if verbose:
        for line in report_lines:
            print(line)

    # Сохраняем в файл если нужно
    if save_to_file:
        with open(save_to_file, 'w', encoding='utf-8') as f:
            # Добавляем заголовок с информацией о сценарии
            if scenario_name:
                f.write("=" * 110 + "\n")
                f.write(f" Медианный анализ по годам\n")
                f.write(f" Сценарий: {scenario_name} | Дата: {time.strftime('%Y-%m-%d %H:%M')}\n")
                f.write("=" * 110 + "\n\n")

            for line in report_lines:
                f.write(line + "\n")

    # Возвращаем строки на случай, если нужно еще где-то использовать
    return report_lines


def run_multiple_simulations(params: ModelParams,
                             data_files: Tuple,
                             parallel: bool = True,
                             show_progress: bool = True,
                             years_to_keep: Union[List[int], range, None] = None) -> List[Dict]:
    """
    Запускает ансамбль симуляций (Monte Carlo) и валидирует стартовые условия.
    """
    birth_rate, death_rate, tfr_data, age_structure, fert_factors = data_files

    #  Безопасная очистка кэша
    from caches import _SIMULATION_CACHE, _CACHE_LOCK
    with _CACHE_LOCK:
        _SIMULATION_CACHE.clear()
        if show_progress: print(" Кэш очищен. Подготовка к свежему запуску...")

    # Настройка параллелизма
    max_workers = (os.cpu_count() or 4) - 1 if parallel else 1
    if not parallel: max_workers = 1

    results = []
    all_startup_stats = []

    # Подготовка фильтра (создаем set для O(1) поиска)
    years_set = set(years_to_keep) if years_to_keep is not None else None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Используем partial для фиксации неизменных аргументов
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

        # Запускаем задачи
        future_to_run = {executor.submit(run_func, run_id=i): i for i in range(params.num_runs)}

        # Обработка результатов по мере готовности
        iterator = as_completed(future_to_run)
        if show_progress:
            iterator = tqdm(iterator, total=params.num_runs, desc=" Monte Carlo Ensemble", unit="sim")

        for future in iterator:

            run_result = future.result(timeout=600)  # Увеличили таймаут для тяжелых ABM
            if not run_result: continue

            # 🟢 ШАГ 1: Валидация основателей (Всегда берем 0-й элемент — 1950 год)
            # Важно: берем статистику до всяких фильтров!
            first_year_data = run_result[0] if isinstance(run_result, list) else run_result
            if 'startup_validation' in first_year_data:
                all_startup_stats.append(first_year_data['startup_validation'])

            # 🟢 ШАГ 2: Фильтрация лет для экономии памяти
            if isinstance(run_result, list):
                for yearly_res in run_result:
                    if years_set is None or yearly_res.get('year') in years_set:
                        results.append(yearly_res)
            else:
                if years_set is None or run_result.get('year') in years_set:
                    results.append(run_result)


    # 🟢 ШАГ 3: Финальный научный аудит старта (1950 г.)
    if all_startup_stats:
        # Корректный расчет целей (обрабатываем и 0.9, и 90.0)
        def to_pct(val): return val * 100 if val <= 1.0 else val

        target_metrics = {
            'arm_count': to_pct(params.ethnic_distribution.get('Armenian', 0.9)),
            'oth_count': to_pct(params.ethnic_distribution.get('Other', 0.1)),
            'endo_count': to_pct(params.ethnic_assortativity),
            'exo_count': to_pct(1 - params.ethnic_assortativity)
        }

        # Вызываем нашу красивую функцию из предыдущего шага
        print_final_validation_average(all_startup_stats, target_metrics)

    print(f"Успешно завершено: {params.num_runs} прогонов. Собрано {len(results)} точек данных.")
    return results


def run_model(model_params: ModelParams):
    for f in glob.glob("*.csv") + glob.glob("*.txt"):
        os.remove(f)


    start_time = time.time()

    # Определяем параметры системы
    cpu_count = psutil.cpu_count(logical=True)
    available_memory = psutil.virtual_memory().available / (1024 ** 3)

    # Автоматический выбор режима
    quick_mode = available_memory < 4 or model_params.num_runs > 20
    max_workers = min(16, cpu_count) if cpu_count else 4

    data_files = load_demographic_data()
    birth_rate, death_rate, tfr_data, age_structure, fert_factors = data_files

    # =========================================================================
    # ЧАСТЬ 1: МНОЖЕСТВЕННЫЕ ПРОГОНЫ (MONTE CARLO)
    # =========================================================================

    console.print(f"\n{'=' * 70}", style="bold blue")
    console.print(f"  ЧАСТЬ 1: Симуляции ({model_params.num_runs} прогонов)", style="bold blue")
    console.print(f" {'=' * 70}", style="bold blue")

    # Запуск множественных прогонов
    all_results = run_multiple_simulations(
        params=model_params,
        data_files=data_files,
        parallel=True,
        show_progress=True,
        years_to_keep=list(range(1950, 2126))
    )

    # Сохраняем результаты
    df = pd.DataFrame(all_results)
    df.to_csv("monte_carlo_all_runs.csv", index=False)
    console.print(f"Сохранено {len(all_results)} результатов в monte_carlo_all_runs.csv")

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

    for _, row in summary_df.iterrows():
        metric_name = row['Метрика'].replace('_', ' ').title()
        ci_val = row.get('95% ДИ (±)', 0)
        mean_val = row['Среднее']
        status = row.get('Валидность', "")

        if 'population' in row['Метрика'] or 'agents' in row['Метрика']:
            console.print(f"{metric_name:<35} | {mean_val:>8.0f} ± {ci_val:>4.0f} | {status}")
        else:
            console.print(f"{metric_name:<35} | {mean_val:>6.2f}% ± {ci_val:>4.2f}% | {status}")

    # Сохраняем сводку
    summary_df.to_csv("monte_carlo_summary.csv", index=False)

    # =========================================================================
    # ЧАСТЬ 2: ДЕТАЛЬНАЯ СИМУЛЯЦИЯ (ОДИН КОНТРОЛЬНЫЙ ПРОГОН)
    # =========================================================================

    console.print(f"\n{'=' * 80}", style="bold magenta")
    console.print("ЧАСТЬ 2: Детальная валидация", style="bold magenta")
    console.print(f"{'=' * 80}", style="bold magenta")

    # Используем один основной объект для всех детальных отчетов
    sim = GenerationSimulation(
        params=model_params,
        birth_rate_df=birth_rate,
        death_rate_df=death_rate,
        fertility_rate_df=tfr_data,
        age_structure_df=age_structure,
        fertility_factors_df=fert_factors
    )

    # Запускаем один раз
    sim.run_simulation_with_calibration()

    console.print("\n Теоритический анализ менделевского наследования")
    sim._print_detailed_inheritance_stats()

    # Определяем текущий сценарий для именования файлов
    current_path = os.getcwd()
    if "scenario_1" in current_path:
        scenario_name = "scenario_1"
    elif "scenario_2" in current_path:
        scenario_name = "scenario_2"
    elif "scenario_3" in current_path:
        scenario_name = "scenario_3"
    else:
        scenario_name = "general"

    sim.print_population_stats(scenario_name)

    # --- Блок записи детальных отчетов в файлы ---
    reports_to_save = [
        ('calibration_check.txt', sim.log_calibration_status, "ПРОВЕРКА КАЛИБРОВКИ ГЕНОТИПОВ"),
        ('allele_report_calibration.txt', sim.print_allele_report, "ОТЧЕТ ПО АЛЛЕЛЯМ"),
        ('final_report_detailed.txt', sim._print_final_summary, "ПОЛНЫЙ ФИНАЛЬНЫЙ ОТЧЕТ"),
        ('fertility_impact.txt', sim.print_fertility_report, "АНАЛИЗ ВЛИЯНИЯ НА ФЕРТИЛЬНОСТЬ"),
    ]

    for filename, method, title in reports_to_save:
        with open(f"{scenario_name}_{filename}", 'w', encoding='utf-8') as f:
            with redirect_stdout(f):
                print("\n" + "=" * 75)
                print(f" {title}")
                print(f" Сценарий: {scenario_name} | Дата: {time.strftime('%Y-%m-%d %H:%M')}")
                print("=" * 75)
                method()
        console.print(f"   {title:<30} -> [bold]{scenario_name}_{filename}[/bold]")

    # =========================================================================
    # НОВЫЙ БЛОК: ДЕТАЛЬНЫЙ ОТЧЕТ ПО PGT ДЛЯ СЦЕНАРИЯ 3
    # =========================================================================
    if scenario_name == "scenario_3" and hasattr(sim, 'print_pgt_detailed_report'):
        with open(f"{scenario_name}_pgt_detailed_report.txt", 'w', encoding='utf-8') as f:
            with redirect_stdout(f):
                print("\n" + "=" * 75)
                print(" Детальный отчет по эффективности PGT")
                print(f" Сценарий: {scenario_name} | Дата: {time.strftime('%Y-%m-%d %H:%M')}")
                print("=" * 75)
                sim.print_pgt_detailed_report()
        console.print(f"    Детальный PGT отчет -> [bold]{scenario_name}_pgt_detailed_report.txt[/bold]")

        # Также выводим в консоль для наглядности
        console.print("\n Детальный отчет по PGT (вывод в консоль):")
        sim.print_pgt_detailed_report()

    # Отчет по скринингу для Сценария 2 и Сценария 3
    if (scenario_name == "scenario_2" or scenario_name == "scenario_3") and hasattr(sim, 'print_screening_report'):
        with open(f"{scenario_name}_screening_efficiency.txt", 'w', encoding='utf-8') as f:
            with redirect_stdout(f):
                sim.print_screening_report()
        console.print(f"    Отчет по скринингу -> {scenario_name}_screening_efficiency.txt")

    # Дополнительный отчет по биопрепаратам для Сценария 3
    if scenario_name == "scenario_3":
        with open(f"{scenario_name}_biologics_report.txt", 'w', encoding='utf-8') as f:
            with redirect_stdout(f):
                print("\n" + "=" * 75)
                print(" Отчет по биопрепаратам (Антитела К ИЛ-1)")
                print(f" Сценарий: {scenario_name} | Дата: {time.strftime('%Y-%m-%d %H:%M')}")
                print("=" * 75)
                living_agents = [a for a in sim.agents.values() if a.alive]
                resistant = [a for a in living_agents if a.is_colchicine_resistant]
                on_antibodies = [a for a in resistant if a.on_antibodies]
                print(f"\nРезистентных к колхицину: {len(resistant)}")
                print(f"Из них получают биопрепараты: {len(on_antibodies)}")
                if resistant:
                    print(f"Охват биопрепаратами: {len(on_antibodies) / len(resistant) * 100:.1f}%")
        console.print(f"   Отчет по биопрепаратам -> {scenario_name}_biologics_report.txt")


    # =============================================================================
    # ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ: МЕДИАНЫ ПО ГОДАМ
    # =============================================================================

    if all_results:
        console.print(f"\n{'=' * 85}", style="bold cyan")
        console.print(f" Медианный анализ (1950-2125)", style="bold cyan")
        console.print(f"{'=' * 85}", style="bold cyan")

        yearly_median_df = analyze_yearly_median_across_runs(
            results_list=all_results,
            years_range=range(1950, 2126),
            output_file='yearly_median_1950_2125.csv'
        )

        if not yearly_median_df.empty:
            calibration_targets = {
                'm694v_homo_in_affected_pct': 11.12,
                'compound_in_affected_pct': 58.26,
                'hetero_in_affected_pct': 25.33,
                'other_homo_in_affected_pct': 2.0,
                'prevalence_total_pct': 0.51
            }

            # ========== ИСПРАВЛЕНИЕ: Сохраняем отчет в файл ==========
            # 1. Вывод в консоль (как было)
            print_yearly_analysis_summary(
                yearly_median_df,
                targets=calibration_targets,
                verbose=False,  # Выводим в консоль
                save_to_file=f"{scenario_name}_yearly_analysis_summary.txt",  # Сохраняем в файл
                scenario_name=scenario_name
            )
            console.print(f"   Медианный анализ -> {scenario_name}_yearly_analysis_summary.txt")
            # =========================================================

            final_year_row = yearly_median_df[yearly_median_df['year'] == 2024]
            if not final_year_row.empty:
                model_val = final_year_row['prevalence_total_pct_median'].values[0]
                target_val = calibration_targets['prevalence_total_pct']
                error = abs(model_val - target_val)

                status_color = "green" if error < 0.05 else "yellow" if error < 0.1 else "red"
                console.print(f"\n[bold {status_color}] вердикт калибровки (2024 г.):[/bold {status_color}]")
                console.print(f"   Модель: {model_val:.3f}% | Цель: {target_val:.3f}% | Ошибка: {error:.3f}%")

            console.print(f"\n[dim] Полная статистика по годам (с CI 95%): yearly_median_1950_2125.csv[/dim]")

    # =========================================================================
    # ИТОГОВЫЙ ОТЧЕТ
    # =========================================================================
    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = total_time % 60
    avg_time_per_run = total_time / model_params.num_runs if model_params.num_runs > 0 else 0


    if all_results:
        output_dir = os.getcwd()
        console.print(f"\n [bold white]Статистика ансамбля ({model_params.num_runs} прогонов):[/bold white]")
        console.print(f"   • База данных: [dim]{output_dir}/[/dim][cyan]monte_carlo_all_runs.csv[/cyan]")
        console.print(f"   • Сводка ДИ:   [dim]{output_dir}/[/dim][cyan]monte_carlo_summary.csv[/cyan]")
        console.print(f"   • Калибровка:  [dim]{output_dir}/[/dim][cyan]yearly_median_1950_2125.csv[/cyan]")

    console.print(f"\n [bold white]Эффективность системы:[/bold white]")
    console.print(f"   ⚡ Общее время: {minutes} мин {seconds:.1f} сек")
    console.print(f"   ⚡ Скорость: {avg_time_per_run:.2f} сек / 1 прогон (симуляция 1950-2125)")


def compare_scenarios(file1, file2, output_dir="comparison_results",
                      name1="S1", name2="S2", bifurcation_year=2010):
    """
    Сравнивает два сценария и строит графики.

    Parameters:
    -----------
    file1, file2 : str
        Пути к CSV файлам
    output_dir : str
        Папка для сохранения результатов
    name1, name2 : str
        Названия сценариев для легенды
    bifurcation_year : int
        Год начала интервенции (точка бифуркации)
    """


    # 2. Информация о времени модификации файлов
    mtime1 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(file1)))
    mtime2 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(file2)))
    console.print(f" {os.path.basename(file1)}: {mtime1}", style="dim")
    console.print(f" {os.path.basename(file2)}: {mtime2}", style="dim")

    # 3. Полная очистка папки перед созданием новых графиков
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 4. Загрузка данных (свежее чтение)
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # 5. Определяем доступные метрики
    available_metrics = []

    base_metrics = [
        'prevalence_total_pct',
        'on_colchicine',
        'prevented_births_total',
        'm694v_homo_in_affected_pct'
    ]

    for metric in base_metrics:
        median_col = f"{metric}_median"
        if median_col in df1.columns or median_col in df2.columns:
            available_metrics.append((metric, get_metric_label(metric)))

    if not available_metrics:
        print(" Нет доступных метрик для построения графиков!")
        print(f"   Доступные колонки в df1: {df1.columns.tolist()[:10]}...")
        print(f"   Доступные колонки в df2: {df2.columns.tolist()[:10]}...")
        return

    plt.style.use('seaborn-v0_8-whitegrid')

    for metric_key, label in available_metrics:
        plt.figure(figsize=(12, 7))
        median_col = f"{metric_key}_median"

        window = 10 if 'm694v' in metric_key else 5

        # Цвета для сценариев
        scenarios = [
            (df1, '#888888', '--', name1),
            (df2, '#1f77b4', '-', name2)
        ]

        for df, color, ls, name in scenarios:
            if median_col in df.columns and not df[median_col].isna().all():
                # Убираем NaN перед rolling
                clean_data = df[['year', median_col]].dropna()
                if len(clean_data) > window:
                    smooth_median = clean_data[median_col].rolling(window=window, center=True).mean()
                    plt.plot(clean_data['year'], smooth_median,
                             color=color, linestyle=ls, linewidth=2.5, label=name)

                    ci_low = f"{metric_key}_ci_low"
                    ci_high = f"{metric_key}_ci_high"

                    if ci_low in df.columns and ci_high in df.columns:
                        clean_low = df[['year', ci_low]].dropna()
                        clean_high = df[['year', ci_high]].dropna()
                        if len(clean_low) > window:
                            smooth_low = clean_low[ci_low].rolling(window=window, center=True).mean()
                            smooth_high = clean_high[ci_high].rolling(window=window, center=True).mean()
                            plt.fill_between(clean_data['year'], smooth_low, smooth_high,
                                             color=color, alpha=0.15)

        # Оформление
        plt.legend(frameon=True, loc='best', fontsize=10)
        plt.title(f"{label} (скользящее среднее {window} лет)\nПрогноз до 2125 г.", fontsize=14, pad=20)
        plt.xlabel("Год", fontsize=12)
        plt.ylabel(label, fontsize=12)
        plt.grid(True, alpha=0.3)

        # Вертикальная линия для точки бифуркации
        plt.axvline(x=bifurcation_year, color='red', linestyle=':', alpha=0.7,
                    label=f'Начало интервенции ({bifurcation_year})')

        plt.xlim(1950, 2125)

        filename = f"{output_dir}/comparison_{metric_key}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        console.print(f" График сохранен: {filename}", style="green")

    # --- РАСЧЕТ ЭФФЕКТИВНОСТИ ---
    last_year = min(df1['year'].max(), df2['year'].max())
    analysis_window = 5

    prev_col = 'prevalence_total_pct_median'
    if prev_col in df1.columns and prev_col in df2.columns:
        prev1_data = df1[df1['year'] > (last_year - analysis_window)][prev_col].dropna()
        prev2_data = df2[df2['year'] > (last_year - analysis_window)][prev_col].dropna()

        if len(prev1_data) > 0 and len(prev2_data) > 0:
            prev1 = prev1_data.mean()
            prev2 = prev2_data.mean()

            console.print(f"\n Анализ эффективности (Среднее за {analysis_window} лет):")
            console.print(f"• Превалентность {name1}: [bold]{prev1:.3f}%[/bold]")
            console.print(f"• Превалентность {name2}: [bold]{prev2:.3f}%[/bold]")

            if prev1 > 0:
                reduction = ((prev1 - prev2) / prev1 * 100)
                console.print(f"• [green]Снижение заболеваемости:[/green] [bold]{reduction:.1f}%[/bold]")
        else:
            console.print(f" Недостаточно данных для расчета эффективности", style="yellow")

    # Проверяем предотвращенные случаи
    prevented_col = 'prevented_births_total_median'
    if prevented_col in df2.columns:
        prevented_val = df2[df2['year'] == last_year][prevented_col].values
        if len(prevented_val) > 0 and not pd.isna(prevented_val[0]):
            console.print(f"• [green]Предотвращено случаев:[/green] [bold]{prevented_val[0]:.1f}[/bold]")
    else:
        print(f" Метрика {prevented_col} не найдена в данных")


def compare_all_scenarios_together(scenario_files_dict, output_dir="comparison_all_scenarios"):
    """
    Сравнивает все три сценария на одном графике.

    Parameters:
    -----------
    scenario_files_dict : dict
        Словарь вида {'S1': 'path/to/scenario1.csv', 'S2': 'path/to/scenario2.csv', 'S3': 'path/to/scenario3.csv'}
    output_dir : str
        Папка для сохранения результатов
    """
    import matplotlib.pyplot as plt

    # Создаем папку для результатов
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Цвета и стили для сценариев
    scenario_styles = {
        'S1': {'color': '#888888', 'linestyle': '--', 'linewidth': 2.0, 'label': 'S1: Status Quo (без вмешательства)'},
        'S2': {'color': '#1f77b4', 'linestyle': '-', 'linewidth': 2.5, 'label': 'S2: Скрининг (с 2010 г.)'},
        'S3': {'color': '#2ca02c', 'linestyle': '-', 'linewidth': 2.5, 'label': 'S3: Скрининг + PGT (с 2018 г.)'}
    }

    # =========================================================================
    # 1. МЕТРИКИ, КОТОРЫЕ БЫЛИ (доля среди больных)
    # =========================================================================
    metrics_among_affected = [
        ('m694v_homo_in_affected_pct', 'Доля гомозигот M694V среди больных (%)', 0, 30),
        ('compound_in_affected_pct', 'Доля компаунд-гетерозигот среди больных (%)', 0, 80),
        ('hetero_in_affected_pct', 'Доля простых гетерозигот среди больных (%)', 0, 60),
        ('other_homo_in_affected_pct', 'Доля других гомозигот среди больных (%)', 0, 10)
    ]

    # =========================================================================
    # 2. НОВЫЕ МЕТРИКИ - АБСОЛЮТНЫЕ КОЛИЧЕСТВА И ДОЛИ В ПОПУЛЯЦИИ
    # =========================================================================
    new_metrics = [
        ('m694v_homo_absolute', 'Абсолютное количество гомозигот M694V во всей популяции (чел.)', 0, None),
        ('m694v_homo_prevalence_pct', 'Доля гомозигот M694V от общего населения (%)', 0, 1.5),
        ('total_affected_absolute', 'Абсолютное количество больных FMF (чел.)', 0, None),
        ('total_carriers_absolute', 'Абсолютное количество носителей FMF (чел.)', 0, None),
    ]

    plt.style.use('seaborn-v0_8-whitegrid')

    # =========================================================================
    # ГРАФИКИ ДЛЯ МЕТРИК СРЕДИ БОЛЬНЫХ (как было)
    # =========================================================================
    for metric_key, metric_label, y_min, y_max in metrics_among_affected:
        plt.figure(figsize=(14, 8))

        median_col = f"{metric_key}_median"
        has_data = False

        for scenario_name, file_path in scenario_files_dict.items():
            if not os.path.exists(file_path):
                console.print(f" Файл не найден: {file_path}", style="yellow")
                continue

            df = pd.read_csv(file_path)
            if df.empty or median_col not in df.columns:
                continue

            style = scenario_styles.get(scenario_name,
                                        {'color': 'black', 'linestyle': '-', 'linewidth': 2, 'label': scenario_name})

            clean_data = df[['year', median_col]].dropna()
            if len(clean_data) > 0:
                has_data = True
                window = 10
                if len(clean_data) > window:
                    smooth_data = clean_data[median_col].rolling(window=window, center=True).mean()
                    plt.plot(clean_data['year'], smooth_data,
                             color=style['color'], linestyle=style['linestyle'],
                             linewidth=style['linewidth'], label=style['label'])
                else:
                    plt.plot(clean_data['year'], clean_data[median_col],
                             color=style['color'], linestyle=style['linestyle'],
                             linewidth=style['linewidth'], label=style['label'])

        if not has_data:
            console.print(f" Нет данных для метрики {metric_key}", style="yellow")
            plt.close()
            continue

        plt.axvline(x=2010, color='#1f77b4', linestyle=':', alpha=0.5, linewidth=1.5, label='Начало скрининга (2010)')
        plt.axvline(x=2018, color='#2ca02c', linestyle=':', alpha=0.5, linewidth=1.5, label='Начало PGT (2018)')

        plt.legend(frameon=True, loc='best', fontsize=10)
        plt.title(f"{metric_label}\nСравнение трех сценариев (1950-2125)", fontsize=14, pad=20)
        plt.xlabel("Год", fontsize=12)
        plt.ylabel(metric_label, fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xlim(1950, 2125)

        if y_min is not None:
            plt.ylim(bottom=y_min)
        if y_max is not None:
            plt.ylim(top=y_max)

        filename = f"{output_dir}/all_scenarios_{metric_key}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        console.print(f" График сохранен: {filename}", style="green")

    # =========================================================================
    # НОВЫЕ ГРАФИКИ: АБСОЛЮТНЫЕ КОЛИЧЕСТВА ГОМОЗИГОТ
    # =========================================================================
    for metric_key, metric_label, y_min, y_max in new_metrics:
        plt.figure(figsize=(14, 8))

        median_col = f"{metric_key}_median"
        has_data = False

        for scenario_name, file_path in scenario_files_dict.items():
            if not os.path.exists(file_path):
                continue

            df = pd.read_csv(file_path)
            if df.empty or median_col not in df.columns:
                # Пробуем создать метрику на лету для absolute значений
                if metric_key == 'm694v_homo_absolute':
                    # Пытаемся вычислить из prevalence и population
                    prev_col = 'm694v_homo_in_affected_pct_median'
                    pop_col = 'total_population_median'
                    if prev_col in df.columns and pop_col in df.columns and 'total_affected_median' in df.columns:
                        # Создаем вычисляемую колонку
                        affected = df['total_affected_median']
                        pct_homo = df[prev_col] / 100
                        df[median_col] = affected * pct_homo
                    else:
                        continue
                elif metric_key == 'm694v_homo_prevalence_pct':
                    prev_col = 'm694v_homo_in_affected_pct_median'
                    pop_col = 'total_population_median'
                    if prev_col in df.columns and pop_col in df.columns and 'total_affected_median' in df.columns:
                        affected = df['total_affected_median']
                        pct_homo = df[prev_col] / 100
                        homo_abs = affected * pct_homo
                        df[median_col] = (homo_abs / df[pop_col]) * 100
                    else:
                        continue
                elif metric_key == 'total_affected_absolute':
                    alt_col = 'total_affected_median'
                    if alt_col in df.columns:
                        df[median_col] = df[alt_col]
                    else:
                        continue
                elif metric_key == 'total_carriers_absolute':
                    alt_col = 'total_carriers_median'
                    if alt_col in df.columns:
                        df[median_col] = df[alt_col]
                    else:
                        continue
                else:
                    continue

            style = scenario_styles.get(scenario_name,
                                        {'color': 'black', 'linestyle': '-', 'linewidth': 2, 'label': scenario_name})

            clean_data = df[['year', median_col]].dropna()
            if len(clean_data) > 0:
                has_data = True
                window = 10 if 'pct' in metric_key else 5
                if len(clean_data) > window:
                    smooth_data = clean_data[median_col].rolling(window=window, center=True).mean()
                    plt.plot(clean_data['year'], smooth_data,
                             color=style['color'], linestyle=style['linestyle'],
                             linewidth=style['linewidth'], label=style['label'])
                else:
                    plt.plot(clean_data['year'], clean_data[median_col],
                             color=style['color'], linestyle=style['linestyle'],
                             linewidth=style['linewidth'], label=style['label'])

                # Добавляем доверительные интервалы для S3
                if scenario_name == 'S3' and 'pct' not in metric_key:
                    ci_low = f"{metric_key}_ci_low"
                    ci_high = f"{metric_key}_ci_high"
                    if ci_low in df.columns and ci_high in df.columns:
                        clean_low = df[['year', ci_low]].dropna()
                        clean_high = df[['year', ci_high]].dropna()
                        if len(clean_low) > 0:
                            plt.fill_between(clean_data['year'], clean_low[ci_low], clean_high[ci_high],
                                             color=style['color'], alpha=0.15)

        if not has_data:
            console.print(f" Нет данных для метрики {metric_key}", style="yellow")
            plt.close()
            continue

        plt.axvline(x=2010, color='#1f77b4', linestyle=':', alpha=0.5, linewidth=1.5, label='Начало скрининга (2010)')
        plt.axvline(x=2018, color='#2ca02c', linestyle=':', alpha=0.5, linewidth=1.5, label='Начало PGT (2018)')

        plt.legend(frameon=True, loc='best', fontsize=10)
        plt.title(f"{metric_label}\nСравнение трех сценариев (1950-2125)", fontsize=14, pad=20)
        plt.xlabel("Год", fontsize=12)
        plt.ylabel(metric_label, fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xlim(1950, 2125)

        if y_min is not None:
            plt.ylim(bottom=y_min)
        if y_max is not None:
            plt.ylim(top=y_max)

        filename = f"{output_dir}/all_scenarios_{metric_key}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()


    # =========================================================================
    # СПЕЦИАЛЬНЫЙ ГРАФИК: СРАВНЕНИЕ АБСОЛЮТНЫХ КОЛИЧЕСТВ M694V ГОМОЗИГОТ
    # =========================================================================
    plot_m694v_homo_comparison(scenario_files_dict, output_dir)

    # =========================================================================
    # ПРЕВАЛЕНТНОСТЬ С ЗОНАМИ (как было)
    # =========================================================================
    plot_prevalence_with_interventions(scenario_files_dict, output_dir)

    # =========================================================================
    # ПРЕДОТВРАЩЕННЫЕ СЛУЧАИ (как было)
    # =========================================================================
    plot_prevented_cases_comparison(scenario_files_dict, output_dir)


def plot_m694v_homo_comparison(scenario_files_dict, output_dir):
    """Специальный график для сравнения абсолютного количества гомозигот M694V"""
    plt.figure(figsize=(14, 8))

    scenario_styles = {
        'S1': {'color': '#888888', 'linestyle': '--', 'linewidth': 2.0, 'label': 'S1: Status Quo'},
        'S2': {'color': '#1f77b4', 'linestyle': '-', 'linewidth': 2.5, 'label': 'S2: Скрининг (2010)'},
        'S3': {'color': '#2ca02c', 'linestyle': '-', 'linewidth': 2.5, 'label': 'S3: Скрининг + PGT (2018)'}
    }

    for scenario_name, file_path in scenario_files_dict.items():
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)
        if df.empty:
            continue

        style = scenario_styles.get(scenario_name,
                                    {'color': 'black', 'linestyle': '-', 'linewidth': 2, 'label': scenario_name})

        # Пытаемся получить или вычислить количество гомозигот M694V
        homo_col = 'm694v_homo_absolute_median'

        if homo_col not in df.columns:
            # Вычисляем из prevalence
            prev_col = 'm694v_homo_in_affected_pct_median'
            pop_col = 'total_population_median'
            affected_col = 'total_affected_median'

            if prev_col in df.columns and pop_col in df.columns and affected_col in df.columns:
                affected = df[affected_col]
                pct_homo = df[prev_col] / 100
                homo_abs = affected * pct_homo
                df[homo_col] = homo_abs
            else:
                continue

        clean_data = df[['year', homo_col]].dropna()
        if len(clean_data) > 0:
            window = 5
            if len(clean_data) > window:
                smooth_data = clean_data[homo_col].rolling(window=window, center=True).mean()
                plt.plot(clean_data['year'], smooth_data,
                         color=style['color'], linestyle=style['linestyle'],
                         linewidth=style['linewidth'], label=style['label'])
            else:
                plt.plot(clean_data['year'], clean_data[homo_col],
                         color=style['color'], linestyle=style['linestyle'],
                         linewidth=style['linewidth'], label=style['label'])

    plt.axvline(x=2010, color='#1f77b4', linestyle=':', alpha=0.5, linewidth=1.5, label='Начало скрининга (2010)')
    plt.axvline(x=2018, color='#2ca02c', linestyle=':', alpha=0.5, linewidth=1.5, label='Начало PGT (2018)')

    plt.legend(frameon=True, loc='best', fontsize=10)
    plt.title("Абсолютное количество гомозигот M694V в популяции\nСравнение трех сценариев", fontsize=14, pad=20)
    plt.xlabel("Год", fontsize=12)
    plt.ylabel("Количество гомозигот M694V (чел.)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(1950, 2125)
    plt.ylim(bottom=0)

    filename = f"{output_dir}/all_scenarios_m694v_homo_absolute.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_prevalence_with_interventions(scenario_files_dict, output_dir):
    """Специальный график распространенности с выделением периода после вмешательств"""
    plt.figure(figsize=(14, 8))

    scenario_styles = {
        'S1': {'color': '#888888', 'linestyle': '--', 'linewidth': 2.0, 'label': 'S1: Status Quo'},
        'S2': {'color': '#1f77b4', 'linestyle': '-', 'linewidth': 2.5, 'label': 'S2: Скрининг (2010)'},
        'S3': {'color': '#2ca02c', 'linestyle': '-', 'linewidth': 2.5, 'label': 'S3: Скрининг + PGT (2018)'}
    }

    metric_key = 'prevalence_total_pct'
    median_col = f"{metric_key}_median"

    for scenario_name, file_path in scenario_files_dict.items():
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)
        if df.empty or median_col not in df.columns:
            continue

        style = scenario_styles.get(scenario_name,
                                    {'color': 'black', 'linestyle': '-', 'linewidth': 2, 'label': scenario_name})

        clean_data = df[['year', median_col]].dropna()
        if len(clean_data) > 0:
            window = 5
            if len(clean_data) > window:
                smooth_data = clean_data[median_col].rolling(window=window, center=True).mean()
                plt.plot(clean_data['year'], smooth_data,
                         color=style['color'], linestyle=style['linestyle'],
                         linewidth=style['linewidth'], label=style['label'])
            else:
                plt.plot(clean_data['year'], clean_data[median_col],
                         color=style['color'], linestyle=style['linestyle'],
                         linewidth=style['linewidth'], label=style['label'])

    # Зоны интервенций
    plt.axvspan(2010, 2018, alpha=0.1, color='#1f77b4', label='Период скрининга (2010-2018)')
    plt.axvspan(2018, 2125, alpha=0.1, color='#2ca02c', label='Период PGT (2018-2125)')

    plt.legend(frameon=True, loc='best', fontsize=10)
    plt.title("Распространенность FMF: сравнение сценариев\nс выделением периодов вмешательств", fontsize=14, pad=20)
    plt.xlabel("Год", fontsize=12)
    plt.ylabel("Распространенность (%)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(1950, 2125)
    plt.ylim(bottom=0)

    filename = f"{output_dir}/all_scenarios_prevalence_with_zones.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    console.print(f"График сохранен: {filename}", style="green")


def plot_prevented_cases_comparison(scenario_files_dict, output_dir):
    """Сравнение предотвращенных случаев между сценариями"""
    plt.figure(figsize=(14, 8))

    scenario_styles = {
        'S2': {'color': '#1f77b4', 'linestyle': '-', 'linewidth': 2.5, 'label': 'S2: Скрининг (с 2010)'},
        'S3': {'color': '#2ca02c', 'linestyle': '-', 'linewidth': 2.5, 'label': 'S3: Скрининг + PGT (с 2018)'}
    }

    metric_key = 'prevented_births_total'
    median_col = f"{metric_key}_median"

    for scenario_name, file_path in scenario_files_dict.items():
        if scenario_name == 'S1':
            continue  # S1 не имеет предотвращенных случаев
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)
        if df.empty or median_col not in df.columns:
            continue

        style = scenario_styles.get(scenario_name,
                                    {'color': 'black', 'linestyle': '-', 'linewidth': 2, 'label': scenario_name})

        clean_data = df[['year', median_col]].dropna()
        if len(clean_data) > 0:
            plt.plot(clean_data['year'], clean_data[median_col],
                     color=style['color'], linestyle=style['linestyle'],
                     linewidth=style['linewidth'], label=style['label'])

    plt.axvline(x=2010, color='#1f77b4', linestyle=':', alpha=0.5, linewidth=1.5, label='Начало скрининга (2010)')
    plt.axvline(x=2018, color='#2ca02c', linestyle=':', alpha=0.5, linewidth=1.5, label='Начало PGT (2018)')

    plt.legend(frameon=True, loc='best', fontsize=10)
    plt.title("Накопленное количество предотвращенных случаев FMF\nСравнение эффективности скрининга и PGT",
              fontsize=14, pad=20)
    plt.xlabel("Год", fontsize=12)
    plt.ylabel("Предотвращенные случаи (накопит.)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(2000, 2125)
    plt.ylim(bottom=0)

    filename = f"{output_dir}/all_scenarios_prevented_cases.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    console.print(f" График сохранен: {filename}", style="green")

def get_metric_label(metric_key):
    """Возвращает читаемое название метрики"""
    labels = {
        'prevalence_total_pct': 'Общая превалентность FMF (%)',
        'on_colchicine': 'Пациенты на лечении (чел.)',
        'prevented_births_total': 'Предотвращенные случаи (накоп. итог)', # Уточнили, что это сумма
        'm694v_homo_in_affected_pct': 'Доля гомозигот M694V среди больных (%)',
        'diagnosed_pct': 'Уровень диагностики (%)',
        'total_population': 'Численность населения (чел.)',
        'allele_freq_M694V': 'Частота аллеля M694V в популяции',
        'fertility_rate': 'Коэффициент рождаемости (TFR)',
        'incidence_annual': 'Новые случаи заболевания (чел./год)'
    }
    return labels.get(metric_key, metric_key.replace('_', ' ').title())


if __name__ == "__main__":
    matplotlib.use('Agg')
    console = Console()
    total_start = time.time()
    root_dir = os.getcwd()

    # --- СЦЕНАРИЙ 1 ---
    console.print(Panel("ЗАПУСК СЦЕНАРИЯ 1: STATUS QUO (БАЗОВЫЙ)\nПериод: 1950 - 2125 гг."))
    path_s1 = os.path.join(root_dir, "scenario_1")
    os.makedirs(path_s1, exist_ok=True)
    os.chdir(path_s1)
    run_model(ModelParams.scenario_1())
    os.chdir(root_dir)

    # --- СЦЕНАРИЙ 2 ---
    console.print(Panel("ЗАПУСК СЦЕНАРИЯ 2: MODERNIZATION (СКРИНИНГ)\nТочка бифуркации: 2010 г."))
    path_s2 = os.path.join(root_dir, "scenario_2")
    os.makedirs(path_s2, exist_ok=True)
    os.chdir(path_s2)
    run_model(ModelParams.scenario_2())
    os.chdir(root_dir)

    # --- СЦЕНАРИЙ 3 ---
    console.print(Panel("ЗАПУСК СЦЕНАРИЯ 3: СНИЖЕНИЕ АССОРТАТИВНОСТИ + МАССОВЫЙ СКРИНИНГ + ПГД\nТочка бифуркации: 2018 г."))
    path_s3 = os.path.join(root_dir, "scenario_3")
    os.makedirs(path_s3, exist_ok=True)
    os.chdir(path_s3)
    run_model(ModelParams.scenario_3())
    os.chdir(root_dir)

    console.print("\n Сравнение сценариев на одном графике")

    scenario_files = {
        'S1': "scenario_1/yearly_median_1950_2125.csv",
        'S2': "scenario_2/yearly_median_1950_2125.csv",
        'S3': "scenario_3/yearly_median_1950_2125.csv"
    }

    compare_all_scenarios_together(
        scenario_files_dict=scenario_files,
        output_dir="comparison_all_scenarios"
    )

    # Дополнительный график предотвращенных случаев
    plot_prevented_cases_comparison(scenario_files, "comparison_all_scenarios")

    # --- СРАВНЕНИЕ СЦЕНАРИЕВ ---
    console.print("\nСравнение сценариев")

    # Сравнение Сценария 1 и 2 (бифуркация 2010)
    compare_scenarios(
        file1="scenario_1/yearly_median_1950_2125.csv",
        file2="scenario_2/yearly_median_1950_2125.csv",
        output_dir="comparison_results_sc1_sc2",
        name1="S1: Status Quo",
        name2="S2: Скрининг (2010)",
        bifurcation_year=2010
    )

    # Сравнение Сценария 1 и 3 (бифуркация 2018)
    compare_scenarios(
        file1="scenario_1/yearly_median_1950_2125.csv",
        file2="scenario_3/yearly_median_1950_2125.csv",
        output_dir="comparison_results_sc1_sc3",
        name1="S1: Status Quo",
        name2="S3: Снижение ассортативности + ПГД (2018)",
        bifurcation_year=2018
    )

    # Сравнение Сценария 2 и 3 (сравнение эффективности)
    compare_scenarios(
        file1="scenario_2/yearly_median_1950_2125.csv",
        file2="scenario_3/yearly_median_1950_2125.csv",
        output_dir="comparison_results_sc2_sc3",
        name1="S2: Скрининг (2010)",
        name2="S3: Снижение ассортативности + ПГД (2018)",
        bifurcation_year=2018
    )

    duration = (time.time() - total_start) / 60
    console.print(f"\n Полный цикл исследования завершен за  {duration:.1f} минут")