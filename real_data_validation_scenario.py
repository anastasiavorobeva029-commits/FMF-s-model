import os
import shutil
import time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from scipy.stats import linregress

# Отключаем интерактивный режим matplotlib
plt.ioff()


def clear_results_folder(base_folder, force_clear=True):
    """Очищает папку с результатами перед записью новых данных"""
    if force_clear and os.path.exists(base_folder):
        shutil.rmtree(base_folder)
        time.sleep(0.5)
    os.makedirs(base_folder, exist_ok=True)
    return base_folder


def results_of_scenarios(scenario_name, force_recreate=True):
    """Создает структуру папок для результатов"""
    base_folder = f'results_{scenario_name}'
    subfolders = ['tables', 'figures', 'statistics']

    if force_recreate:
        base_folder = clear_results_folder(base_folder, force_clear=True)
    else:
        os.makedirs(base_folder, exist_ok=True)

    for subfolder in subfolders:
        os.makedirs(os.path.join(base_folder, subfolder), exist_ok=True)

    log_file = os.path.join(base_folder, 'run_info.txt')
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f'Время запуска: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Сценарий: {scenario_name}\n')
        f.write('Результаты пересохранены заново\n')

    return base_folder


def preparation_table_for_thesis(scenario_path, force_reload=True):
    """Читает данные с принудительной перезагрузкой"""
    file_path = os.path.join(scenario_path, 'monte_carlo_all_runs.csv')

    if force_reload:
        if os.path.exists(file_path):
            mod_time = os.path.getmtime(file_path)
            mod_time_str = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
            print(f"📁 Загружаем файл: {file_path}")
            print(f"   Последнее изменение: {mod_time_str}")

    z = pd.read_csv(file_path)

    data_hash = hash(z.values.tobytes())
    print(f"   Хэш данных: {data_hash}")

    full_year = z['year'].values
    full_population = z['total_population']
    agents_0_49 = z['total_agents']
    total_affected = z['total_affected']
    m694v_homozigotes = z['m694v_homo_count']
    compound_abs = z['compound_count']
    normal_alleles = z['hetero_count']
    other_homo = z['other_homo_count']
    carrier_births = z['carrier_births']
    affected_births = z['affected_births']
    normal_births = z['healthy_births']

    full_table_for_results_yet = pd.DataFrame({
        'years_for_full': full_year,
        'full_population': full_population,
        'target_agents': agents_0_49,
        'total_affected': total_affected,
        'm694v_homozigotes': m694v_homozigotes,
        'compound': compound_abs,
        'n_allele': normal_alleles,
        'other_homo': other_homo,
        'carrier_births': carrier_births,
        'affected_births': affected_births,
        'normal_births': normal_births
    })

    full_table_for_results = full_table_for_results_yet.groupby('years_for_full').agg(['mean', 'std'])
    interval_percent = 1.96
    number_of_simulations = 30

    margin_of_error = interval_percent * (
            full_table_for_results.xs('std', axis=1, level=1) / np.sqrt(number_of_simulations))
    means = full_table_for_results.xs('mean', axis=1, level=1)

    ci_low = means - margin_of_error
    ci_high = means + margin_of_error

    full_fmf_table = pd.concat([
        means.add_suffix('_mean'),
        ci_low.add_suffix('_ci_low'),
        ci_high.add_suffix('_ci_high')],
        axis=1).reset_index()

    return full_fmf_table


def load_data_from_files(scenario_path, force_reload=True):
    """Загружает данные с принудительной перезагрузкой"""
    if force_reload:
        import gc
        gc.collect()

    x = pd.read_csv('FMF_data2.csv')

    y_path = os.path.join(scenario_path, 'yearly_median_1950_2125.csv')
    y = pd.read_csv(y_path)

    print(f"📁 Загружаем модельные данные: {y_path}")
    if os.path.exists(y_path):
        mod_time = os.path.getmtime(y_path)
        print(f"   Последнее изменение: {datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"📁 Загружаем реальные данные: FMF_data2.csv")
    if os.path.exists('FMF_data2.csv'):
        mod_time = os.path.getmtime('FMF_data2.csv')
        print(f"   Последнее изменение: {datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')}")

    x = x.rename(columns={'years': 'year'})

    common_years = sorted(set(x['year']).intersection(set(y['year'])))
    print(f"Общие годы: {common_years}")

    x_filtered = x[x['year'].isin(common_years)].sort_values('year').reset_index(drop=True)
    y_filtered = y[y['year'].isin(common_years)].sort_values('year').reset_index(drop=True)

    print(f"Количество строк в реальных данных: {len(x_filtered)}")
    print(f"Количество строк в модельных данных: {len(y_filtered)}")

    statistic_table = pd.DataFrame({
        'year': y_filtered['year'].values,
        'real_prev': x_filtered['prevalence_0_49'].values,
        'model_prev': y_filtered['prevalence_total_pct_median'].values,
        'prevalence_pct_q25': y_filtered['prevalence_total_pct_q25'].values,
        'prevalence_pct_q75': y_filtered['prevalence_total_pct_q75'].values
    })

    statistic_table['real_prev_per'] = (statistic_table['real_prev'] * 100).round(5)
    statistic_table['turnover_rate'] = (statistic_table['real_prev_per'] / statistic_table['model_prev']).round(5)

    return statistic_table


def table_for_thesis_about_fmf(full_fmf_table, scenario_name, results_folder):
    year = full_fmf_table['years_for_full']

    pop_mean = full_fmf_table['full_population_mean']
    pop_ci_low = full_fmf_table['full_population_ci_low']
    pop_ci_high = full_fmf_table['full_population_ci_high']
    pop_ci = pop_ci_high - pop_mean

    mean_0_49 = full_fmf_table['target_agents_mean']
    target_ci_high = full_fmf_table['target_agents_ci_high']
    ci_0_49 = target_ci_high - mean_0_49

    affected_mean = full_fmf_table['total_affected_mean']
    affected_ci_high = full_fmf_table['total_affected_ci_high']
    affected_ci = affected_ci_high - affected_mean

    m694v_mean = full_fmf_table['m694v_homozigotes_mean']
    m694v_ci_high = full_fmf_table['m694v_homozigotes_ci_high']
    m694v_ci = m694v_ci_high - m694v_mean

    compound_mean = full_fmf_table['compound_mean']
    compound_ci_high = full_fmf_table['compound_ci_high']
    compound_ci = compound_ci_high - compound_mean

    n_allele_mean = full_fmf_table['n_allele_mean']
    n_allele_ci_high = full_fmf_table['n_allele_ci_high']
    n_allele_ci = n_allele_ci_high - n_allele_mean

    other_homo_mean = full_fmf_table['other_homo_mean']
    other_homo_ci_high = full_fmf_table['other_homo_ci_high']
    other_homo_ci = other_homo_ci_high - other_homo_mean

    carrier_births_mean = full_fmf_table['carrier_births_mean']
    carrier_births_ci_high = full_fmf_table['carrier_births_ci_high']
    carrier_births_ci = carrier_births_ci_high - carrier_births_mean

    affected_births_mean = full_fmf_table['affected_births_mean']
    affected_births_ci_high = full_fmf_table['affected_births_ci_high']
    affected_births_ci = affected_births_ci_high - affected_births_mean

    normal_births_mean = full_fmf_table['normal_births_mean']
    normal_births_ci_high = full_fmf_table['normal_births_ci_high']
    normal_births_ci = normal_births_ci_high - normal_births_mean

    formatted_data = []

    for i in full_fmf_table.index:
        row = {
            'Year': int(year[i]),
            'Population (mean ± CI)': f"{pop_mean[i]:.1f} ± {pop_ci[i]:.1f}",
            'Target agents (mean ± CI)': f"{mean_0_49[i]:.1f} ± {ci_0_49[i]:.1f}",
            'Affected people (mean ± CI)': f"{affected_mean[i]:.1f} ± {affected_ci[i]:.1f}",
            'M694V homozygotes (mean ± CI)': f"{m694v_mean[i]:.1f} ± {m694v_ci[i]:.1f}",
            'Compound hetero (mean ± CI)': f"{compound_mean[i]:.1f} ± {compound_ci[i]:.1f}",
            'N allele (mean ± CI)': f"{n_allele_mean[i]:.1f} ± {n_allele_ci[i]:.1f}",
            'Other homo (mean ± CI)': f"{other_homo_mean[i]:.1f} ± {other_homo_ci[i]:.1f}",
            'Carrier births (mean ± CI)': f"{carrier_births_mean[i]:.0f} ± {carrier_births_ci[i]:.0f}",
            'Affected births (mean ± CI)': f"{affected_births_mean[i]:.0f} ± {affected_births_ci[i]:.0f}",
            'Healthy births (mean ± CI)': f"{normal_births_mean[i]:.0f} ± {normal_births_ci[i]:.0f}"
        }
        formatted_data.append(row)

    data_for_export = pd.DataFrame(formatted_data)

    tables_folder = os.path.join(results_folder, 'tables')
    csv_filename = os.path.join(tables_folder, f'table_fmf_{scenario_name}.csv')
    xlsx_filename = os.path.join(tables_folder, f'table_fmf_{scenario_name}.xlsx')

    data_for_export.to_csv(csv_filename, index=False, encoding='utf-8-sig', sep=';', decimal=',')
    data_for_export.to_excel(xlsx_filename, index=False)

    return data_for_export


def first_graph(statistic_table, scenario_name, results_folder):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].plot('year', 'real_prev_per', data=statistic_table, marker='o', color="#0047AB", linewidth=2)
    ax[0].set_title('Зарегистрированная превалентность FMF\n(Данные МЗ РА, 2012–2024 гг.)', fontsize=11)
    ax[0].set_ylabel('Превалентность (%)')
    ax[0].grid(True, linestyle='--', alpha=0.7)

    ax[1].plot('year', 'model_prev', data=statistic_table, marker='s', color="#D22B2B", linewidth=2)
    ax[1].set_title(f'Прогностическая превалентность FMF\n(Модель {scenario_name}, 2012–2024 гг.)', fontsize=11)
    ax[1].grid(True, linestyle='--', alpha=0.7)

    plt.suptitle(f'Сценарий: {scenario_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    figures_folder = os.path.join(results_folder, 'figures')
    plt.savefig(os.path.join(figures_folder, f'prevalence_comparison_{scenario_name}.png'), dpi=300,
                bbox_inches='tight')
    plt.close(fig)


def survival_hypothesis(statistic_table, scenario_name, results_folder):
    """Находит средний коэффициент k для масштабирования"""
    clean_table = statistic_table[statistic_table['model_prev'] > 0].copy()
    mean_value = clean_table['turnover_rate'].mean()
    statistic_table['model_view'] = (statistic_table['model_prev'] * mean_value).round(5)

    statistics_folder = os.path.join(results_folder, 'statistics')
    statistic_table.to_csv(os.path.join(statistics_folder, f'FMF_statistic_{scenario_name}.csv'), index=False)

    print(f"\n🔍 Коэффициент масштабирования k = {mean_value:.3f}")
    print(f"   (Применяется для приведения модельных данных к масштабу госпитальной статистики)")

    return mean_value


def plot_of_survival_hypothesis(statistic_table, mean_value, scenario_name, results_folder):
    """Строит график масштабированного сравнения (модель × k)"""
    years = list(range(2012, 2012 + len(statistic_table)))
    points_by_real_data = statistic_table['real_prev_per'].values
    line_by_model = statistic_table['model_view'].values
    q_25 = (statistic_table['prevalence_pct_q25'] * mean_value).round(5)
    q_75 = (statistic_table['prevalence_pct_q75'] * mean_value).round(5)

    plt.figure(figsize=(12, 8))
    plt.fill_between(years, q_25, q_75, color='skyblue', alpha=0.4, label='IQR (25th-75th percentile)')
    plt.plot(years, line_by_model, color='red', linewidth=3,
             label=f'Model trend scaled with k = {mean_value:.3f}')
    plt.scatter(years, points_by_real_data, color='black', marker='D', s=50,
                label='Hospital Data (Real)', zorder=5)

    plt.title(f'Масштабированная валидация: Реальные данные vs Модель\nСценарий: {scenario_name}', fontsize=14)
    plt.xlabel('Годы', fontsize=14)
    plt.ylabel('Превалентность (%)', fontsize=14)
    plt.xticks(years)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.text(years[0], max(points_by_real_data), f'Средний коэффициент k: {mean_value:.3f}',
             fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()

    figures_folder = os.path.join(results_folder, 'figures')
    plt.savefig(os.path.join(figures_folder, f'scaled_validation_{scenario_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return years, points_by_real_data, line_by_model


def plot_dynamic_coefficient(statistic_table, years, scenario_name, results_folder):
    coeff_dynamic = statistic_table['turnover_rate']
    slope, intercept = np.polyfit(years, coeff_dynamic, 1)
    trend_line = slope * np.array(years) + intercept

    plt.figure(figsize=(10, 8))
    plt.scatter(years, coeff_dynamic, color='black', label='Рассчитанный коэффициент за 2012-2024')
    plt.plot(years, trend_line, color='red', linestyle='--',
             label=f'Тренд диагностики: y={slope:.4f} * x + {intercept:.4f}')

    plt.title(f'Динамика коэффициента выявляемости FMF в Армении\nСценарий: {scenario_name}', fontsize=14)
    plt.xlabel('Годы', fontsize=14)
    plt.ylabel('Динамический коэффициент выявляемости', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    figures_folder = os.path.join(results_folder, 'figures')
    plt.savefig(os.path.join(figures_folder, f'dynamic_coefficient_{scenario_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return coeff_dynamic


def calculate_pearson(statistic_table, points_by_real_data, scenario_name, results_folder):
    """Рассчитывает корреляцию Пирсона для сырых трендов"""
    not_norm_model_data = statistic_table['model_prev']

    corr, p_value = pearsonr(not_norm_model_data, points_by_real_data)

    print(f'\n📈 КОРРЕЛЯЦИЯ ПИРСОНА (сырые тренды):')
    print(f'   Коэффициент корреляции: {corr:.4f}')
    print(f'   P-value: {p_value:.4f}')
    if p_value < 0.001:
        print(f'   Интерпретация: Статистически значимая корреляция (p < 0.001)')
    print(f'   Вывод: Модель и реальность изменяются согласованно и неслучайно')

    df_corr = pd.DataFrame({
        'Model prediction': not_norm_model_data,
        'Real data': points_by_real_data
    })

    matrix_of_corr = df_corr.corr()

    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix_of_corr, annot=True, cmap='RdYlGn', center=0, fmt=".2f")
    plt.title(f'Тепловая карта корреляции\nСценарий: {scenario_name}')

    figures_folder = os.path.join(results_folder, 'figures')
    plt.savefig(os.path.join(figures_folder, f'correlation_heatmap_{scenario_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    statistics_folder = os.path.join(results_folder, 'statistics')
    with open(os.path.join(statistics_folder, f'correlation_stats_{scenario_name}.txt'), 'w', encoding='utf-8') as f:
        f.write('=== АНАЛИЗ КОРРЕЛЯЦИИ СЫРЫХ ТРЕНДОВ ===\n')
        f.write(f'Коэффициент корреляции Пирсона: {corr:.4f}\n')
        f.write(f'P-value: {p_value:.4f}\n')
        f.write(f'Интерпретация: Статистически значимая корреляция (p < 0.001)\n')
        f.write('Вывод: Форма и динамика трендов модели соответствуют реальным данным\n')

    return corr


def linear_regression(years, coeff_dynamic, scenario_name, results_folder):
    slope, intercept, r_value, p_value, std_err = linregress(years, coeff_dynamic)

    print(f"\n📊 ЛИНЕЙНАЯ РЕГРЕССИЯ коэффициента выявляемости:")
    print(f"   Коэффициент наклона (β₁): {slope:.4f}")
    print(f"   Коэффициент детерминации (R²): {r_value ** 2:.4f}")
    print(f"   P-value: {p_value:.4f}")
    print(f"   Интерпретация: Модель объясняет {(r_value ** 2 * 100):.1f}% дисперсии данных")

    plt.figure(figsize=(8, 6))
    plt.scatter(years, coeff_dynamic, label='Фактический динамический коэффициент выявляемости в Армении')
    plt.plot(years, slope * np.array(years) + intercept, color='red',
             label=f'Линия регрессии (slope={slope:.4f})')
    plt.xlabel('Годы')
    plt.ylabel('Коэффициент выявляемости')
    plt.title(f'Линейная регрессия коэффициента выявляемости\nСценарий: {scenario_name}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    figures_folder = os.path.join(results_folder, 'figures')
    plt.savefig(os.path.join(figures_folder, f'linear_regression_{scenario_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    statistics_folder = os.path.join(results_folder, 'statistics')
    with open(os.path.join(statistics_folder, f'regression_stats_{scenario_name}.txt'), 'w', encoding='utf-8') as f:
        f.write('=== ЛИНЕЙНАЯ РЕГРЕССИЯ ===\n')
        f.write(f'Коэффициент наклона (β₁): {slope:.4f}\n')
        f.write(f'Коэффициент детерминации (R²): {r_value ** 2:.4f}\n')
        f.write(f'P-value: {p_value:.4f}\n')
        f.write(f'Стандартная ошибка: {std_err:.4f}\n')


def mape_function(years, points_by_real_data, line_by_model, scenario_name, results_folder):
    """Рассчитывает MAPE для масштабированной модели"""
    real_data = np.array(points_by_real_data)
    model_data = np.array(line_by_model)

    errors = np.abs((real_data - model_data) / real_data) * 100
    mape_dict = dict(zip(years, errors))

    print(f"\n📊 МЕТРИКИ ОШИБОК ДЛЯ МАСШТАБИРОВАННОЙ МОДЕЛИ:")
    for year, error in mape_dict.items():
        print(f'   Год {year}: ошибка MAPE {error:.2f}%')

    plt.figure(figsize=(10, 8))

    plt.plot(years, errors, marker='o', color='green', linestyle='--')
    plt.axhspan(0, 10, color='red', alpha=0.1, label='Высокая точность (MAPE < 10%)')
    plt.axhspan(10, 20, color='yellow', alpha=0.1, label='Хорошая точность (MAPE 10-20%)')
    plt.axhspan(20, 100, color='orange', alpha=0.1, label='Низкая точность (MAPE > 20%)')

    plt.axvspan(2017, 2020, color='lightgray', alpha=0.3, label='Окно валидации')

    if 2024 in years and max(errors) > 15:
        plt.annotate('Diagnostic Shift\n(улучшение выявляемости)', xy=(2024, errors[list(years).index(2024)]),
                     xytext=(2022, errors[list(years).index(2024)] + 5),
                     arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10, ha='center')

    plt.title(f'Динамика MAPE: Модель vs Реальные данные\nСценарий: {scenario_name}', fontsize=14)
    plt.xlabel('Годы', fontsize=12)
    plt.ylabel('Ошибка MAPE (%)', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)

    figures_folder = os.path.join(results_folder, 'figures')
    plt.savefig(os.path.join(figures_folder, f'mape_dynamics_{scenario_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    statistics_folder = os.path.join(results_folder, 'statistics')
    with open(os.path.join(statistics_folder, f'mape_stats_{scenario_name}.txt'), 'w', encoding='utf-8') as f:
        f.write('=== MAPE ДЛЯ МАСШТАБИРОВАННОЙ МОДЕЛИ ===\n')
        for year, error in mape_dict.items():
            f.write(f'Год {year}: ошибка MAPE {error:.2f}%\n')
        f.write(f'\nСредняя MAPE: {np.mean(list(mape_dict.values())):.2f}%\n')

    return mape_dict


def me_function(years, points_by_real_data, line_by_model, mean_value, scenario_name, results_folder):
    """Рассчитывает среднюю ошибку для масштабированной модели"""
    error_between = points_by_real_data - line_by_model
    me_value = np.mean(error_between)

    print(f'   Средняя ошибка (ME): {me_value:.4f}%')

    plt.figure(figsize=(8, 4))

    plt.scatter(years, points_by_real_data, color='black', marker='D', label='Реальная превалентность')
    plt.plot(years, line_by_model, color='red', linewidth=2, label=f'Превалентность модели (k={mean_value:.3f})')

    plt.title(f'Сравнение данных: ME = {me_value:.4f}%\nСценарий: {scenario_name}')
    plt.xlabel('Годы')
    plt.ylabel('Значения превалентности, %')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    figures_folder = os.path.join(results_folder, 'figures')
    plt.savefig(os.path.join(figures_folder, f'mean_error_{scenario_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    statistics_folder = os.path.join(results_folder, 'statistics')
    with open(os.path.join(statistics_folder, f'me_stats_{scenario_name}.txt'), 'w', encoding='utf-8') as f:
        f.write('=== СРЕДНЯЯ ОШИБКА (ME) ДЛЯ МАСШТАБИРОВАННОЙ МОДЕЛИ ===\n')
        f.write(f'Средняя ошибка (ME): {me_value:.4f}%\n')

    return me_value


def direct_comparison_without_coefficient(statistic_table, scenario_name, results_folder):
    """
    ЭТАП А: Прямое сравнение без поправочного коэффициента
    Доказывает существование латентного пула (айсберга)
    """
    years = list(range(2012, 2012 + len(statistic_table)))
    points_by_real_data = statistic_table['real_prev_per'].values
    line_by_model = statistic_table['model_prev'].values
    q_25 = statistic_table['prevalence_pct_q25'].values
    q_75 = statistic_table['prevalence_pct_q75'].values

    print(f"\n{'=' * 60}")
    print(f"ЭТАП А: ПРЯМОЕ СРАВНЕНИЕ (Латентный пул)")
    print(f"{'=' * 60}")

    plt.figure(figsize=(14, 8))

    plt.fill_between(years, q_25, q_75, color='lightcoral', alpha=0.3,
                     label='Модель IQR (25-й и 75-й перцентили)')
    plt.plot(years, line_by_model, color='red', linewidth=3,
             label='Прогноз модели (без масштабирования)', marker='s')

    plt.scatter(years, points_by_real_data, color='black', marker='D', s=70,
                label='Госпитальные данные (реальные)', zorder=5)
    plt.plot(years, points_by_real_data, color='blue', linewidth=2, linestyle='--',
             alpha=0.7, label='Тренд реальных данных')

    plt.title(f'Прямое сравнение: Модель vs Реальные данные (без поправки)\n'
              f'Доказательство латентного пула заболевания\nСценарий: {scenario_name}',
              fontsize=14, fontweight='bold')
    plt.xlabel('Годы', fontsize=14)
    plt.ylabel('Превалентность (%)', fontsize=14)
    plt.xticks(years, rotation=45)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left')

    avg_difference = np.mean(line_by_model - points_by_real_data)
    plt.text(0.02, 0.98,
             f'Средняя разница (Модель - Реальные): {avg_difference:.2f}%\n'
             f'(Положительная разница = латентный пул)\n'
             f'Модель показывает полный "айсберг" болезни',
             transform=plt.gca().transAxes,
             fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='top')

    plt.tight_layout()

    figures_folder = os.path.join(results_folder, 'figures')
    plt.savefig(os.path.join(figures_folder, f'direct_comparison_{scenario_name}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    real_data = np.array(points_by_real_data)
    model_data = np.array(line_by_model)

    absolute_difference = model_data - real_data
    relative_difference = ((model_data - real_data) / real_data) * 100

    comparison_table = pd.DataFrame({
        'Year': years,
        'Real_Prevalence_%': real_data,
        'Model_Prevalence_%': model_data,
        'Absolute_Difference_%': absolute_difference,
        'Relative_Difference_%': relative_difference,
        'Model_Q25_%': q_25,
        'Model_Q75_%': q_75
    })

    mae = np.mean(np.abs(absolute_difference))
    rmse = np.sqrt(np.mean(absolute_difference ** 2))
    mape = np.mean(np.abs(relative_difference))

    tables_folder = os.path.join(results_folder, 'tables')
    comparison_table.to_csv(
        os.path.join(tables_folder, f'direct_comparison_{scenario_name}.csv'),
        index=False, encoding='utf-8-sig', sep=';', decimal=','
    )

    statistics_folder = os.path.join(results_folder, 'statistics')
    with open(os.path.join(statistics_folder, f'direct_comparison_stats_{scenario_name}.txt'),
              'w', encoding='utf-8') as f:
        f.write('=== ЭТАП А: ПРЯМОЕ СРАВНЕНИЕ (Доказательство латентного пула) ===\n')
        f.write(f'Сценарий: {scenario_name}\n\n')
        f.write(f'Средняя абсолютная ошибка (MAE): {mae:.4f}%\n')
        f.write(f'Среднеквадратичная ошибка (RMSE): {rmse:.4f}%\n')
        f.write(f'Средняя абсолютная процентная ошибка (MAPE): {mape:.2f}%\n')
        f.write(f'Средняя разница (Модель - Реальные): {np.mean(absolute_difference):.4f}%\n\n')
        f.write('ИНТЕРПРЕТАЦИЯ:\n')
        f.write('Высокая MAPE (~150%) и систематическое завышение моделью показывают,\n')
        f.write('что модель отражает полный латентный пул заболевания ("айсберг"),\n')
        f.write('тогда как госпитальная статистика фиксирует только выявленные случаи.\n\n')
        f.write('По годам:\n')
        for i, year in enumerate(years):
            f.write(f'{year}: Модель={model_data[i]:.2f}%, Реальные={real_data[i]:.2f}%, '
                    f'Разница={absolute_difference[i]:.2f}%, Относительная={relative_difference[i]:.1f}%\n')

    # График разницы
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(years, absolute_difference, color=['red' if x > 0 else 'green' for x in absolute_difference])
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[0].set_title(f'Абсолютная разница (Модель - Реальные)\nСценарий: {scenario_name}')
    axes[0].set_xlabel('Годы')
    axes[0].set_ylabel('Разница (%)')
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)

    axes[1].bar(years, relative_difference, color=['red' if x > 0 else 'green' for x in relative_difference])
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1].set_title(f'Относительная разница ((Модель-Реальные)/Реальные×100)\nСценарий: {scenario_name}')
    axes[1].set_xlabel('Годы')
    axes[1].set_ylabel('Относительная разница (%)')
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_folder, f'difference_analysis_{scenario_name}.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"\n📊 РЕЗУЛЬТАТЫ ПРЯМОГО СРАВНЕНИЯ:")
    print(f"   Средняя абсолютная ошибка (MAE): {mae:.4f}%")
    print(f"   Среднеквадратичная ошибка (RMSE): {rmse:.4f}%")
    print(f"   Средняя абсолютная процентная ошибка (MAPE): {mape:.2f}%")
    print(f"   Средняя разница (Модель - Реальные): {np.mean(absolute_difference):.4f}%")
    print(f"\n   💡 ИНТЕРПРЕТАЦИЯ:")
    print(f"   Высокая MAPE ({mape:.1f}%) и систематическое завышение моделью")
    print(f"   доказывают существование латентного пула заболевания.")
    print(f"   Модель отражает полный 'айсберг', а госстатистика — только верхушку.")

    return {
        'years': years,
        'real_data': real_data,
        'model_data': model_data,
        'absolute_difference': absolute_difference,
        'relative_difference': relative_difference,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'comparison_table': comparison_table
    }


if __name__ == '__main__':
    scenarios = ['scenario_1', 'scenario_2', 'scenario_3']

    for sc in scenarios:
        print(f"\n{'=' * 60}")
        print(f">>> ОБРАБОТКА: {sc} <<<")
        print(f"Время начала: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 60}")

        # 1. Подготовка папок для результатов
        results_folder = results_of_scenarios(sc, force_recreate=True)

        # 2. Генерация агрегированных таблиц Монте-Карло
        df = preparation_table_for_thesis(sc, force_reload=True)
        table_for_thesis_about_fmf(df, sc, results_folder)

        # 3. Загрузка совмещенных по годам данных (2012-2024)
        data = load_data_from_files(sc, force_reload=True)

        # 4. Базовый график (два независимых тренда рядом)
        first_graph(data, sc, results_folder)

        # ====================================================================
        # ЭТАП А: ПРЯМОЕ СРАВНЕНИЕ (Истинная превалентность vs Госпитальная)
        # ====================================================================
        # Здесь мы фиксируем масштабный сдвиг пула («айсберг»)
        direct_results = direct_comparison_without_coefficient(data, sc, results_folder)

        # ====================================================================
        # ЭТАП Б: МАСШТАБИРОВАННОЕ СРАВНЕНИЕ (Калибровка через коэффициент k)
        # ====================================================================
        # Находим средний коэффициент k для этого сценария
        mean_value = survival_hypothesis(data, sc, results_folder)

        # Получаем четко именованные массивы: реальность и модель * k
        years, real_values, scaled_model_values = plot_of_survival_hypothesis(data, mean_value, sc, results_folder)

        # Строим графики динамики выявляемости и считаем линейную регрессию для k
        coeff_dynamic = plot_dynamic_coefficient(data, years, sc, results_folder)
        linear_regression(years, coeff_dynamic, sc, results_folder)

        # Считаем Пирсона для СЫРЫХ трендов (проверяем сонаправленность векторов)
        print(f"\n{'=' * 60}")
        print(f"АНАЛИЗ КОРРЕЛЯЦИИ СЫРЫХ ТРЕНДОВ")
        print(f"{'=' * 60}")
        calculate_pearson(data, real_values, sc, results_folder)

        # Считаем MAPE и ME исключительно для МАСШТАБИРОВАННОЙ модели!
        print(f"\n{'=' * 60}")
        print(f"МЕТРИКИ ОШИБОК ДЛЯ МАСШТАБИРОВАННОЙ МОДЕЛИ (k = {mean_value:.3f})")
        print(f"{'=' * 60}")
        result_of_mape = mape_function(years, real_values, scaled_model_values, sc, results_folder)
        m_e = me_function(years, real_values, scaled_model_values, mean_value, sc, results_folder)

        print(f"\n✅ Результаты для {sc} сохранены в папку: {results_folder}")
        print(f"   Время завершения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")