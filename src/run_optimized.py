import hashlib
import random
import time
from typing import Union, Dict, Any, List
import pandas as pd
import numpy as np

from GenerationSimulation import GenerationSimulation
from ModelParams import ModelParams

_MAX_CACHE_SIZE = 128
_CACHE_VERSION = "v3"  # Увеличивайте при изменении логики симуляции


def run_single_simulation_optimized(run_id: Union[str, int],
                                    params: ModelParams,
                                    birth_rate_df,
                                    death_rate_df,
                                    tfr_df,
                                    age_structure_df,
                                    fertility_factors_df,
                                    verbose: bool = False,
                                    use_cache: bool = True,
                                    force_recompute: bool = False) -> List[Dict[str, Any]]:
    """
    Запускает одну симуляцию с возможностью кэширования.

    Args:
        force_recompute: Если True - игнорирует кэш и выполняет свежий расчет
    """
    # 1. Уникальный сид для каждого потока/прогона
    if isinstance(run_id, int):
        seed_val = run_id
    else:
        seed_val = int(hashlib.md5(str(run_id).encode()).hexdigest(), 16) % (2 ** 32)

    random.seed(seed_val)
    np.random.seed(seed_val)

    # 2. Кэширование
    from caches import _SIMULATION_CACHE, _CACHE_LOCK

    cache_key = None
    if use_cache and not force_recompute:
        cache_key = get_cache_key(params, birth_rate_df, death_rate_df, run_id)

        with _CACHE_LOCK:
            if cache_key in _SIMULATION_CACHE:
                if verbose:
                    print(f"🔄 Извлечено из кэша: {run_id}")
                # Возвращаем глубокую копию
                import copy
                return copy.deepcopy(_SIMULATION_CACHE[cache_key])

    # 3. Инициализация и запуск
    sim = GenerationSimulation(
        params=params,
        birth_rate_df=birth_rate_df,
        death_rate_df=death_rate_df,
        fertility_rate_df=tfr_df,
        age_structure_df=age_structure_df,
        fertility_factors_df=fertility_factors_df
    )

    # Выполняем симуляцию
    yearly_results = sim.run_simulation_with_calibration(run_id=run_id)

    # Добавляем метаданные для последующей агрегации
    for entry in yearly_results:
        entry['scenario'] = params.__class__.__name__
        entry['run_id'] = run_id
        entry['cache_version'] = _CACHE_VERSION
        entry['timestamp'] = time.time()

    # 4. Сохранение в кэш (только если не force_recompute)
    if use_cache and cache_key and not force_recompute:
        with _CACHE_LOCK:
            # Ограничиваем размер кэша
            if len(_SIMULATION_CACHE) >= _MAX_CACHE_SIZE:
                # Удаляем самый старый элемент
                oldest_key = next(iter(_SIMULATION_CACHE))
                del _SIMULATION_CACHE[oldest_key]
            _SIMULATION_CACHE[cache_key] = yearly_results

    if verbose:
        print(f"✅ Прогон {run_id} завершен успешно.")

    return yearly_results


def get_cache_key(params: ModelParams, birth_rate_df, death_rate_df, run_id=None) -> int:
    """Генерирует уникальный ключ для кэша с учетом версии"""
    scenario_kind = params.__class__.__name__

    # Берем только основные параметры (исключаем изменяемые во время выполнения)
    p_dict = vars(params).copy()
    # Удаляем потенциально проблемные поля
    p_dict.pop('_instance', None)

    param_str = f"{scenario_kind}_{_CACHE_VERSION}_"
    param_str += str(sorted(p_dict.items()))

    # Хэш данных (только если датафреймы не изменились)
    try:
        birth_hash = pd.util.hash_pandas_object(birth_rate_df).sum()
        death_hash = pd.util.hash_pandas_object(death_rate_df).sum()
        data_hash = f"{birth_hash}_{death_hash}"
    except Exception:
        # Если не удалось захэшировать, используем время
        data_hash = str(int(time.time() / 3600))  # Меняется каждый час

    if run_id is not None:
        param_str += f"_run_{run_id}"

    combined = f"{param_str}_{data_hash}"
    return int(hashlib.md5(combined.encode()).hexdigest(), 16) % (2 ** 32)


def clear_simulation_cache(verbose: bool = True):
    """Принудительная очистка всего кэша симуляций"""
    from caches import _SIMULATION_CACHE, _CACHE_LOCK
    with _CACHE_LOCK:
        cache_size = len(_SIMULATION_CACHE)
        _SIMULATION_CACHE.clear()
        if verbose:
            print(f"🧹 Очищен кэш симуляций ({cache_size} записей)")


def get_cache_stats() -> Dict[str, Any]:
    """Возвращает статистику кэша"""
    from caches import _SIMULATION_CACHE, _CACHE_LOCK
    with _CACHE_LOCK:
        return {
            'size': len(_SIMULATION_CACHE),
            'max_size': _MAX_CACHE_SIZE,
            'version': _CACHE_VERSION,
            'keys': list(_SIMULATION_CACHE.keys())[:10]  # Первые 10 ключей
        }