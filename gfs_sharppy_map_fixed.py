import os
import sys
import ssl
import copy
import time
import threading
import multiprocessing as mp
import datetime
import requests
import numpy as np

# --- Отключаем строгую SSL-проверку для скачивания карт Cartopy ---
ssl._create_default_https_context = ssl._create_unverified_context

import xarray as xr
import cfgrib

import matplotlib
matplotlib.use('Agg')  # Для предотвращения ошибок GUI в разных потоках
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# --- Импорты SHARPpy ---
from sharppy.sharptab import profile, interp, utils
import sharppy.sharptab.params as params
import sharppy.sharptab.thermo as thermo

try:
    import sharppy.sharptab.winds as wind
except ImportError:
    try:
        from sharppy.sharptab import wind
    except ImportError:
        import sharppy.sharptab.wind as wind


# =====================================================================
#  OUTLOOK-ДВИЖОК
#
#  Устройство (почему так, а не набор if-ов):
#
#  1. НЕЧЁТКИЕ МЕМБЕРШИПЫ вместо жёстких порогов. STP=0.99 и STP=1.01 —
#     физически одно и то же, а жёсткий порог рисует между ними границу
#     категории. Поэтому каждый ингредиент переводится в непрерывную
#     величину 0..1, а категория получается округлением в самом конце.
#
#  2. БАЗА ОТ ЗАДАННЫХ ПОРОГОВ, остальное — модификаторы. Пороги STP
#     (0-1-2) и DCAPE заданы пользователем и остаются главными; прочие
#     ингредиенты только двигают результат на доли категории, а не
#     переопределяют его. Так логика остаётся предсказуемой.
#
#  3. РЕЖИМ КОНВЕКЦИИ решает, какие явления вообще возможны. При
#     EBWD<12 м/с суперячейка не живёт, сколько бы ни было CAPE —
#     значит торнадо-риск там ограничен сверху независимо от STP.
#
#  4. СОГЛАСОВАННОСТЬ. Если высокий балл держится на одном-единственном
#     ингредиенте, а остальные молчат — это чаще артефакт модели, чем
#     реальная угроза. Такая точка не поднимается выше 2-й категории.
#
#  5. ТРИГГЕР СЧИТАЕТСЯ ОТДЕЛЬНО и НЕ понижает потенциал (по условию:
#     "риск есть, а будет ли триггер — смотрите сами"). Это отдельный
#     слой: convective_temp против прогнозной максимальной температуры.
# =====================================================================

_EPS = 0.02  # нижний пол для геом. среднего, чтобы ln(0) не ломал расчёт
KTS_TO_MS_LOCAL = 0.514444

# ---------------------------------------------------------------------
#  ПОРОГИ: свои или американские
#
#  По умолчанию стоят константы из практики SPC. Для Якутии они смещены:
#  климатология ERA5 показала, что порог SigSevere «Локальный» = 8000
#  срабатывает здесь примерно раз в сезон, тогда как эта категория
#  должна означать «бывает регулярно». Одновременно DCAPE-порог 300
#  занижен втрое — сухой континентальный воздух даёт сильный нисходящий
#  поток даже без большой неустойчивости.
#
#  Если рядом лежит yakutia_thresholds.json (его создаёт
#  era5_clim_thresholds.py), пороги берутся оттуда. Пара значений —
#  начало 2-й категории и начало 4-й; между ними переход плавный.
# ---------------------------------------------------------------------
CLIMO = {
    'sig_severe': (8000.0, 32000.0),
    'dcape':      (300.0, 1100.0),
    'wndg':       (0.4, 1.8),
    'ship':       (0.4, 1.8),
    'mburst':     (3.0, 10.0),
    'dcp':        (0.5, 3.0),
    'mmp':        (0.4, 0.9),
    'sherb':      (0.7, 1.6),
    'srh1_support': 120.0,
    'pwat_dry':   16.0,
}
CLIMO_SOURCE = "пороги SPC (США), климатология не подключена"

#: Эталонные пороги SPC — то, с чем сравниваем. Держим отдельно, чтобы
#: можно было вернуться к ним одним переключателем, а не перезаписью файла.
CLIMO_SPC = {
    'sig_severe': (8000.0, 32000.0),
    'dcape':      (300.0, 1100.0),
    'wndg':       (0.4, 1.8),
    'ship':       (0.4, 1.8),
    'mburst':     (3.0, 10.0),
    'dcp':        (0.5, 3.0),
    'mmp':        (0.4, 0.9),
    'sherb':      (0.7, 1.6),
    'srh1_support': 120.0,
    'pwat_dry':   16.0,
}

#: Наборы порогов. Ключ — то, что видно в выпадающем списке.
#: Файлы ищутся рядом со скриптом и в ~/ERA5_Climatology.
#:
#: Регионы заведены отдельными наборами, потому что климат в них разный:
#: юго-запад Якутии (где и случились все пять задокументированных
#: смерчей) заметно теплее и влажнее центральной части, а северные
#: улусы — принципиально другой режим. Один набор на всю республику
#: смазывал бы эту разницу.
CLIMO_SETS = {
    "SPC (США) — эталон":        None,
    "Якутия — центр":            "yakutia_thresholds.json",
    "Якутия — юго-запад":        "yakutia_sw_thresholds.json",
    "Якутия — север":            "yakutia_north_thresholds.json",
    "Свой файл...":              "__custom__",
}


def _find_climo_file(name):
    """Ищет файл набора рядом со скриптом и в папке климатологии."""
    here = os.path.dirname(os.path.abspath(__file__))
    for c in (os.path.join(here, name),
              os.path.join(os.path.expanduser("~"), "ERA5_Climatology", name)):
        if os.path.exists(c):
            return c
    return name          # пусть загрузчик сам сообщит, что не нашёл


def use_spc_thresholds():
    """Возвращает движок к порогам SPC — для сравнения или для расчётов
    вне Якутии, где местная климатология неприменима."""
    global CLIMO_SOURCE
    CLIMO.update(CLIMO_SPC)
    CLIMO_SOURCE = "пороги SPC (США)"


def load_climo_thresholds(path=None):
    """
    Подхватывает пороги из yakutia_thresholds.json, если он найден.
    Ищет рядом со скриптом и в ~/ERA5_Climatology. Молча остаётся на
    значениях по умолчанию, если файла нет — тогда карты считаются как
    раньше, по американским порогам.
    """
    global CLIMO_SOURCE
    candidates = [path] if path else []
    here = os.path.dirname(os.path.abspath(__file__))
    candidates += [
        os.path.join(here, "yakutia_thresholds.json"),
        os.path.join(os.path.expanduser("~"), "ERA5_Climatology",
                     "yakutia_thresholds.json"),
    ]
    for c in candidates:
        if not c or not os.path.exists(c):
            continue
        try:
            import json
            with open(c, encoding='utf-8') as f:
                data = json.load(f)
            # Перед загрузкой ВОЗВРАЩАЕМ всё к SPC. Иначе при смене набора
            # величины, которых нет в новом файле, остались бы от прежнего —
            # и получилась бы смесь двух климатологий, о которой никто
            # не догадывается.
            CLIMO.update(CLIMO_SPC)

            th = data.get('thresholds', {})
            # ключ в JSON -> ключ в CLIMO. Список расширен под файлы v2,
            # где считаются и WNDG с MMP, и прочие некалиброванные раньше.
            for src, dst in (('sigsvr', 'sig_severe'), ('dcape', 'dcape'),
                             ('ship', 'ship'), ('wndg', 'wndg'),
                             ('mmp', 'mmp'), ('dcp', 'dcp'),
                             ('mburst', 'mburst'), ('sherb', 'sherb')):
                row = th.get(src)
                if row and 'Локальный' in row and 'Критический' in row:
                    CLIMO[dst] = (float(row['Локальный']), float(row['Критический']))
            row = th.get('srh1')
            if row and 'Локальный' in row:
                CLIMO['srh1_support'] = float(row['Локальный'])
            gate = (data.get('gates') or {}).get('pwat_dry')
            if gate is not None:
                CLIMO['pwat_dry'] = float(gate)
            months = data.get('months')
            hours = data.get('hours')
            n_th = sum(1 for k in th if k in
                       ('sigsvr', 'dcape', 'ship', 'wndg', 'mmp', 'dcp',
                        'mburst', 'sherb', 'srh1'))
            CLIMO_SOURCE = (f"{os.path.basename(c)}: {n_th} порогов, "
                            f"месяцы {months}, сроки {hours}")
            return True
        except Exception:
            continue
    return False


load_climo_thresholds()


def _ramp_up(x, lo, hi):
    """0 при x<=lo, 1 при x>=hi, линейно между. None/NaN -> 0."""
    if x is None:
        return 0.0
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(x):
        return 0.0
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    return float(min(1.0, max(0.0, (x - lo) / (hi - lo))))


def _ramp_down(x, good, bad):
    """1 при x<=good, 0 при x>=bad (для величин, где меньше — лучше: LCL)."""
    if x is None:
        return 0.0
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(x):
        return 0.0
    if bad <= good:
        return 1.0 if x <= good else 0.0
    return float(min(1.0, max(0.0, (bad - x) / (bad - good))))


def _safe(val, default=0.0):
    """Приводит возможный masked/NaN/None к обычному float."""
    try:
        if val is None:
            return default
        if np.ma.is_masked(val):
            return default
        f = float(val)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _classify_mode(mucape, ebwd_ms, esrh, shear6_ms):
    """
    Режим конвекции. Определяет, какие явления в принципе возможны,
    и служит потолком для торнадо-риска.

    Возвращает один из: 'none', 'pulse', 'multicell', 'hslc', 'supercell'.
    """
    deep_shear = max(_safe(ebwd_ms), _safe(shear6_ms))
    if mucape < 100:
        return 'none'
    # HSLC проверяется ПЕРВЫМ: при MUCAPE<=1000 и сильном сдвиге формальные
    # признаки суперячейки выполняются, но физика там другая (низкотопные
    # шторма, слабые смерчи), и обычные композиты её недооценивают.
    if mucape <= 1000 and deep_shear >= 15:
        return 'hslc'
    if deep_shear >= 18 and esrh >= 100 and mucape >= 300:
        return 'supercell'
    if deep_shear >= 12:
        return 'multicell'
    return 'pulse'


#: Потолок торнадо-категории для каждого режима.
_MODE_TOR_CAP = {
    'none': 1.0,
    'pulse': 2.0,       # без глубокого сдвига мезоциклон не удержится
    'multicell': 3.0,
    'hslc': 3.5,        # QLCS-торнадо реальны, но обычно слабые
    'supercell': 4.0,
}


def _tornado_category(stp, srh1k, srh3k, esrh, mllcl, mlcape,
                      ebwd_ms, shear6_ms, lr0_3, mode):
    """
    Непрерывная торнадо-категория 1..4.

    Каркас — заданные пороги STP (0-1 / 1-2 / >2), внутри каждой полосы
    значение меняется плавно. Дальше — модификаторы по остальным
    ингредиентам. Возвращает (категория, словарь диагностики).
    """
    stp = _safe(stp)

    # --- каркас: полосы STP, непрерывные внутри полосы ---
    if stp <= 0:
        cat = 1.0
    elif stp <= 1:
        cat = 2.0 + 0.99 * (stp / 1.0)
    elif stp <= 2:
        cat = 3.0 + 0.99 * ((stp - 1.0) / 1.0)
    else:
        cat = 4.0

    notes = []

    # --- проверка вращения для верхней категории ---
    # STP может быть большим за счёт одного лишь CAPE; без реального
    # вращения в нижних слоях это не торнадная обстановка.
    srh_ok = (_safe(srh1k) >= CLIMO['srh1_support']) or (100 <= _safe(srh3k) <= 150) \
        or (_safe(esrh) >= 100)
    if cat >= 4.0 and not srh_ok:
        cat -= 1.0
        notes.append("нет поддержки по SRH")

    # --- глубокий сдвиг: без него мезоциклон не организуется ---
    deep = max(_safe(ebwd_ms), _safe(shear6_ms))
    if deep < 12 and cat > 2.0:
        cat -= 1.0
        notes.append("слабый глубокий сдвиг")
    elif deep < 15 and cat > 2.0:
        cat -= 0.5
        notes.append("умеренный глубокий сдвиг")

    # --- высота базы облака ---
    lcl = _safe(mllcl, default=9999.0)
    if lcl > 1500:
        cat -= 0.5
        notes.append("высокая база (LCL>1500м)")
    elif lcl < 800 and cat > 1.5:
        cat += 0.25
        notes.append("низкая база (LCL<800м)")

    # --- растяжение вихря в нижнем слое (крутой градиент 0-3 км) ---
    if _safe(lr0_3) >= 7.5 and cat > 1.5:
        cat += 0.25
        notes.append("крутой градиент 0-3км")

    # --- потолок по режиму конвекции ---
    mode_cap = _MODE_TOR_CAP.get(mode, 4.0)
    if cat > mode_cap:
        cat = mode_cap
        if mode_cap < 4.0:
            notes.append("ограничено режимом: %s" % mode)

    # --- согласованность: сколько независимых ингредиентов "за" ---
    votes = sum([
        stp > 0.5,
        _safe(srh1k) >= 120,
        deep >= 15,
        _safe(mlcape) >= 250,
        lcl <= 1200,
    ])
    if votes <= 1 and cat > 2.0:
        cat = 2.0
        notes.append("сигнал только по одному ингредиенту")

    diag = {'stp': stp, 'srh1k': _safe(srh1k), 'esrh': _safe(esrh),
            'lcl': lcl, 'deep_shear': deep, 'mode': mode,
            'votes': votes, 'notes': notes}
    return cat, diag


def _overall_category(sig_severe, dcape, wndg, dcp, mburst, ship, sherb, mmp,
                      mucape, mlcape, lr7_5, mllcl, ebwd_ms, shear6_ms, mode):
    """
    Непрерывная overall-категория 1..4.

    Разные явления (шквал / град / долгоживущий MCS / HSLC) — это
    АЛЬТЕРНАТИВЫ, поэтому между ними берётся максимум, а не произведение:
    достаточно одного реализовавшегося сценария.
    Внутри каждого сценария ингредиенты, наоборот, взаимозависимы.
    """
    notes = []
    sub = {}

    # ---------- Шквалистый ветер ----------
    w = 1.0
    w = max(w, 1.0 + 3.0 * _ramp_up(_safe(sig_severe), *CLIMO['sig_severe']))
    w = max(w, 1.0 + 3.0 * _ramp_up(_safe(dcape), *CLIMO['dcape']))
    w = max(w, 1.0 + 3.0 * _ramp_up(_safe(wndg), *CLIMO['wndg']))
    # DCP — композит под организованный ветровой шторм (деречо)
    w = max(w, 1.0 + 3.0 * _ramp_up(_safe(dcp), *CLIMO['dcp']))
    sub['ветер'] = w

    # ---------- Микропорыв / сухая гроза ----------
    # Классический композит + собственная поправка на высокую базу:
    # чем выше LCL при наличии неустойчивости, тем сильнее сухой нисходящий
    # поток. Работает и там, где STP молчит.
    mb = 1.0 + 3.0 * _ramp_up(_safe(mburst), *CLIMO['mburst'])
    if _safe(mlcape) >= 150:
        mb = max(mb, 1.0 + 2.2 * _ramp_up(_safe(mllcl), 900, 2000)
                          * _ramp_up(_safe(dcape), 200, 900))
    sub['микропорыв'] = mb

    # ---------- Град ----------
    # SHIP осмыслен только при реальной неустойчивости и крутом
    # среднетропосферном градиенте — иначе даёт ложные максимумы.
    h = 1.0
    if _safe(mucape) >= 400:
        h = 1.0 + 3.0 * _ramp_up(_safe(ship), *CLIMO['ship'])
        if _safe(lr7_5) < 6.0:
            h = min(h, 2.5)
            notes.append("пологий градиент 700-500 ограничивает град")
    sub['град'] = h

    # ---------- Долгоживущий MCS ----------
    # MMP — вероятность, что зрелый MCS сохранит интенсивность.
    m = 1.0
    if _safe(mucape) >= 500:
        m = 1.0 + 2.5 * _ramp_up(_safe(mmp), *CLIMO['mmp'])
    sub['MCS'] = m

    # ---------- HSLC ----------
    # Отдельная ветка: мало CAPE, много сдвига. Обычные композиты
    # (SigSvr, SHIP) здесь занижены по построению, поэтому SHERB.
    hs = 1.0
    if mode == 'hslc' or _safe(mucape) <= 1000:
        hs = 1.0 + 2.5 * _ramp_up(_safe(sherb), *CLIMO['sherb'])
    sub['HSLC'] = hs

    cat = max(sub.values())

    # --- режимный потолок: без организации нет продолжительных явлений ---
    if mode == 'pulse' and cat > 3.0:
        cat = 3.0
        notes.append("импульсный режим: ограничено")
    if mode == 'none':
        cat = 1.0
        notes.append("нет неустойчивости")

    # --- согласованность ---
    votes = sum([
        _safe(sig_severe) >= CLIMO['sig_severe'][0],
        _safe(dcape) >= CLIMO['dcape'][0],
        _safe(wndg) >= CLIMO['wndg'][0],
        _safe(ship) >= CLIMO['ship'][0],
        _safe(dcp) >= CLIMO['dcp'][0],
        _safe(mucape) >= 500,
    ])
    if votes <= 1 and cat > 2.0:
        cat = 2.0
        notes.append("сигнал только по одному ингредиенту")

    diag = {'sub': {k: round(v, 2) for k, v in sub.items()},
            'mode': mode, 'votes': votes, 'notes': notes}
    return cat, diag


def _trigger_category(conv_t, max_t, mlcin, mucape):
    """
    Отдельный слой: НАСКОЛЬКО РЕАЛЬНА ИНИЦИАЦИЯ.

    Не понижает потенциал — по условию задачи потенциал показывается
    независимо ("если рванёт, то рванёт"). Это отдельная информация:
    хватит ли одного дневного прогрева, или нужен фронт/динамика.

    1 — прогрева заведомо не хватает (нужен внешний подъём)
    2 — на грани, нужен дополнительный механизм
    3 — прогрева достаточно, инициация вероятна
    """
    if _safe(mucape) < 100:
        return 1.0
    ct = _safe(conv_t, default=None)
    mt = _safe(max_t, default=None)
    if ct is None or mt is None:
        # нет конвективной температуры — судим только по CIN
        cin = _safe(mlcin)
        if cin > -25:
            return 3.0
        if cin > -60:
            return 2.0
        return 1.0
    delta = ct - mt  # насколько прогрева НЕ хватает, °C
    if delta <= 0:
        return 3.0
    if delta <= 2.0:
        return 2.0
    return 1.0


#: Собственная четырёхступенчатая шкала. Оформление карт взято у SPC
#: (заливка полигонами, легенда в углу), но сами категории и их пороги
#: остались нашими: они выведены из климатологии Якутии, а не из
#: американской практики, и подменять их чужой шкалой значило бы
#: потерять всю проделанную калибровку.
OUTLOOK_NAMES_G = {1: "Фоновый", 2: "Локальный",
                   3: "Очаговый", 4: "Критический"}
OUTLOOK_LEVELS = [
    (1, "Фоновый",     "гроза маловероятна"),
    (2, "Локальный",   "отдельные обычные грозы"),
    (3, "Очаговый",    "сильные грозы в отдельных районах"),
    (4, "Критический", "высокая вероятность опасной грозовой погоды"),
]


def _apply_cin_gate(category, mlcin, sbcin):
    """
    MLCIN < -70 — экстремальная крышка, обнуляем риск полностью.
    MLCIN < -38 И SBCIN тоже < -38 — крышка почти везде, снижаем на категорию.
    Если хотя бы один из двух >= -38 — крышка "дырявая", не трогаем.
    """
    mlcin = _safe(mlcin)
    sbcin = _safe(sbcin)
    if mlcin < -70:
        return 1.0
    if mlcin < -38 and sbcin < -38:
        return max(1.0, category - 1.0)
    return category


def _apply_pwat_gate(category, pwat_mm):
    """
    Сухая обстановка — потолок 2-й категории.

    Порог берётся из климатологии (нижняя четверть по влагозапасу), а не
    зашитый. Зашитые 16 мм для Якутии оказались МЕДИАНОЙ — то есть гейт
    срабатывал бы в половине сроков и почти всегда срезал категорию.
    """
    if _safe(pwat_mm, default=99.0) < CLIMO['pwat_dry']:
        return min(category, 2.0)
    return category


def _enforce_spatial_coherence(grid, min_neighbors=2):
    """
    Пространственная связность (постобработка по всей сетке).

    Одиночный пик высокой категории посреди спокойного фона — почти
    всегда шум одной точки профиля, а не реальный очаг: настоящие
    очаги имеют размер. Точка 3-4 категории, у которой в окне 3x3
    меньше min_neighbors соседей сопоставимого уровня, понижается.

    Работает по убыванию категорий, чтобы понижение 4->3 могло
    повлечь понижение 3->2 на том же проходе.
    """
    out = grid.copy()
    n_lat, n_lon = out.shape
    for level in (4, 3):
        demote = []
        for i in range(n_lat):
            for j in range(n_lon):
                if out[i, j] < level:
                    continue
                i0, i1 = max(0, i - 1), min(n_lat, i + 2)
                j0, j1 = max(0, j - 1), min(n_lon, j + 2)
                window = out[i0:i1, j0:j1]
                support = int(np.sum(window >= level - 1)) - 1  # минус сама точка
                if support < min_neighbors:
                    demote.append((i, j))
        for i, j in demote:
            out[i, j] = level - 1
    return out


def _sharppy_point_worker(job):
    """
    Выполняется в ОТДЕЛЬНОМ ПРОЦЕССЕ (multiprocessing.Pool).
    ВАЖНО: функция обязана быть на верхнем уровне модуля (не методом класса) —
    иначе Windows (spawn) не сможет её сериализовать (pickle) для передачи
    в дочерние процессы. Никакого доступа к self/GUI/xarray-датасетам здесь
    нет и не должно быть — только чистые (picklable) numpy-массивы на входе.

    job = (i, j, lat, lon, pres, gh, tmp, dwpk, u, v, selected_params)
    Возвращает (i, j, results_dict, error_str_or_None).
    """
    i, j, lat, lon, pres, gh, tmp, dwpk, u, v, selected_params = job
    results = {}
    try:
        prof = profile.create_profile(
            profile='default',
            pres=pres,
            hght=gh,
            tmpc=tmp,
            dwpc=dwpk,
            wspd=np.hypot(u, v),
            wdir=(np.arctan2(-u, -v) * 180.0 / np.pi) % 360.0,
            missing=-9999
        )
        if prof is None:
            raise ValueError("profile.create_profile() вернул None")

        sfcpcl = params.parcelx(prof, flag=1)  # Surface-Based
        mlpcl = params.parcelx(prof, flag=4)   # Mixed-Layer
        mupcl = params.parcelx(prof, flag=3)   # Most-Unstable

        if selected_params.get('sbcape'):
            results['sbcape'] = max(0.0, sfcpcl.bplus)
        if selected_params.get('mlcape'):
            results['mlcape'] = max(0.0, mlpcl.bplus)
        # CIN отрицателен по определению (энергия торможения). Оставляем
        # знак как есть: он и означает силу крышки, а модуль ничего не
        # добавляет, зато путает при чтении карты.
        if selected_params.get('mlcin'):
            results['mlcin'] = min(0.0, _safe(mlpcl.bminus))
        if selected_params.get('sbcin'):
            results['sbcin'] = min(0.0, _safe(sfcpcl.bminus))
        if selected_params.get('mucape'):
            results['mucape'] = max(0.0, mupcl.bplus)
        if selected_params.get('cape3k'):
            results['cape3k'] = max(0.0, params.parcelx(
                prof, flag=1, pbot=prof.pres[0],
                ptop=interp.pres(prof, interp.to_msl(prof, 3000))
            ).bplus)

        if selected_params.get('shear1k'):
            shu, shv = wind.wind_shear(prof, pbot=prof.pres[0], ptop=interp.pres(prof, interp.to_msl(prof, 1000)))
            results['shear1k'] = np.hypot(shu, shv) * 0.514444
        if selected_params.get('shear3k'):
            shu, shv = wind.wind_shear(prof, pbot=prof.pres[0], ptop=interp.pres(prof, interp.to_msl(prof, 3000)))
            results['shear3k'] = np.hypot(shu, shv) * 0.514444
        if selected_params.get('shear6k'):
            shu, shv = wind.wind_shear(prof, pbot=prof.pres[0], ptop=interp.pres(prof, interp.to_msl(prof, 6000)))
            results['shear6k'] = np.hypot(shu, shv) * 0.514444

        need_stp_val = (selected_params.get('stp') or selected_params.get('outlook_tornado'))
        need_scp_stp_block = (selected_params.get('scp') or need_stp_val
                               or selected_params.get('outlook_overall'))
        need_bunkers = (selected_params.get('srh1k') or selected_params.get('srh3k')
                        or need_scp_stp_block)

        stp_val = 0.0
        srh1k_val = None
        srh3k_val = None
        eff_inflow = (np.ma.masked, np.ma.masked)
        effective_srh = None
        ebwspd = None

        if need_bunkers:
            rstu, rstv, lstu, lstv = wind.non_parcel_bunkers_motion(prof)
            if selected_params.get('srh1k') or selected_params.get('outlook_tornado'):
                srh1k_val = wind.helicity(prof, 0, 1000., stu=rstu, stv=rstv)[0]
                if selected_params.get('srh1k'):
                    results['srh1k'] = srh1k_val
            if selected_params.get('srh3k') or selected_params.get('outlook_tornado'):
                srh3k_val = wind.helicity(prof, 0, 3000., stu=rstu, stv=rstv)[0]
                if selected_params.get('srh3k'):
                    results['srh3k'] = srh3k_val

            if need_scp_stp_block:
                eff_inflow = params.effective_inflow_layer(prof)
                if not np.ma.is_masked(eff_inflow[0]) and not np.ma.is_masked(eff_inflow[1]):
                    ebot_hght = interp.to_agl(prof, interp.hght(prof, eff_inflow[0]))
                    etop_hght = interp.to_agl(prof, interp.hght(prof, eff_inflow[1]))
                    effective_srh = wind.helicity(prof, ebot_hght, etop_hght, stu=rstu, stv=rstv)[0]

                    ebwd_u, ebwd_v = wind.wind_shear(prof, pbot=eff_inflow[0], ptop=eff_inflow[1])
                    ebwspd = np.hypot(ebwd_u, ebwd_v)

                    if selected_params.get('scp'):
                        results['scp'] = max(0.0, params.scp(mupcl.bplus, effective_srh, ebwspd))

                    stp_val = max(0.0, params.stp_cin(
                        mlpcl.bplus, effective_srh, ebwspd, mlpcl.lclhght, mlpcl.bminus
                    ))
                    if selected_params.get('stp'):
                        results['stp'] = stp_val
                # Нет эффективного слоя притока — stp_val/effective_srh остаются
                # как заданы выше (0.0/None), это физически корректный итог.

        mlcin = mlpcl.bminus
        sbcin = sfcpcl.bminus

        want_tor = selected_params.get('outlook_tornado')
        want_all = selected_params.get('outlook_overall')
        want_trg = selected_params.get('outlook_trigger')

        if want_tor or want_all or want_trg:
            # ---- общий набор диагностических величин ----
            # Всё в try/except: на вырожденных профилях (мало уровней,
            # нет эффективного слоя) отдельные композиты SHARPpy кидают
            # исключения, и это нормальная ситуация, а не ошибка точки.
            def _try(fn, default=None):
                try:
                    return fn()
                except Exception:
                    return default

            pwat_mm = _try(lambda: params.precip_water(prof) * 25.4, None)
            lr0_3 = _try(lambda: params.lapse_rate(prof, 0, 3000, pres=False), None)
            lr7_5 = _try(lambda: params.lapse_rate(prof, 700, 500, pres=True), None)
            shear6_ms = _try(
                lambda: np.hypot(*wind.wind_shear(
                    prof, pbot=prof.pres[0],
                    ptop=interp.pres(prof, interp.to_msl(prof, 6000)))) * 0.514444,
                None)
            ebwd_ms = (ebwspd * 0.514444) if ebwspd is not None else None

            mode = _classify_mode(_safe(mupcl.bplus),
                                  _safe(ebwd_ms), _safe(effective_srh),
                                  _safe(shear6_ms))

            if want_tor:
                cat, diag = _tornado_category(
                    stp_val, srh1k_val, srh3k_val, effective_srh,
                    mlpcl.lclhght, mlpcl.bplus, ebwd_ms, shear6_ms, lr0_3, mode)
                cat = _apply_cin_gate(cat, mlcin, sbcin)
                cat = _apply_pwat_gate(cat, pwat_mm)
                results['outlook_tornado'] = float(min(4.0, max(1.0, cat)))
                if cat >= 2.5:
                    results.setdefault('_diag', {})['tornado'] = diag

            if want_all:
                sig_severe_val = _try(lambda: params.sig_severe(prof, mlpcl=mlpcl), 0.0)
                dcape_val = _try(lambda: params.dcape(prof)[0], 0.0)
                wndg_val = _try(lambda: params.wndg(prof, mlpcl=mlpcl), 0.0)
                dcp_val = _try(lambda: params.dcp(prof), 0.0)
                mburst_val = _try(lambda: params.mburst(prof), 0.0)
                ship_val = _try(lambda: params.ship(prof, mupcl=mupcl), 0.0)
                mmp_val = _try(lambda: params.mmp(prof), 0.0)

                sherb_val = None
                if not np.ma.is_masked(eff_inflow[0]) and not np.ma.is_masked(eff_inflow[1]):
                    sherb_val = _try(lambda: params.sherb(
                        prof, effective=True, ebottom=eff_inflow[0],
                        etop=eff_inflow[1], mupcl=mupcl), None)
                if sherb_val is None:
                    sherb_val = _try(lambda: params.sherb(prof), None)

                cat, diag = _overall_category(
                    sig_severe_val, dcape_val, wndg_val, dcp_val, mburst_val,
                    ship_val, sherb_val, mmp_val,
                    mupcl.bplus, mlpcl.bplus, lr7_5, mlpcl.lclhght,
                    ebwd_ms, shear6_ms, mode)
                cat = _apply_cin_gate(cat, mlcin, sbcin)
                cat = _apply_pwat_gate(cat, pwat_mm)
                results['outlook_overall'] = float(min(4.0, max(1.0, cat)))
                if cat >= 2.5:
                    results.setdefault('_diag', {})['overall'] = diag

            if want_trg:
                conv_t = _try(lambda: params.convective_temp(prof), None)
                max_t = _try(lambda: params.max_temp(prof), None)
                results['outlook_trigger'] = float(
                    _trigger_category(conv_t, max_t, mlcin, mupcl.bplus))

        return (i, j, results, None)
    except Exception as e:
        return (i, j, {}, f"{type(e).__name__}: {e}")


def _dewpoint_from_rh(t_c, rh_pct):
    """
    Точка росы из температуры и относительной влажности по формуле Магнуса.

    ЗАЧЕМ ОТДЕЛЬНАЯ ФУНКЦИЯ: раньше здесь стояло "правило большого пальца"
    Td = T - (100-RH)/5. Оно сносно работает только у насыщения, а в сухом
    воздухе завышает точку росы вплоть до +9 °C, причём ВСЕГДА в одну
    сторону — атмосфера выглядит влажнее, чем есть. Поскольку CAPE очень
    чувствителен к влажности нижних слоёв, это давало систематически
    завышенный CAPE относительно нормально считающих источников.

    Работает и со скалярами, и с numpy-массивами.
    """
    A, B = 17.625, 243.04
    rh = np.clip(np.asarray(rh_pct, dtype=float), 0.1, 100.0)
    t = np.asarray(t_c, dtype=float)
    gamma = np.log(rh / 100.0) + (A * t) / (B + t)
    td = (B * gamma) / (A - gamma)
    return np.minimum(td, t)  # точка росы не может превышать температуру


_YAKUTIA_GEOM = None
_YAKUTIA_TRIED = False


def get_yakutia_geometry():
    """
    Полигон Республики Саха из Natural Earth (административные единицы
    первого уровня, масштаб 10 млн). Cartopy скачивает файл сам при
    первом обращении и кладёт в свой кэш, дальше берёт с диска.

    Возвращает геометрию либо None, если найти не удалось — тогда карта
    просто рисуется без маски, а не падает.
    """
    global _YAKUTIA_GEOM, _YAKUTIA_TRIED
    if _YAKUTIA_TRIED:
        return _YAKUTIA_GEOM
    _YAKUTIA_TRIED = True
    try:
        import cartopy.io.shapereader as shpreader
        from shapely.ops import unary_union
        path = shpreader.natural_earth(resolution='10m', category='cultural',
                                       name='admin_1_states_provinces')
        parts = []
        for rec in shpreader.Reader(path).records():
            a = rec.attributes
            # В разных версиях Natural Earth республика подписана
            # по-разному: Sakha, Saha, Yakutia, Yakutiya. Ищем по корню,
            # заодно проверяя, что это Россия.
            names = " ".join(str(a.get(k, "")) for k in
                             ('name', 'name_en', 'name_alt', 'woe_name', 'gn_name'))
            country = str(a.get('admin', '')) + str(a.get('iso_a2', ''))
            if ('RU' in country or 'Russia' in country) and \
                    any(t in names for t in ('Sakha', 'Saha', 'Yakut')):
                parts.append(rec.geometry)
        if parts:
            _YAKUTIA_GEOM = unary_union(parts)
    except Exception:
        _YAKUTIA_GEOM = None
    return _YAKUTIA_GEOM


def compute_fronts(t_k, u_ms, v_ms, lat_c, lon_c, level_hpa=850.0):
    """
    Объективный поиск фронтов по полю на уровне давления.

    Метод — Thermal Front Parameter (Renard & Clarke): TFP показывает,
    где градиент потенциальной температуры сам резко меняется, то есть
    находит ТЁПЛУЮ КРОМКУ бароклинной зоны. Именно её и рисуют как
    линию фронта на синоптических картах.

        TFP = -∇|∇θ| · (∇θ / |∇θ|)

    Тип фронта определяется знаком адвекции температуры на том же
    уровне: натекает холод — холодный фронт, тепло — тёплый.

    Возвращает (tfp, grad_mag, adv) на той же сетке. Все три нужны:
    по TFP ищется положение линии, по grad_mag отсекаются слабые
    зоны, по adv красится тип.
    """
    from scipy.ndimage import gaussian_filter

    # Потенциальная температура: она, в отличие от обычной, не меняется
    # при вертикальных смещениях и потому годится для поиска фронтов.
    theta = np.asarray(t_k, dtype=float) * (1000.0 / level_hpa) ** 0.2854
    # Сглаживание обязательно: на сырой сетке производные второго
    # порядка — сплошной шум.
    theta = gaussian_filter(theta, sigma=1.4)

    # Шаг сетки в метрах. Долготный шаг сжимается с широтой.
    dlat = float(np.mean(np.diff(lat_c)))
    dlon = float(np.mean(np.diff(lon_c)))
    dy = dlat * 111320.0
    dx = dlon * 111320.0 * np.cos(np.radians(lat_c))[:, None]

    dth_dy, dth_dx = np.gradient(theta, axis=0), np.gradient(theta, axis=1)
    dth_dy = dth_dy / dy
    dth_dx = dth_dx / dx

    grad_mag = np.hypot(dth_dx, dth_dy)
    gm = gaussian_filter(grad_mag, sigma=1.0)

    dgm_dy = np.gradient(gm, axis=0) / dy
    dgm_dx = np.gradient(gm, axis=1) / dx

    eps = 1e-12
    tfp = -(dgm_dx * dth_dx + dgm_dy * dth_dy) / (grad_mag + eps)

    # Адвекция температуры: минус скалярное произведение ветра на градиент.
    adv = -(np.asarray(u_ms, float) * dth_dx + np.asarray(v_ms, float) * dth_dy)

    return gaussian_filter(tfp, sigma=0.8), grad_mag, gaussian_filter(adv, sigma=1.0)


#: Модели Open-Meteo, у которых ЕСТЬ уровни давления. Без них профиль
#: не построить, поэтому AROME France HD и подобные сюда не входят.
#: Третий элемент — зона покрытия, None означает глобальную модель.
OM_MODELS = {
    "GFS (NOAA)":            ("gfs_global", None),
    "ICON Global 11км":      ("icon_global", None),
    "ICON-EU 7км":           ("icon_eu", "Европа"),
    "ICON-D2 2км":           ("icon_d2", "Германия"),
    "ECMWF IFS 0.25°":       ("ecmwf_ifs025", None),
    "UKMO Global 10км":      ("ukmo_seamless", None),
    "ARPEGE Global":         ("meteofrance_arpege_world", None),
    "GEM Global (Канада)":   ("gem_global", None),
    "JMA GSM (Япония)":      ("jma_gsm", None),
}


def fetch_openmeteo_grid(lats, lons, model_code, date_str, hour_str, step_str,
                         log=print, historical=False):
    """
    Тянет профили для ВСЕЙ СЕТКИ из Open-Meteo одним пакетом запросов.

    Open-Meteo принимает до 1000 точек в одном обращении через списки
    координат — сетка 31x31 (961 точка) влезает целиком. Это открывает
    для карт модели, которых нет в GRIB-выдаче NOMADS: ICON, ECMWF,
    UKMO, ARPEGE.

    Возвращает список заданий в том же виде, что и GRIB-путь:
        (i, j, lat, lon, pres, gh, tmp, dwpk, u_kt, v_kt)

    Осторожно с объёмом: на каждую точку запрашивается ~120 переменных
    (24 уровня x 5), поэтому точки идут порциями — иначе ответ слишком
    большой и сервер обрывает соединение.
    """
    import time
    import requests

    # Уровней меньше, чем у бота: Open-Meteo считает не запросы, а ОБЪЁМ
    # данных, и на сетке в тысячу точек полный набор мгновенно съедает
    # дневную квоту (ответ 429). Шестнадцать уровней профиль почти не
    # огрубляют — выброшены только промежуточные, где градиенты плавные.
    LEVELS_OM = [1000, 950, 925, 900, 850, 800, 700, 600,
                 500, 400, 300, 250, 200, 150, 100]
    MS_TO_KTS = 1.94384
    CHUNK = 25          # точек в одном запросе
    PAUSE = 1.2         # пауза между запросами, с
    RETRY_WAIT = 20     # сколько ждать после 429

    hourly = ['temperature_2m', 'relative_humidity_2m', 'surface_pressure',
              'wind_speed_10m', 'wind_direction_10m']
    for lv in LEVELS_OM:
        hourly += [f'temperature_{lv}hPa', f'relative_humidity_{lv}hPa',
                   f'geopotential_height_{lv}hPa', f'wind_speed_{lv}hPa',
                   f'wind_direction_{lv}hPa']

    # Целевой момент: дата и час запуска плюс шаг прогноза.
    base = datetime.datetime.strptime(f"{date_str} {hour_str}", "%Y-%m-%d %H")
    target = base + datetime.timedelta(hours=int(step_str))
    want = target.strftime("%Y-%m-%dT%H:00")
    day = target.strftime("%Y-%m-%d")

    if historical:
        url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
        extra = {"start_date": day, "end_date": day}
    else:
        url = "https://api.open-meteo.com/v1/forecast"
        extra = {"forecast_days": 16}

    # Плоский список всех узлов сетки
    nodes = [(i, j, float(la), float(lo))
             for i, la in enumerate(lats) for j, lo in enumerate(lons)]
    log(f"🌐 Open-Meteo: {len(nodes)} точек, модель {model_code}, "
        f"срок {want} UTC")

    jobs = []
    n_bad = 0
    for start in range(0, len(nodes), CHUNK):
        part = nodes[start:start + CHUNK]
        params = dict(
            latitude=",".join(f"{n[2]:.4f}" for n in part),
            longitude=",".join(f"{n[3]:.4f}" for n in part),
            hourly=",".join(hourly),
            models=model_code,
            windspeed_unit="ms",
            timezone="UTC",
            **extra)
        # Повтор при 429: лимит скользящий, через паузу обычно отпускает.
        payload = None
        for attempt in range(4):
            try:
                r = requests.get(url, params=params, timeout=180)
                if r.status_code == 429:
                    wait = RETRY_WAIT * (attempt + 1)
                    log(f"   лимит запросов, жду {wait} с "
                        f"(попытка {attempt + 1}/4)")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                payload = r.json()
                break
            except Exception as e:
                if attempt == 3:
                    n_bad += len(part)
                    log(f"   порция {start // CHUNK + 1}: "
                        f"{type(e).__name__}: {str(e)[:120]}")
                else:
                    time.sleep(5)
        if payload is None:
            if n_bad > len(nodes) * 0.5:
                raise Exception(
                    "Open-Meteo отклоняет запросы (лимит исчерпан).\n"
                    "Подождите час или уменьшите сетку: шаг 0.5° вместо "
                    "0.25° сокращает число точек вчетверо.")
            continue

        time.sleep(PAUSE)

        # При одной точке приходит объект, при нескольких — список
        blocks = payload if isinstance(payload, list) else [payload]
        for (i, j, la, lo), blk in zip(part, blocks):
            h = blk.get('hourly') or {}
            times = h.get('time') or []
            try:
                k = next(t for t, v in enumerate(times) if v.startswith(want))
            except StopIteration:
                n_bad += 1
                continue

            sp = h.get('surface_pressure', [None])[k]
            t0 = h.get('temperature_2m', [None])[k]
            if sp is None or t0 is None:
                n_bad += 1
                continue

            P = [float(sp)]
            H = [10.0]
            T = [float(t0)]
            D = [_dewpoint_from_rh(t0, h.get('relative_humidity_2m', [0])[k])]
            spd = float(h.get('wind_speed_10m', [0])[k] or 0.0) * MS_TO_KTS
            drc = float(h.get('wind_direction_10m', [0])[k] or 0.0)
            U = [-spd * np.sin(np.radians(drc))]
            V = [-spd * np.cos(np.radians(drc))]

            for lv in LEVELS_OM:
                if lv >= sp:
                    continue
                tl = h.get(f'temperature_{lv}hPa', [None])[k]
                hl = h.get(f'geopotential_height_{lv}hPa', [None])[k]
                if tl is None or hl is None or hl <= H[-1]:
                    continue
                P.append(float(lv)); H.append(float(hl)); T.append(float(tl))
                D.append(_dewpoint_from_rh(
                    tl, h.get(f'relative_humidity_{lv}hPa', [0])[k]))
                s_ = float(h.get(f'wind_speed_{lv}hPa', [0])[k] or 0.0) * MS_TO_KTS
                d_ = float(h.get(f'wind_direction_{lv}hPa', [0])[k] or 0.0)
                U.append(-s_ * np.sin(np.radians(d_)))
                V.append(-s_ * np.cos(np.radians(d_)))

            if len(P) < 5:
                n_bad += 1
                continue

            jobs.append((i, j, la, lo,
                         np.array(P), np.array(H), np.array(T),
                         np.array(D), np.array(U), np.array(V)))

        done = min(start + CHUNK, len(nodes))
        if (start // CHUNK) % 5 == 0 or done >= len(nodes):
            log(f"   {done}/{len(nodes)} точек, профилей собрано {len(jobs)}")

    if n_bad:
        log(f"⚠️ Пропущено точек: {n_bad}")
    if not jobs:
        raise Exception(
            "Open-Meteo не вернул ни одной точки.\n"
            "Проверьте: доступна ли модель в этом регионе (у региональных "
            "моделей своя зона), есть ли у неё уровни давления, и не "
            "выходит ли срок за предел прогноза.")
    return jobs


def _crop_indices(coord, val_min, val_max, pad=2):
    """
    По 1D-массиву координат (широта или долгота) находит диапазон индексов,
    покрывающий [val_min, val_max] с небольшим запасом (pad узлов с каждой
    стороны — на случай края региона и погрешности округления).
    """
    i_a = int(np.argmin(np.abs(coord - val_min)))
    i_b = int(np.argmin(np.abs(coord - val_max)))
    lo = max(0, min(i_a, i_b) - pad)
    hi = min(len(coord) - 1, max(i_a, i_b) + pad)
    return lo, hi


def _nearest_index_regular(coord, val):
    """
    O(1)-поиск ближайшего индекса в РЕГУЛЯРНОЙ сетке координат (постоянный
    шаг) через прямую арифметику — вместо xarray .sel(method="nearest"),
    у которого на каждый вызов заметные накладные расходы. Сетка GFS
    регулярная (0.25°/0.5° и т.п. от полюса до полюса) — это не зависит от
    того, какой регион выбран в GUI, поэтому формула безопасна для любых
    координат, которые введёт пользователь.
    """
    step = coord[1] - coord[0] if len(coord) > 1 else 1.0
    idx = int(round((val - coord[0]) / step))
    return min(max(idx, 0), len(coord) - 1)


def _find_source_dataset(datasets, var_names):
    """Возвращает (датасет, имя_переменной) для первого совпадения из var_names."""
    for ds in datasets:
        for name in var_names:
            if name in ds.data_vars:
                return ds, name
    return None, None


def _crop_and_extract(ds, var_name, lat_min, lat_max, lon_min, lon_max):
    """
    Вырезает из датасета только нужный регион (с запасом) и сразу забирает
    переменную как обычный numpy-массив — одна операция на весь регион вместо
    тысяч отдельных ds.sel(...) на каждую точку сетки.
    Возвращает (lat_coord_sub, lon_coord_sub, values_ndarray).
    """
    lat_coord = ds['latitude'].values
    lon_coord = ds['longitude'].values
    i0, i1 = _crop_indices(lat_coord, lat_min, lat_max)
    j0, j1 = _crop_indices(lon_coord, lon_min, lon_max)
    sub = ds.isel(latitude=slice(i0, i1 + 1), longitude=slice(j0, j1 + 1))
    values = sub[var_name].values
    return lat_coord[i0:i1 + 1], lon_coord[j0:j1 + 1], values


class GFSMapApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GFS Grid → SHARPpy → Карта параметров")
        # Окно шире: настройки слева, лог справа во всю высоту. Раньше
        # лог был внизу и при длинном выводе уезжал за край экрана даже
        # в полноэкранном режиме.
        self.geometry("1520x900")
        self.minsize(1220, 700)

        self.generated_figs = []  # Хранение фигур для последующего сохранения
        self.current_model = "GFS"  # Используется при формировании пути сохранения

        self._create_widgets()

    def _create_widgets(self):
        outer = ttk.Frame(self, padding=8)
        outer.pack(fill="both", expand=True)

        # Левая колонка — настройки, фиксированной ширины: иначе при
        # растягивании окна поля разъезжаются, а лог не расширяется.
        left = ttk.Frame(outer, width=680)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        # Правая — лог, забирает всё оставшееся место.
        right = ttk.Frame(outer)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))

        main_frame = left

        # ---------------- 1. Регион ----------------
        region_frame = ttk.LabelFrame(main_frame, text="Регион (широта/долгота)", padding=10)
        region_frame.pack(fill="x", pady=5)

        ttk.Label(region_frame, text="Широта min:").grid(row=0, column=0, sticky="e", padx=2, pady=2)
        self.lat_min_entry = ttk.Entry(region_frame, width=8)
        self.lat_min_entry.insert(0, "58.3")
        self.lat_min_entry.grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(region_frame, text="max:").grid(row=0, column=2, sticky="e", padx=2, pady=2)
        self.lat_max_entry = ttk.Entry(region_frame, width=8)
        self.lat_max_entry.insert(0, "65.8")
        self.lat_max_entry.grid(row=0, column=3, padx=5, pady=2)

        ttk.Label(region_frame, text="Шаг (°):").grid(row=0, column=4, sticky="e", padx=10, pady=2)
        self.grid_step_entry = ttk.Entry(region_frame, width=8)
        self.grid_step_entry.insert(0, "0.25")
        self.grid_step_entry.grid(row=0, column=5, padx=5, pady=2)

        ttk.Label(region_frame, text="Долгота min:").grid(row=1, column=0, sticky="e", padx=2, pady=2)
        self.lon_min_entry = ttk.Entry(region_frame, width=8)
        self.lon_min_entry.insert(0, "126.0")
        self.lon_min_entry.grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(region_frame, text="max:").grid(row=1, column=2, sticky="e", padx=2, pady=2)
        self.lon_max_entry = ttk.Entry(region_frame, width=8)
        self.lon_max_entry.insert(0, "133.5")
        self.lon_max_entry.grid(row=1, column=3, padx=5, pady=2)

        ttk.Label(region_frame, text="(Регион Якутии)", font=("Arial", 8, "italic")).grid(
            row=1, column=4, columnspan=2, sticky="w", padx=10
        )

        # ---- Заготовки областей ----
        # Шаг задан для каждой отдельно: по всей республике 0.25° дал бы
        # больше 15 тысяч точек, а это десятки минут расчёта ради карты,
        # на которой Якутск занимает несколько пикселей.
        preset_box = ttk.Frame(region_frame)
        preset_box.grid(row=2, column=0, columnspan=6, sticky="w", pady=(6, 0))
        ttk.Label(preset_box, text="Область:").pack(side="left", padx=(0, 4))
        for label, (la0, la1, lo0, lo1, stp) in self.REGION_PRESETS.items():
            ttk.Button(preset_box, text=label, width=len(label) + 2,
                       command=lambda v=(la0, la1, lo0, lo1, stp): self._apply_preset(*v)
                       ).pack(side="left", padx=2)
        self.preset_info = ttk.Label(preset_box, text="", font=("Arial", 8, "italic"),
                                     foreground="#555555")
        self.preset_info.pack(side="left", padx=10)
        self._update_preset_info()
        for e in (self.lat_min_entry, self.lat_max_entry, self.lon_min_entry,
                  self.lon_max_entry, self.grid_step_entry):
            e.bind("<KeyRelease>", lambda ev: self._update_preset_info())

        # ---------------- 2. Выбор источника данных ----------------
        source_frame = ttk.LabelFrame(main_frame, text="Источник данных", padding=10)
        source_frame.pack(fill="x", pady=5)

        self.source_var = tk.StringVar(value="nomads")

        rb_nomads = ttk.Radiobutton(
            source_frame,
            text="Оперативный (NOAA NOMADS) — Свежие данные (1–2 дня)",
            variable=self.source_var,
            value="nomads"
        )
        rb_nomads.pack(anchor="w", pady=2)

        rb_aws = ttk.Radiobutton(
            source_frame,
            text="Исторический архив (AWS S3) — За любые даты с 2021 года",
            variable=self.source_var,
            value="aws"
        )
        rb_aws.pack(anchor="w", pady=2)

        rb_era5 = ttk.Radiobutton(
            source_frame,
            text="Реанализ ERA5 (Copernicus CDS) — с 1940 года, ЛЮБОЙ час 00–23",
            variable=self.source_var,
            value="era5"
        )
        rb_era5.pack(anchor="w", pady=2)

        rb_om = ttk.Radiobutton(
            source_frame,
            text="Open-Meteo — ЛЮБАЯ модель (ICON, ECMWF, UKMO, ARPEGE...)",
            variable=self.source_var, value="openmeteo")
        rb_om.pack(anchor="w", pady=2)

        om_box = ttk.Frame(source_frame)
        om_box.pack(anchor="w", padx=18)
        ttk.Label(om_box, text="модель:").pack(side="left")
        self.om_model_var = tk.StringVar(value="ICON Global 11км")
        ttk.Combobox(om_box, textvariable=self.om_model_var, state="readonly",
                     width=26, values=list(OM_MODELS.keys())).pack(side="left", padx=4)
        ttk.Label(source_frame,
                  text="Open-Meteo: до 1000 точек за запрос, поэтому сетка "
                       "качается целиком. Данные те же, что у бота. "
                       "Региональные модели работают только в своей зоне.",
                  font=("Arial", 8, "italic"), foreground="#555555",
                  wraplength=640, justify="left").pack(anchor="w", padx=18)

        ttk.Label(source_frame,
                  text="ERA5: почасовой реанализ (не прогноз). Поле «Шаг» не используется. "
                       "Задержка ~5 суток. Нужен пакет cdsapi и ключ в ~/.cdsapirc.",
                  font=("Arial", 8, "italic"), foreground="#555555",
                  wraplength=640, justify="left").pack(anchor="w", padx=18)

        # ---------------- 3. Время ----------------
        time_frame = ttk.LabelFrame(main_frame, text="Дата / час / шаг прогноза", padding=10)
        time_frame.pack(fill="x", pady=5)

        today_str = datetime.datetime.utcnow().strftime("%Y-%m-%d")

        ttk.Label(time_frame, text="Дата (YYYY-MM-DD):").grid(row=0, column=0, sticky="e", padx=2, pady=2)
        self.date_entry = ttk.Entry(time_frame, width=12)
        self.date_entry.insert(0, today_str)
        self.date_entry.grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(time_frame, text="Час (GFS: 00/06/12/18, ERA5: 00–23):").grid(row=0, column=2, sticky="e", padx=5, pady=2)
        self.hour_entry = ttk.Entry(time_frame, width=5)
        self.hour_entry.insert(0, "00")
        self.hour_entry.grid(row=0, column=3, padx=5, pady=2)

        ttk.Label(time_frame, text="Шаг (000/003/...):").grid(row=0, column=4, sticky="e", padx=5, pady=2)
        self.step_entry = ttk.Entry(time_frame, width=5)
        self.step_entry.insert(0, "000")
        self.step_entry.grid(row=0, column=5, padx=5, pady=2)

        # ---------------- 4. Выбор параметров ----------------
        param_frame = ttk.LabelFrame(main_frame, text="Параметры для расчёта и отрисовки", padding=10)
        param_frame.pack(fill="x", pady=5)

        self.cb_sbcape = tk.BooleanVar(value=False)
        self.cb_mlcape = tk.BooleanVar(value=False)
        self.cb_mucape = tk.BooleanVar(value=False)
        self.cb_cape3k = tk.BooleanVar(value=False)
        self.cb_mlcin = tk.BooleanVar(value=False)
        self.cb_sbcin = tk.BooleanVar(value=False)

        self.cb_shear1k = tk.BooleanVar(value=False)
        self.cb_shear3k = tk.BooleanVar(value=False)
        self.cb_shear6k = tk.BooleanVar(value=False)
        self.cb_srh1k = tk.BooleanVar(value=False)
        self.cb_srh3k = tk.BooleanVar(value=False)

        self.cb_scp = tk.BooleanVar(value=False)
        self.cb_stp = tk.BooleanVar(value=False)

        self.cb_outlook_tornado = tk.BooleanVar(value=False)
        self.cb_outlook_overall = tk.BooleanVar(value=False)
        self.cb_outlook_trigger = tk.BooleanVar(value=False)
        # Приглушение соседних регионов: полезно на крупных областях,
        # где иначе непонятно, где кончается республика.
        self.cb_mask_yakutia = tk.BooleanVar(value=True)
        # Фронты разведены на три независимых переключателя: считать,
        # показывать линии и влиять на риск — это разные вещи, и раньше
        # они были связаны одной галочкой.
        self.cb_fronts = tk.BooleanVar(value=False)        # вычислять
        self.cb_fronts_draw = tk.BooleanVar(value=True)    # рисовать линии
        self.cb_fronts_risk = tk.BooleanVar(value=False)   # повышать риск
        # Риск на сутки: максимум по нескольким срокам вместо одного часа.
        self.cb_daily = tk.BooleanVar(value=False)

        col0 = ttk.Frame(param_frame)
        col0.grid(row=0, column=0, sticky="nw", padx=10)
        ttk.Label(col0, text="CAPE:", font=("Arial", 9, "bold")).pack(anchor="w")
        ttk.Checkbutton(col0, text="SBCAPE (J/kg)", variable=self.cb_sbcape).pack(anchor="w")
        ttk.Checkbutton(col0, text="MLCAPE (J/kg)", variable=self.cb_mlcape).pack(anchor="w")
        ttk.Checkbutton(col0, text="MUCAPE (J/kg)", variable=self.cb_mucape).pack(anchor="w")
        ttk.Checkbutton(col0, text="3km CAPE (J/kg)", variable=self.cb_cape3k).pack(anchor="w")
        ttk.Checkbutton(col0, text="MLCIN (J/kg)", variable=self.cb_mlcin).pack(anchor="w")
        ttk.Checkbutton(col0, text="SBCIN (J/kg)", variable=self.cb_sbcin).pack(anchor="w")

        col1 = ttk.Frame(param_frame)
        col1.grid(row=0, column=1, sticky="nw", padx=10)
        ttk.Label(col1, text="Shear:", font=("Arial", 9, "bold")).pack(anchor="w")
        ttk.Checkbutton(col1, text="0-1km Shear (m/s)", variable=self.cb_shear1k).pack(anchor="w")
        ttk.Checkbutton(col1, text="0-3km Shear (m/s)", variable=self.cb_shear3k).pack(anchor="w")
        ttk.Checkbutton(col1, text="0-6km Shear (m/s)", variable=self.cb_shear6k).pack(anchor="w")
        ttk.Checkbutton(col1, text="0-1km SRH (m²/s²)", variable=self.cb_srh1k).pack(anchor="w")
        ttk.Checkbutton(col1, text="0-3km SRH (m²/s²)", variable=self.cb_srh3k).pack(anchor="w")

        col2 = ttk.Frame(param_frame)
        col2.grid(row=0, column=2, sticky="nw", padx=10)
        ttk.Label(col2, text="Композитные:", font=("Arial", 9, "bold")).pack(anchor="w")
        ttk.Checkbutton(col2, text="SCP", variable=self.cb_scp).pack(anchor="w")
        ttk.Checkbutton(col2, text="STP", variable=self.cb_stp).pack(anchor="w")

        col3 = ttk.Frame(param_frame)
        col3.grid(row=0, column=3, sticky="nw", padx=10)
        ttk.Label(col3, text="Outlook (эксперим.):", font=("Arial", 9, "bold")).pack(anchor="w")
        ttk.Checkbutton(col3, text="Торнадо-риск", variable=self.cb_outlook_tornado).pack(anchor="w")
        ttk.Checkbutton(col3, text="Overall-риск", variable=self.cb_outlook_overall).pack(anchor="w")
        ttk.Checkbutton(col3, text="Триггер (инициация)", variable=self.cb_outlook_trigger).pack(anchor="w")
        ttk.Separator(col3, orient="horizontal").pack(fill="x", pady=4)
        ttk.Checkbutton(col3, text="Затемнить вне Якутии",
                        variable=self.cb_mask_yakutia).pack(anchor="w")
        ttk.Checkbutton(col3, text="Фронты (850 гПа)",
                        variable=self.cb_fronts).pack(anchor="w")
        ttk.Checkbutton(col3, text="     показывать линии",
                        variable=self.cb_fronts_draw).pack(anchor="w")
        ttk.Checkbutton(col3, text="     повышать риск",
                        variable=self.cb_fronts_risk).pack(anchor="w")
        ttk.Checkbutton(col3, text="Риск на сутки (макс. по 4 срокам)",
                        variable=self.cb_daily).pack(anchor="w")

        ttk.Separator(col3, orient="horizontal").pack(fill="x", pady=4)
        ttk.Label(col3, text="Пороги:", font=("Arial", 8)).pack(anchor="w")
        self.climo_var = tk.StringVar(value="Якутия — центр")
        self.climo_combo = ttk.Combobox(
            col3, textvariable=self.climo_var, state="readonly", width=24,
            values=list(CLIMO_SETS.keys()))
        self.climo_combo.pack(anchor="w", pady=(0, 2))
        self.climo_info = ttk.Label(col3, text="", font=("Arial", 7, "italic"),
                                    foreground="#555555", wraplength=190,
                                    justify="left")
        self.climo_info.pack(anchor="w")
        self.climo_combo.bind("<<ComboboxSelected>>", self._on_climo_change)
        self._on_climo_change()

        btn_box = ttk.Frame(param_frame)
        btn_box.grid(row=1, column=0, columnspan=3, sticky="w", pady=5)
        ttk.Button(btn_box, text="Выбрать всё", command=self._select_all_params).pack(side="left", padx=2)
        ttk.Button(btn_box, text="Сбросить", command=self._deselect_all_params).pack(side="left", padx=2)

        # ---------------- 5. Путь сохранения ----------------
        save_frame = ttk.LabelFrame(main_frame, text="Сохранение карт", padding=10)
        save_frame.pack(fill="x", pady=5)

        default_dir = os.path.join(os.path.expanduser("~"), "Model_Maps")

        ttk.Label(save_frame, text="Папка:").pack(side="left", padx=5)
        self.dir_entry = ttk.Entry(save_frame, width=50)
        self.dir_entry.insert(0, default_dir)
        self.dir_entry.pack(side="left", padx=5, fill="x", expand=True)

        ttk.Button(save_frame, text="Обзор", command=self._browse_dir).pack(side="left", padx=5)

        # ---------------- 6. Кнопки управления ----------------
        ctrl_frame = ttk.Frame(main_frame, padding=5)
        ctrl_frame.pack(fill="x", pady=5)

        self.btn_run = ttk.Button(
            ctrl_frame,
            text="🚀 Загрузить GRIB → SHARPpy → Карты",
            command=self._start_process
        )
        self.btn_run.pack(side="left", padx=5)

        self.btn_save = ttk.Button(
            ctrl_frame,
            text="💾 Сохранить карты",
            state="disabled",
            command=self._save_maps
        )
        self.btn_save.pack(side="left", padx=5)

        ttk.Label(ctrl_frame, text="Процессов:").pack(side="left", padx=(15, 2))
        default_workers = max(1, (os.cpu_count() or 2) - 1)
        self.workers_entry = ttk.Entry(ctrl_frame, width=4)
        self.workers_entry.insert(0, str(default_workers))
        self.workers_entry.pack(side="left")

        # ---------------- 7. Логи и прогресс ----------------
        self.progress = ttk.Progressbar(main_frame, mode="determinate")
        self.progress.pack(fill="x", pady=5)

        self.status_label = ttk.Label(main_frame, text="Готов к работе", font=("Arial", 9, "italic"))
        self.status_label.pack(anchor="w")

        log_frame = ttk.LabelFrame(right, text="Лог выполнения", padding=5)
        log_frame.pack(fill="both", expand=True)

        log_scroll = ttk.Scrollbar(log_frame, orient="vertical")
        log_scroll.pack(side="right", fill="y")
        self.log_text = tk.Text(log_frame, bg="#111625", fg="#00FF66",
                                insertbackground="white", wrap="word",
                                yscrollcommand=log_scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll.config(command=self.log_text.yview)

        btn_row = ttk.Frame(right)
        btn_row.pack(fill="x", pady=(4, 0))
        ttk.Button(btn_row, text="Очистить",
                   command=lambda: self.log_text.delete("1.0", tk.END)
                   ).pack(side="left")
        ttk.Button(btn_row, text="Копировать всё",
                   command=self._copy_log).pack(side="left", padx=4)

    #: Готовые области. Кортеж: широта min/max, долгота min/max, шаг.
    REGION_PRESETS = {
        "Вся Якутия":   (55.0, 72.0, 105.0, 163.0, 0.5),
        "Центр и юг":   (55.8, 63.5, 123.9, 133.1, 0.25),
        "Якутск":       (58.3, 65.8, 126.0, 133.5, 0.25),
    }

    def _copy_log(self):
        """Весь лог в буфер обмена — удобнее, чем выделять мышкой."""
        try:
            self.clipboard_clear()
            self.clipboard_append(self.log_text.get("1.0", tk.END))
        except Exception:
            pass

    def _apply_preset(self, la0, la1, lo0, lo1, step):
        for entry, val in ((self.lat_min_entry, la0), (self.lat_max_entry, la1),
                           (self.lon_min_entry, lo0), (self.lon_max_entry, lo1),
                           (self.grid_step_entry, step)):
            entry.delete(0, tk.END)
            entry.insert(0, str(val))
        self._update_preset_info()

    def _update_preset_info(self):
        """Показывает размер сетки: по нему сразу видно цену выбора."""
        try:
            la0 = float(self.lat_min_entry.get()); la1 = float(self.lat_max_entry.get())
            lo0 = float(self.lon_min_entry.get()); lo1 = float(self.lon_max_entry.get())
            st = float(self.grid_step_entry.get())
            n_lat = int((la1 - la0) / st) + 1
            n_lon = int((lo1 - lo0) / st) + 1
            total = n_lat * n_lon
            warn = "  — это надолго" if total > 8000 else ""
            self.preset_info.config(text=f"сетка {n_lat}×{n_lon} = {total} точек{warn}")
        except (ValueError, ZeroDivisionError, AttributeError):
            self.preset_info.config(text="")

    def _on_climo_change(self, event=None):
        """Подключает выбранный набор порогов и показывает, что вышло."""
        name = self.climo_var.get()
        target = CLIMO_SETS.get(name)

        if target is None:
            use_spc_thresholds()
            self.climo_info.config(
                text="Пороги из практики SPC. Годятся везде, но для Якутии "
                     "завышены по CAPE и занижены по DCAPE.")
            return

        if target == "__custom__":
            from tkinter import filedialog
            p = filedialog.askopenfilename(
                title="Файл порогов", filetypes=[("JSON", "*.json")])
            if not p:
                self.climo_var.set("SPC (США) — эталон")
                self._on_climo_change()
                return
            ok = load_climo_thresholds(p)
        else:
            ok = load_climo_thresholds(_find_climo_file(target))

        if ok:
            self.climo_info.config(text=CLIMO_SOURCE)
        else:
            # Файла нет — честно говорим и откатываемся на SPC, а не
            # считаем молча непонятно по чему.
            use_spc_thresholds()
            self.climo_info.config(
                text=f"Файл {target} не найден — считаю по порогам SPC. "
                     f"Соберите его: era5_clim_thresholds_v2.py")

    def _select_all_params(self):
        for var in [self.cb_sbcape, self.cb_mlcape, self.cb_mucape, self.cb_cape3k,
                    self.cb_shear1k, self.cb_shear3k, self.cb_shear6k,
                    self.cb_srh1k, self.cb_srh3k, self.cb_scp, self.cb_stp,
                    self.cb_outlook_tornado, self.cb_outlook_overall,
                    self.cb_outlook_trigger]:
            var.set(True)

    def _deselect_all_params(self):
        for var in [self.cb_sbcape, self.cb_mlcape, self.cb_mucape, self.cb_cape3k,
                    self.cb_shear1k, self.cb_shear3k, self.cb_shear6k,
                    self.cb_srh1k, self.cb_srh3k, self.cb_scp, self.cb_stp,
                    self.cb_outlook_tornado, self.cb_outlook_overall,
                    self.cb_outlook_trigger]:
            var.set(False)

    def _browse_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.dir_entry.delete(0, tk.END)
            self.dir_entry.insert(0, d)

    def log(self, text):
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)

    def _start_process(self):
        self.btn_run.config(state="disabled")
        self.btn_save.config(state="disabled")
        self.progress["value"] = 0
        self.generated_figs.clear()

        thread = threading.Thread(target=self._worker_process)
        thread.daemon = True
        thread.start()

    # ---------------- ERA5 через Copernicus CDS ----------------
    #: Уровни давления ERA5, которые запрашиваем. До 50 гПа — этого с
    #: запасом хватает на EL даже у самых глубоких штормов, а качать все
    #: 37 уровней вплоть до 1 гПа смысла нет (только объём).
    ERA5_LEVELS = [
        '1000', '975', '950', '925', '900', '875', '850', '825', '800',
        '775', '750', '700', '650', '600', '550', '500', '450', '400',
        '350', '300', '250', '225', '200', '175', '150', '125', '100',
        '70', '50',
    ]

    def _download_era5_cds(self, date_str, hour_str, lat_min, lat_max,
                           lon_min, lon_max, cache_dir):
        """
        Качает ERA5 через Copernicus Climate Data Store двумя запросами:
        уровни давления + приземные поля. Возвращает (путь_PL, путь_SL).

        Требует установленный пакет cdsapi и ключ в ~/.cdsapirc
        (регистрация бесплатная: cds.climate.copernicus.eu).
        """
        try:
            import cdsapi
        except ImportError:
            raise Exception(
                "cdsapi недоступен в этом окружении.\n\n"
                "Свежий cdsapi требует Python 3.8+ (cads_api_client использует "
                "typing.Literal), а здесь Python 3.7 из-за самой SHARPpy — "
                "поставить его сюда нельзя. Ставить cdsapi<0.7 бесполезно: "
                "старые версии не работают с новым CDS.\n\n"
                "Решение — скачать данные отдельно, в современном окружении:\n"
                "    conda create -n era5 python=3.11 -y\n"
                "    conda activate era5\n"
                '    pip install "cdsapi>=0.7.7"\n'
                "    python era5_download.py --date ГГГГ-ММ-ДД --hour ЧЧ \\\n"
                "        --lat-min .. --lat-max .. --lon-min .. --lon-max ..\n\n"
                "Скрипт положит GRIB в тот же кэш (~/GFS_Cache) с теми же "
                "именами, и это приложение подхватит их без скачивания — "
                "главное указать здесь тот же регион, дату и час."
            )

        yyyy, mm, dd = date_str.split('-')
        tag = f"era5_{yyyy}{mm}{dd}_{hour_str}_{lat_min}_{lat_max}_{lon_min}_{lon_max}"
        pl_path = os.path.join(cache_dir, tag + "_pl.grib")
        sl_path = os.path.join(cache_dir, tag + "_sl.grib")

        have_pl = os.path.exists(pl_path) and os.path.getsize(pl_path) > 10000
        have_sl = os.path.exists(sl_path) and os.path.getsize(sl_path) > 5000
        if have_pl and have_sl:
            self.log("📁 ERA5 уже скачан ранее — беру из кэша.")
            return pl_path, sl_path

        # area = [север, запад, юг, восток], с запасом в 1 градус
        area = [round(lat_max + 1.0, 2), round(lon_min - 1.0, 2),
                round(lat_min - 1.0, 2), round(lon_max + 1.0, 2)]
        time_str = f"{hour_str}:00"

        client = cdsapi.Client(quiet=True, progress=False)

        def _retrieve(dataset, request, target):
            # Новый CDS (с осени 2024) ждёт 'data_format', старый — 'format'.
            # Пробуем современный вариант, при ошибке откатываемся.
            try:
                client.retrieve(dataset, dict(request, data_format='grib'), target)
            except Exception as e_new:
                self.log(f"   (пробую совместимость со старым CDS: {type(e_new).__name__})")
                client.retrieve(dataset, dict(request, format='grib'), target)

        if not have_pl:
            self.log("Запрос ERA5 (уровни давления) в CDS... "
                     "первый запрос может стоять в очереди несколько минут.")
            self.status_label.config(text="CDS: очередь на уровни давления...")
            _retrieve('reanalysis-era5-pressure-levels', {
                'product_type': 'reanalysis',
                'variable': ['temperature', 'relative_humidity',
                             'u_component_of_wind', 'v_component_of_wind',
                             'geopotential'],
                'pressure_level': self.ERA5_LEVELS,
                'year': yyyy, 'month': mm, 'day': dd,
                'time': time_str,
                'area': area,
            }, pl_path)
            self.log(f"   ✅ уровни давления: {os.path.getsize(pl_path)/1e6:.1f} МБ")

        if not have_sl:
            self.log("Запрос ERA5 (приземные поля) в CDS...")
            self.status_label.config(text="CDS: очередь на приземные поля...")
            _retrieve('reanalysis-era5-single-levels', {
                'product_type': 'reanalysis',
                'variable': ['surface_pressure', '2m_temperature',
                             '2m_dewpoint_temperature',
                             '10m_u_component_of_wind', '10m_v_component_of_wind',
                             'geopotential'],
                'year': yyyy, 'month': mm, 'day': dd,
                'time': time_str,
                'area': area,
            }, sl_path)
            self.log(f"   ✅ приземные поля: {os.path.getsize(sl_path)/1e6:.1f} МБ")

        return pl_path, sl_path

    @staticmethod
    def _era5_add_geopotential_height(ds):
        """
        ERA5 хранит ГЕОПОТЕНЦИАЛ z в м²/с², а НЕ геопотенциальную высоту
        в метрах (в отличие от GFS, где сразу есть gh). Если этого не
        учесть, высоты окажутся примерно в 9.8 раза больше реальных, и
        профиль развалится на проверке монотонности высоты.

        Делим на стандартное g = 9.80665 (именно это значение использует
        IFS) и добавляем в датасет привычное поле gh / orog.
        """
        G = 9.80665
        if 'z' not in ds.data_vars:
            return ds
        is_isobaric = ('isobaricInhPa' in ds.coords or 'isobaricInhPa' in ds.dims)
        name = 'gh' if is_isobaric else 'orog'
        if name in ds.data_vars:
            return ds
        return ds.assign(**{name: ds['z'] / G})

    def _worker_process(self):
        try:
            lat_min = float(self.lat_min_entry.get())
            lat_max = float(self.lat_max_entry.get())
            lon_min = float(self.lon_min_entry.get())
            lon_max = float(self.lon_max_entry.get())
            grid_step = float(self.grid_step_entry.get())

            date_str = self.date_entry.get().strip()
            hour_str = self.hour_entry.get().strip().zfill(2)
            step_str = self.step_entry.get().strip().zfill(3)

            source_type = self.source_var.get()
            # Задел на будущее: сейчас единственный поддерживаемый источник — GFS
            # (и aws, и nomads — это GFS, просто разные способы его скачать).
            # Когда добавится ARPEGE/ICON/др., здесь нужно будет проставлять
            # соответствующее имя модели в зависимости от source_type.
            if source_type == "era5":
                self.current_model = "ERA5"
            elif source_type == "openmeteo":
                # В имени папки — сама модель, а не «Open-Meteo»: иначе
                # ICON и ECMWF свалятся в одну кучу.
                self.current_model = self.om_model_var.get().split()[0].upper()
            else:
                self.current_model = "GFS"

            selected_params = {
                'sbcape': self.cb_sbcape.get(),
                'mlcape': self.cb_mlcape.get(),
                'mucape': self.cb_mucape.get(),
                'cape3k': self.cb_cape3k.get(),
                'mlcin': self.cb_mlcin.get(),
                'sbcin': self.cb_sbcin.get(),
                'shear1k': self.cb_shear1k.get(),
                'shear3k': self.cb_shear3k.get(),
                'shear6k': self.cb_shear6k.get(),
                'srh1k': self.cb_srh1k.get(),
                'srh3k': self.cb_srh3k.get(),
                'scp': self.cb_scp.get(),
                'stp': self.cb_stp.get(),
                'outlook_tornado': self.cb_outlook_tornado.get(),
                'outlook_overall': self.cb_outlook_overall.get(),
                'outlook_trigger': self.cb_outlook_trigger.get(),
            }

            if not any(selected_params.values()):
                raise Exception("Не выбран ни один параметр для расчёта!")

            lats = np.arange(lat_min, lat_max + 1e-5, grid_step)
            lons = np.arange(lon_min, lon_max + 1e-5, grid_step)
            # Размеры сетки нужны РАНЬШЕ, чем строится grid_results:
            # блок фронтов создаёт по ним маску влияния, а он идёт до
            # подготовки точек. Без этого фронты падали с
            # UnboundLocalError, причём только при включённом расчёте
            # фронтов — оттого и не замечалось.
            n_lat, n_lon = len(lats), len(lons)

            self.log(f"==================================================")
            self.log(f"Сетка: {len(lats)}×{len(lons)} = {len(lats)*len(lons)} точек")
            if (selected_params.get('outlook_tornado')
                    or selected_params.get('outlook_overall')):
                self.log(f"📐 Пороги outlook: {CLIMO_SOURCE}")

            yyyymmdd = date_str.replace("-", "")

            if source_type == "nomads":
                # NOMADS grib-фильтр хранит только последние ~10 суток запусков
                # модели — запрос вне этого окна даёт неинформативную ошибку
                # скачивания (403 "Request for Future Data" и т.п.).
                try:
                    requested_dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                    days_old = (datetime.datetime.utcnow() - requested_dt).days
                    if days_old < 0:
                        # ВАЖНО: поле "Дата" — это дата ЗАПУСКА модели, она не
                        # может быть в будущем. Чтобы получить прогноз на
                        # будущую дату, нужно ставить Дата=сегодня (или
                        # последний доступный запуск) и увеличивать поле "Шаг"
                        # прогноза (до 384ч вперёд), а не переставлять саму дату.
                        self.log(f"⚠️ Дата ({date_str}) в будущем — это невозможно: поле «Дата» "
                                 f"означает дату ЗАПУСКА модели, а не дату, на которую нужен прогноз. "
                                 f"Чтобы получить прогноз на {date_str}, поставьте «Дата» = сегодня "
                                 f"(или последний доступный запуск) и увеличьте «Шаг прогноза» — "
                                 f"столько часов вперёд от запуска, сколько нужно, чтобы дойти до "
                                 f"нужной даты (шаг доступен до 384ч).")
                    elif days_old > 10:
                        self.log(f"⚠️ NOMADS хранит только последние ~10 суток запусков, а запрошенная "
                                 f"дата ({date_str}) старше. Скорее всего скачивание не удастся — "
                                 f"переключитесь на «Исторический архив (AWS S3)».")
                except ValueError:
                    pass

            # ---------------- Срок или сутки ----------------
            # Риск на сутки считается как МАКСИМУМ по нескольким срокам:
            # карта отвечает «что может случиться за день», а не «что будет
            # ровно в этот час». Так же устроены суточные обзоры SPC.
            if self.cb_daily.get():
                base_h = int(hour_str)
                base_s = int(step_str)
                # Шаги, покрывающие сутки от заданного момента. Для Якутии
                # (UTC+9) это 12:00-21:00 местного — всё конвективное окно.
                run_list = [f"{base_s + d:03d}" for d in (0, 3, 6, 9)]
                self.log(f"📅 Режим суток: сроки +{', +'.join(run_list)}ч "
                         f"от {date_str} {hour_str}z, берём максимум по каждой точке")
            else:
                run_list = [step_str]

            step_str_req = step_str          # исходный шаг для имён файлов
            daily_grids = None
            fronts = None
            front_near = None

            for _run_i, step_str in enumerate(run_list, 1):
                if len(run_list) > 1:
                    self.log(f"───── срок {_run_i}/{len(run_list)}: +{step_str}ч ─────")

                cache_dir = os.path.join(os.path.expanduser("~"), "GFS_Cache")
                os.makedirs(cache_dir, exist_ok=True)

                # ---------------- Open-Meteo: сетка одним пакетом ----------------
                # Отдельная ветка: здесь нет GRIB вовсе, профили приходят
                # готовыми из API. Зато доступны модели, которых нет в
                # выдаче NOMADS — ICON, ECMWF, UKMO, ARPEGE.
                if source_type == "openmeteo":
                    om_name = self.om_model_var.get()
                    om_code, om_area = OM_MODELS[om_name]
                    if om_area:
                        self.log(f"ℹ️ {om_name} покрывает только «{om_area}» — "
                                 f"за пределами зоны точки вернутся пустыми.")
                    # Архив прогнозов нужен для дат старше нескольких суток.
                    _hist = False
                    try:
                        _age = (datetime.datetime.utcnow()
                                - datetime.datetime.strptime(date_str, "%Y-%m-%d")).days
                        _hist = _age > 5
                    except ValueError:
                        pass
                    if _hist:
                        self.log("ℹ️ Дата старше 5 суток — беру архив прогнозов "
                                 "(доступен с 2021 года).")
                    # Предупреждаем ДО запроса: Open-Meteo считает объём
                    # данных, а не число обращений, и большая сетка съедает
                    # бесплатную квоту за один расчёт. Лучше узнать заранее,
                    # чем на середине получить 429.
                    if len(lats) * len(lons) > 400:
                        self.log(f"⚠️ {len(lats) * len(lons)} точек — это много "
                                 f"для бесплатной квоты Open-Meteo.")
                        self.log(f"   Если упрётся в лимит (429), возьмите шаг "
                                 f"0.5° или область поменьше.")
                    point_jobs = fetch_openmeteo_grid(
                        lats, lons, om_code, date_str, hour_str, step_str,
                        log=self.log, historical=_hist)
                    fr_t = np.full((n_lat, n_lon), np.nan)
                    fr_u = np.full((n_lat, n_lon), np.nan)
                    fr_v = np.full((n_lat, n_lon), np.nan)
                    # Поля на 850 гПа для фронтов достаём из готовых профилей.
                    for (_i, _j, _la, _lo, _p, _h, _t, _d, _u, _v) in point_jobs:
                        _k = int(np.argmin(np.abs(_p - 850.0)))
                        if abs(_p[_k] - 850.0) < 30:
                            fr_t[_i, _j] = _t[_k]
                            fr_u[_i, _j] = _u[_k]
                            fr_v[_i, _j] = _v[_k]
                else:
                    # ---------------- ERA5 (реанализ через CDS) ----------------
                    if source_type == "era5":
                        try:
                            req_dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                            days_old = (datetime.datetime.utcnow() - req_dt).days
                            if days_old < 5:
                                self.log(f"⚠️ ERA5 публикуется с задержкой ~5 суток, а до "
                                         f"{date_str} осталось меньше. Запрос, скорее всего, "
                                         f"вернёт ошибку или пустой файл — для свежих дат "
                                         f"используйте NOMADS/AWS.")
                            if req_dt.year < 1940:
                                self.log("⚠️ ERA5 начинается с 1940 года.")
                        except ValueError:
                            pass
                        if step_str != "000":
                            self.log("ℹ️ ERA5 — это реанализ, а не прогноз: поле «Шаг» "
                                     "игнорируется, берётся состояние атмосферы на "
                                     f"{date_str} {hour_str}:00 UTC.")
                        era5_pl, era5_sl = self._download_era5_cds(
                            date_str, hour_str, lat_min, lat_max, lon_min, lon_max, cache_dir)
                        grib_files = [era5_pl, era5_sl]
                    else:
                        grib_files = None

                    tmp_grib = os.path.join(cache_dir, f"gfs_{source_type}_{yyyymmdd}_{hour_str}_f{step_str}.grib2")

                    if source_type == "era5":
                        pass  # уже скачано выше
                    elif os.path.exists(tmp_grib) and os.path.getsize(tmp_grib) > 1000000:
                        self.log(f"📁 Найден сохраненный файл на ПК! Пропускаем скачивание...")
                    else:
                        if source_type == "aws":
                            url = f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{yyyymmdd}/{hour_str}/atmos/gfs.t{hour_str}z.pgrb2.0p25.f{step_str}"
                            self.log(f"Качаю исторический GRIB2 с AWS S3...")
                        else:
                            url = (
                                f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?"
                                f"file=gfs.t{hour_str}z.pgrb2.0p25.f{step_str}&"
                                f"all_lev=on&all_var=on&"
                                f"subregion=&leftlon={lon_min}&rightlon={lon_max}&toplat={lat_max}&bottomlat={lat_min}&"
                                f"dir=%2Fgfs.{yyyymmdd}%2F{hour_str}%2Fatmos"
                            )
                            self.log(f"Качаю GRIB2 с NOMADS...")

                        self.status_label.config(text="Загрузка GRIB2 файла...")

                        # ВАЖНО: без явного User-Agent NOAA-CDN нередко отдаёт 403 —
                        # дефолтный "python-requests/x.x.x" многие CDN блокируют как
                        # признак бота. Прикидываемся обычным браузером.
                        headers = {
                            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                                            "Chrome/124.0.0.0 Safari/537.36")
                        }

                        resp = None
                        last_error_text = ""
                        max_attempts = 3
                        for attempt in range(1, max_attempts + 1):
                            resp = requests.get(url, headers=headers, stream=True, timeout=90)
                            if resp.status_code == 200:
                                break
                            last_error_text = resp.text[:300] if resp.text else ""
                            # Диагностика: по заголовкам обычно видно, КТО именно блокирует
                            # (Cloudflare/Akamai/сам сервер) — полезно, если подозреваете
                            # блокировку по IP (например, из-за VPN/WARP), а не rate-limit.
                            diag_headers = {k: v for k, v in resp.headers.items()
                                             if k.lower() in ('server', 'cf-ray', 'cf-mitigated',
                                                               'x-served-by', 'via', 'x-cache')}
                            self.log(f"⚠️ Попытка {attempt}/{max_attempts}: сервер вернул "
                                     f"{resp.status_code}. {last_error_text}")
                            if diag_headers:
                                self.log(f"   Заголовки ответа: {diag_headers}")
                            if attempt < max_attempts:
                                # NOMADS сам просит "сделать паузу перед повторной отправкой
                                # запроса" при перегрузке — так и делаем, с нарастающей паузой.
                                wait_s = 5 * attempt
                                self.log(f"⏳ Жду {wait_s} сек. перед повтором...")
                                time.sleep(wait_s)

                        if resp.status_code != 200:
                            raise Exception(
                                f"Ошибка скачивания ({resp.status_code}) после {max_attempts} попыток. "
                                f"Ответ сервера: {last_error_text or '(пусто)'}"
                            )

                        with open(tmp_grib, "wb") as f:
                            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                                if chunk:
                                    f.write(chunk)

                    self.log("GRIB2 успешно загружен. Чтение профилей...")
                    self.status_label.config(text="Чтение GRIB2 через cfgrib...")

                    # --- Чтение всех наборов данных из GRIB2 ---
                    # backend_kwargs={'indexpath': ''} чтобы cfgrib не пытался писать .idx
                    # рядом с файлом (частые проблемы прав/блокировок на Windows) и не
                    # переиспользовал битый кэш индекса от прошлых запусков.
                    if grib_files is None:
                        grib_files = [tmp_grib]

                    datasets = []
                    for gf in grib_files:
                        datasets.extend(cfgrib.open_datasets(gf, backend_kwargs={'indexpath': ''}))

                    if source_type == "era5":
                        # ERA5 отдаёт геопотенциал z (м²/с²), а не высоту gh (м).
                        # Без пересчёта высоты выйдут ~в 9.8 раза больше реальных.
                        datasets = [self._era5_add_geopotential_height(ds) for ds in datasets]

                    ds_isobaric = None
                    surface_datasets = []  # список ВСЕХ non-isobaric датасетов (не мерджим!)

                    for ds in datasets:
                        if 'isobaricInhPa' in ds.coords or 'isobaricInhPa' in ds.dims:
                            # Если изобарических датасетов несколько (бывает - разный
                            # набор уровней для влажности и ветра) - берём тот, где
                            # больше переменных, а остальные тоже кладём в surface_datasets
                            # на случай, если там есть нужные вертикальные переменные.
                            if ds_isobaric is None or len(ds.data_vars) > len(ds_isobaric.data_vars):
                                if ds_isobaric is not None:
                                    surface_datasets.append(ds_isobaric)
                                ds_isobaric = ds
                            else:
                                surface_datasets.append(ds)
                        else:
                            surface_datasets.append(ds)

                    if ds_isobaric is None:
                        raise Exception("Не найден изобарический (isobaricInhPa) набор данных в GRIB2 — проверьте, "
                                         "что скачанный файл содержит уровни давления (t, u, v, gh, r).")

                    if not surface_datasets:
                        raise Exception("Не найдены приземные (non-isobaric) наборы данных в GRIB2.")

                    self.log(f"Изобарический набор: {list(ds_isobaric.data_vars)}, уровней: "
                             f"{ds_isobaric.sizes.get('isobaricInhPa', '?')}")
                    for k, sds in enumerate(surface_datasets):
                        self.log(f"Surface-набор #{k}: {list(sds.data_vars)}")

                    # ------------------------------------------------------------
                    # Однократная вырезка нужного региона + выгрузка в numpy.
                    # РАНЬШЕ: на каждую из тысяч точек сетки — до 7 отдельных
                    # ds.sel(..., method="nearest") (по одному на изобарический срез
                    # и на каждую приземную переменную, ещё и перебором по всем
                    # surface-датасетам). Именно это давало 5-6 минут на "Подготовку
                    # данных". ТЕПЕРЬ: régион вырезается один раз, дальше поиск
                    # ближайшего узла — O(1) арифметика по регулярной сетке GFS
                    # (не зависит от того, какой регион введён в GUI — сетка модели
                    # везде одна и та же, шаг фиксирован).
                    # ------------------------------------------------------------
                    self.status_label.config(text="Вырезаю регион из GRIB...")

                    pres_levels = ds_isobaric['isobaricInhPa'].values.astype(float)
                    if np.max(pres_levels) > 2000:
                        pres_levels = pres_levels / 100.0  # Паскали -> гПа

                    iso_lat_sub, iso_lon_sub, t_arr = _crop_and_extract(
                        ds_isobaric, 't', lat_min, lat_max, lon_min, lon_max)
                    t_arr = t_arr - 273.15
                    _, _, gh_arr = _crop_and_extract(ds_isobaric, 'gh', lat_min, lat_max, lon_min, lon_max)
                    _, _, u_arr = _crop_and_extract(ds_isobaric, 'u', lat_min, lat_max, lon_min, lon_max)
                    u_arr = u_arr * 1.94384
                    _, _, v_arr = _crop_and_extract(ds_isobaric, 'v', lat_min, lat_max, lon_min, lon_max)
                    v_arr = v_arr * 1.94384

                    if 'r' in ds_isobaric:
                        _, _, r_arr = _crop_and_extract(ds_isobaric, 'r', lat_min, lat_max, lon_min, lon_max)
                        d_arr = None
                    elif 'd' in ds_isobaric:
                        _, _, d_arr = _crop_and_extract(ds_isobaric, 'd', lat_min, lat_max, lon_min, lon_max)
                        d_arr = d_arr - 273.15
                        r_arr = None
                    else:
                        r_arr = None
                        d_arr = None

                    # Приземные переменные: для каждой ищем исходный датасет ОДИН РАЗ
                    # (а не на каждой точке, как раньше), затем вырезаем регион.
                    sfc_var_specs = {
                        'sp': ['sp'],
                        't2m': ['t2m'],
                        'd2m': ['d2m'],
                        'rh2m': ['r2', 'rh2m'],
                        'u10': ['u10'],
                        'v10': ['v10'],
                        'orog': ['orog'],
                    }
                    sfc_arrays = {}  # key -> (lat_coord_sub, lon_coord_sub, values_2d)
                    for key, candidate_names in sfc_var_specs.items():
                        src_ds, found_name = _find_source_dataset(surface_datasets, candidate_names)
                        if src_ds is None:
                            continue
                        s_lat, s_lon, s_vals = _crop_and_extract(
                            src_ds, found_name, lat_min, lat_max, lon_min, lon_max)
                        sfc_arrays[key] = (s_lat, s_lon, s_vals)

                    # ---- Фронты по полю 850 гПа ----
                    # Считается один раз на весь регион: это сеточная величина,
                    # а не точечная — фронт виден только по горизонтальному
                    # градиенту, из одного зондирования его не достать.
                    # Фронты берём с ПЕРВОГО срока: за сутки они смещаются, и
                    # накладывать положение из четырёх моментов на одну карту
                    # значило бы рисовать несуществующую картину. Маска влияния
                    # на риск по той же причине фиксируется первым сроком.
                    if self.cb_fronts.get() and fronts is None:
                        try:
                            lvl_i = int(np.argmin(np.abs(pres_levels - 850.0)))
                            lvl_p = float(pres_levels[lvl_i])
                            t850 = t_arr[lvl_i] + 273.15          # обратно в Кельвины
                            u850 = u_arr[lvl_i] * KTS_TO_MS_LOCAL
                            v850 = v_arr[lvl_i] * KTS_TO_MS_LOCAL
                            tfp, gmag, adv = compute_fronts(t850, u850, v850,
                                                            iso_lat_sub, iso_lon_sub, lvl_p)
                            fronts = (iso_lat_sub, iso_lon_sub, tfp, gmag, adv, lvl_p)

                            # Близость к фронту нужна не только для рисования: фронт
                            # даёт вынужденный подъём, то есть работает спусковым
                            # механизмом там, где одного дневного прогрева не хватает.
                            # Считаем маску на сетке ФРОНТОВ и переносим на сетку
                            # расчёта — они разные, поэтому берём ближайший узел.
                            thr_f = max(float(np.percentile(gmag[np.isfinite(gmag)], 88)),
                                        3.0e-5)
                            band_f = (gmag > thr_f) & (tfp > 0)
                            from scipy.ndimage import binary_dilation, label as _label
                            # Убираем мелкие обрывки: фронт — протяжённая зона.
                            lab, nlab = _label(band_f)
                            for k in range(1, nlab + 1):
                                if int((lab == k).sum()) < 8:
                                    band_f[lab == k] = False
                            # «Рядом с фронтом» — примерно 150 км, характерная
                            # ширина зоны вынужденного подъёма перед фронтом.
                            cells = max(1, int(round(150.0 / (
                                abs(float(iso_lat_sub[1] - iso_lat_sub[0])) * 111.32))))
                            near_f = binary_dilation(band_f, iterations=cells)

                            front_near = np.zeros((n_lat, n_lon), dtype=bool)
                            for _i, _la in enumerate(lats):
                                _ii = _nearest_index_regular(iso_lat_sub, _la)
                                for _j, _lo in enumerate(lons):
                                    _jj = _nearest_index_regular(iso_lon_sub, _lo)
                                    front_near[_i, _j] = near_f[_ii, _jj]

                            self.log(f"🌀 Фронты: уровень {lvl_p:.0f} гПа, "
                                     f"узлов в зоне фронта {int(band_f.sum())}, "
                                     f"под влиянием фронта {int(front_near.sum())} "
                                     f"из {n_lat * n_lon} точек карты")
                        except Exception as e:
                            self.log(f"⚠️ Фронты не посчитались: {type(e).__name__}: {e}")

                    missing_sfc = [k for k in ('sp', 't2m', 'u10', 'v10') if k not in sfc_arrays]
                    if missing_sfc:
                        raise Exception(f"В GRIB не найдены обязательные приземные переменные: {missing_sfc}")

                    def _sfc_val(key, lat_q, lon_q):
                        """O(1)-выборка приземного значения по ближайшему узлу."""
                        if key not in sfc_arrays:
                            raise KeyError(key)
                        s_lat, s_lon, vals = sfc_arrays[key]
                        si = _nearest_index_regular(s_lat, lat_q)
                        sj = _nearest_index_regular(s_lon, lon_q)
                        val = float(vals[si, sj])
                        if np.isnan(val):
                            raise KeyError(key)
                        return val

                    grid_results = {k: np.zeros((n_lat, n_lon)) for k, v in selected_params.items() if v}

                    total_points = n_lat * n_lon
                    processed = 0
                    error_count = 0
                    MAX_LOGGED_ERRORS = 8  # чтобы не заспамить лог на больших сетках
                    dropped_logged = [0]  # счётчик залогированных сообщений о чистке высоты (список — для мутации из вложенного цикла)
                    point_jobs = []  # (i, j, lat, lon, pres, gh, tmp, dwpk, u, v) — уходят в пул процессов

                    # Поля на 850 гПа для выделения фронтов. Уровень выбран
                    # стандартный: ниже сказывается рельеф и приземный слой,
                    # выше фронтальная зона размывается.
                    fr_t = np.full((n_lat, n_lon), np.nan)
                    fr_u = np.full((n_lat, n_lon), np.nan)
                    fr_v = np.full((n_lat, n_lon), np.nan)
                    _lvl850 = int(np.argmin(np.abs(pres_levels - 850.0)))

                    self.status_label.config(text="Подготовка данных (чтение GRIB)...")

                    for i, lat in enumerate(lats):
                        for j, lon in enumerate(lons):
                            try:
                                ii = _nearest_index_regular(iso_lat_sub, lat)
                                jj = _nearest_index_regular(iso_lon_sub, lon)

                                # 1. Приземные параметры (приведение к гПа и °C)
                                sfc_p = _sfc_val('sp', lat, lon)
                                if sfc_p > 2000:
                                    sfc_p /= 100.0  # Паскали -> гПа

                                sfc_t = _sfc_val('t2m', lat, lon) - 273.15

                                try:
                                    sfc_td = _sfc_val('d2m', lat, lon) - 273.15
                                except KeyError:
                                    try:
                                        rh = _sfc_val('rh2m', lat, lon)
                                        sfc_td = float(_dewpoint_from_rh(sfc_t, rh))
                                    except KeyError:
                                        sfc_td = sfc_t - 4.0

                                sfc_td = min(sfc_td, sfc_t)  # Td не может превышать T

                                sfc_u = _sfc_val('u10', lat, lon) * 1.94384
                                sfc_v = _sfc_val('v10', lat, lon) * 1.94384

                                try:
                                    sfc_gh = _sfc_val('orog', lat, lon)
                                except KeyError:
                                    sfc_gh = 0.0

                                # 2. Высотные уровни — просто срез готового 3D-массива
                                # (level, lat, lon) по индексам ближайшего узла, без
                                # какого-либо повторного обращения к GRIB/xarray.
                                pres = pres_levels.copy()
                                tmp = t_arr[:, ii, jj].copy()
                                gh = gh_arr[:, ii, jj].copy()
                                u = u_arr[:, ii, jj].copy()
                                v = v_arr[:, ii, jj].copy()

                                if r_arr is not None:
                                    rh_col = r_arr[:, ii, jj]
                                    dwpk = _dewpoint_from_rh(tmp, rh_col)
                                elif d_arr is not None:
                                    dwpk = d_arr[:, ii, jj].copy()
                                else:
                                    dwpk = tmp - 10.0

                                dwpk = np.minimum(dwpk, tmp)

                                # Снимаем значения на 850 гПа до фильтрации: если
                                # уровень окажется под землёй (горы), останется NaN,
                                # и такая точка просто не участвует в поиске фронта.
                                if pres_levels[_lvl850] < sfc_p:
                                    fr_t[i, j] = t_arr[_lvl850, ii, jj]
                                    fr_u[i, j] = u_arr[_lvl850, ii, jj]
                                    fr_v[i, j] = v_arr[_lvl850, ii, jj]

                                # 3. Фильтрация подземных уровней (давление больше приземного)
                                valid_mask = pres < (sfc_p - 1.0)
                                pres = pres[valid_mask]
                                tmp = tmp[valid_mask]
                                dwpk = dwpk[valid_mask]
                                gh = gh[valid_mask]
                                u = u[valid_mask]
                                v = v[valid_mask]

                                # 4. Вставка поверхностного слоя
                                pres = np.insert(pres, 0, sfc_p)
                                tmp = np.insert(tmp, 0, sfc_t)
                                dwpk = np.insert(dwpk, 0, sfc_td)
                                gh = np.insert(gh, 0, sfc_gh)
                                u = np.insert(u, 0, sfc_u)
                                v = np.insert(v, 0, sfc_v)

                                # 5. Сортировка от большего давления к меньшему (от земли вверх)
                                sort_idx = np.argsort(pres)[::-1]
                                pres = pres[sort_idx]
                                tmp = tmp[sort_idx]
                                dwpk = dwpk[sort_idx]
                                gh = gh[sort_idx]
                                u = u[sort_idx]
                                v = v[sort_idx]

                                # Удаление возможных дубликатов по давлению.
                                # ВАЖНО: np.unique(..., return_index=True) возвращает индексы
                                # первого вхождения для значений, отсортированных ПО ВОЗРАСТАНИЮ.
                                # Поскольку pres на этом шаге уже отсортирован по УБЫВАНИЮ (шаг 5),
                                # эти индексы УЖЕ идут в перевёрнутом порядке относительно массива —
                                # достаточно просто отсортировать их по возрастанию (без [::-1]!),
                                # чтобы восстановить исходный убывающий порядок давления.
                                # Лишний [::-1] здесь переворачивал весь профиль вверх ногами
                                # (давление начинало расти, высота — падать), из-за чего фильтр
                                # монотонности высоты ниже отбрасывал почти все уровни.
                                _, unique_idx = np.unique(pres, return_index=True)
                                unique_idx = np.sort(unique_idx)

                                pres = pres[unique_idx]
                                tmp = tmp[unique_idx]
                                dwpk = dwpk[unique_idx]
                                gh = gh[unique_idx]
                                u = u[unique_idx]
                                v = v[unique_idx]

                                # 6b. Гарантируем строго возрастающую высоту снизу вверх —
                                # этого требует QC-проверка SHARPpy (DataQualityException
                                # 'Invalid height data'). Нарушения типичны на стыке
                                # приземного уровня (реальная орография) и ближайшего
                                # уровня давления, чья geopotential height из GFS иногда
                                # оказывается ниже/равна орографии в сложном рельефе.
                                # Отбрасываем уровни, не дающие строгого прироста высоты,
                                # сохраняя более нижний (обычно приземный) из конфликтующей пары.
                                keep_mask = np.ones(len(gh), dtype=bool)
                                last_gh = -1e9
                                for k in range(len(gh)):
                                    if gh[k] <= last_gh:
                                        keep_mask[k] = False
                                    else:
                                        last_gh = gh[k]

                                dropped = int(np.sum(~keep_mask))
                                if dropped > 0:
                                    pres = pres[keep_mask]
                                    tmp = tmp[keep_mask]
                                    dwpk = dwpk[keep_mask]
                                    gh = gh[keep_mask]
                                    u = u[keep_mask]
                                    v = v[keep_mask]
                                    if dropped_logged[0] < 5:
                                        self.log(f"ℹ️ Точка ({lat:.2f}, {lon:.2f}): отброшено {dropped} "
                                                 f"уровень(ей) из-за немонотонной высоты")
                                        dropped_logged[0] += 1

                                if len(pres) < 3:
                                    raise ValueError(f"Слишком мало уровней после чистки высоты ({len(pres)})")


                                # 6. Профиль и расчёт SHARPpy теперь НЕ делаем здесь —
                                # это уходит в отдельный процесс (см. _sharppy_point_worker
                                # ниже). Тут только складываем лёгкие numpy-массивы в очередь.
                                point_jobs.append((i, j, lat, lon, pres, gh, tmp, dwpk, u, v))

                            except Exception as e:
                                error_count += 1
                                if error_count <= MAX_LOGGED_ERRORS:
                                    self.log(f"⚠️ Точка ({lat:.2f}, {lon:.2f}): {type(e).__name__}: {e}")
                                elif error_count == MAX_LOGGED_ERRORS + 1:
                                    self.log("⚠️ ... дальнейшие ошибки по точкам больше не логируются ...")

                            processed += 1
                            if processed % 50 == 0 or processed == total_points:
                                progress_val = int((processed / total_points) * 100)
                                self.progress["value"] = progress_val
                                self.status_label.config(text=f"Подготовка данных: {progress_val}%")

                    ds_isobaric.close()
                    for sds in surface_datasets:
                        sds.close()

                    # ---------------- Параллельный расчёт SHARPpy ----------------
                    self.log(f"📦 Подготовлено {len(point_jobs)} точек, запускаю расчёт SHARPpy...")

                    try:
                        num_workers = max(1, int(self.workers_entry.get().strip()))
                    except Exception:
                        num_workers = max(1, (os.cpu_count() or 2) - 1)
                    num_workers = min(num_workers, max(1, len(point_jobs)))

                    self.log(f"⚙️ Процессов: {num_workers} (ядер/потоков доступно: {os.cpu_count()})")
                    self.status_label.config(text=f"Расчёт SHARPpy на {num_workers} процессах...")
                    self.progress["value"] = 0

                jobs_with_params = [job + (selected_params,) for job in point_jobs]
                total_jobs = len(jobs_with_params)
                sharppy_processed = 0
                point_diags = []  # (lat, lon, {'tornado': {...}, 'overall': {...}})

                if total_jobs > 0:
                    # chunksize побольше снижает накладные расходы на межпроцессный
                    # обмен для многочисленных мелких заданий.
                    chunksize = max(1, total_jobs // (num_workers * 8))
                    with mp.Pool(processes=num_workers) as pool:
                        for i, j, results, err in pool.imap_unordered(_sharppy_point_worker, jobs_with_params, chunksize=chunksize):
                            if err is not None:
                                error_count += 1
                                if error_count <= MAX_LOGGED_ERRORS:
                                    self.log(f"⚠️ Точка ({lats[i]:.2f}, {lons[j]:.2f}): {err}")
                                elif error_count == MAX_LOGGED_ERRORS + 1:
                                    self.log("⚠️ ... дальнейшие ошибки по точкам больше не логируются ...")
                            else:
                                for key, val in results.items():
                                    if key == '_diag':
                                        # Диагностика — не сеточная величина,
                                        # копится отдельно для лога.
                                        point_diags.append((lats[i], lons[j], val))
                                        continue
                                    grid_results[key][i, j] = val

                            sharppy_processed += 1
                            if sharppy_processed % 50 == 0 or sharppy_processed == total_jobs:
                                progress_val = int((sharppy_processed / total_jobs) * 100)
                                self.progress["value"] = progress_val
                                self.status_label.config(text=f"Расчёт SHARPpy: {progress_val}%")

                if error_count > 0:
                    self.log(f"⚠️ Всего точек с ошибками: {error_count} из {total_points} "
                             f"({100.0 * error_count / total_points:.1f}%)")
                if error_count == total_points:
                    self.log("❌ ВСЕ точки завершились ошибкой — карты будут полностью нулевые. "
                             "Смотрите сообщения об ошибках выше.")


                # Накопление максимума по срокам
                if daily_grids is None:
                    daily_grids = {k: v.copy() for k, v in grid_results.items()}
                else:
                    for k, v in grid_results.items():
                        daily_grids[k] = np.maximum(daily_grids[k], v)

            grid_results = daily_grids
            step_str = step_str_req

            # --- Постобработка outlook-сеток ---
            # До этого момента категории непрерывные (2.37, 3.81 и т.п.) —
            # именно поэтому не было "обрывов" на границах порогов. Теперь
            # округляем и убираем одиночные пики: реальный очаг имеет
            # размер, единичная точка 4-й категории посреди фона — это
            # почти всегда шум одного профиля, а не угроза.
            # ---- Фронт как усилитель риска ----
            # Осознанное ограничение: фронт поднимает категорию ТОЛЬКО там,
            # где риск уже есть (от 2 и выше). Фронт даёт подъём, но не
            # создаёт неустойчивость из ничего — иначе карта рисовала бы
            # риск вдоль всякой бароклинной зоны над сухим холодным
            # воздухом, где грозе взяться неоткуда.
            if (self.cb_fronts_risk.get() and front_near is not None
                    and front_near.any()):
                for okey in ('outlook_tornado', 'outlook_overall'):
                    if okey not in grid_results:
                        continue
                    g = grid_results[okey]
                    boost = front_near & (g >= 2.0)
                    n_up = int(boost.sum())
                    if n_up:
                        grid_results[okey] = np.where(
                            boost, np.minimum(g + 1.0, 4.0), g)
                        self.log(f"🌀 Фронт поднял {okey} в {n_up} точках "
                                 f"(только там, где риск уже был)")

            for okey in ('outlook_tornado', 'outlook_overall'):
                if okey in grid_results:
                    rounded = np.clip(np.round(grid_results[okey]), 1, 4)
                    coherent = _enforce_spatial_coherence(rounded)
                    n_demoted = int(np.sum(coherent < rounded))
                    if n_demoted:
                        self.log(f"🧹 {okey}: понижено {n_demoted} одиночных точек "
                                 f"(нет поддержки соседями)")
                    grid_results[okey] = coherent
            if 'outlook_trigger' in grid_results:
                trg = np.round(grid_results['outlook_trigger'])
                # Фронт как спусковой механизм. Дневной прогрев — не
                # единственный способ запустить конвекцию: у фронта воздух
                # поднимается принудительно, и там, где прогрева «на грани»
                # не хватало, фронт вопрос закрывает. Поэтому в зоне его
                # влияния ступень поднимается на единицу.
                if front_near is not None and front_near.any():
                    before = trg.copy()
                    trg = np.where(front_near, np.minimum(trg + 1, 3), trg)
                    n_up = int((trg > before).sum())
                    if n_up:
                        self.log(f"🌀 Фронт поднял оценку инициации "
                                 f"в {n_up} точках")
                grid_results['outlook_trigger'] = np.clip(trg, 1, 3)

            # Сводка по площади категорий — быстрый sanity-check прогона.
            for okey in ('outlook_tornado', 'outlook_overall'):
                if okey in grid_results:
                    g = grid_results[okey]
                    parts = [f"{OUTLOOK_NAMES_G[c]}: {int(np.sum(g == c))}"
                             for c in (2, 3, 4) if np.sum(g == c) > 0]
                    self.log(f"📊 {okey}: " + (", ".join(parts) if parts
                                                else "везде фоновая категория"))

            # Диагностика: почему самые опасные точки получили свою категорию.
            if point_diags:
                self.log(f"🔍 Разбор {min(5, len(point_diags))} наиболее "
                         f"выделяющихся точек (всего таких: {len(point_diags)}):")
                for plat, plon, d in point_diags[:5]:
                    if 'tornado' in d:
                        t = d['tornado']
                        self.log(f"   ({plat:.2f},{plon:.2f}) ТОР: STP={t['stp']:.2f} "
                                 f"SRH1={t['srh1k']:.0f} LCL={t['lcl']:.0f}м "
                                 f"сдвиг={t['deep_shear']:.0f}м/с режим={t['mode']} "
                                 f"ингр.={t['votes']}"
                                 + (f" | {'; '.join(t['notes'])}" if t['notes'] else ""))
                    if 'overall' in d:
                        o = d['overall']
                        top = max(o['sub'].items(), key=lambda kv: kv[1])
                        self.log(f"   ({plat:.2f},{plon:.2f}) OVR: ведущий сценарий "
                                 f"«{top[0]}»={top[1]:.2f}, все={o['sub']} "
                                 f"режим={o['mode']} ингр.={o['votes']}"
                                 + (f" | {'; '.join(o['notes'])}" if o['notes'] else ""))

            # --- Отрисовка карт ---
            self.log("Генерация карт...")
            self.status_label.config(text="Отрисовка карт...")

            lon_grid, lat_grid = np.meshgrid(lons, lats)

            # Якутск, все 34 райцентра улусов Якутии и ближайшие к Якутску
            # пригороды. Координаты приближённые (для сетки 0.25-0.5° точнее
            # не требуется) — это общегеографические, стабильные во времени
            # данные об административных центрах, не привязанные к погоде.
            SETTLEMENTS = [
                # --- Якутск (столица республики) ---
                ("Якутск", 62.0339, 129.7331),

                # --- Райцентры улусов (административные центры) ---
                ("Белая Гора", 68.53, 146.18),      # Абыйский улус
                ("Алдан", 58.61, 125.39),            # Алданский район
                ("Чокурдах", 70.62, 147.90),         # Аллаиховский улус
                ("Амга", 60.90, 131.98),             # Амгинский улус
                ("Саскылах", 71.96, 114.08),         # Анабарский улус
                ("Тикси", 71.64, 128.87),            # Булунский улус
                ("Верхневилюйск", 63.45, 120.32),    # Верхневилюйский улус
                ("Зырянка", 65.73, 150.87),          # Верхнеколымский улус
                ("Батагай", 67.63, 134.63),          # Верхоянский улус
                ("Вилюйск", 63.75, 121.63),          # Вилюйский улус
                ("Бердигестях", 62.08, 126.68),      # Горный улус
                ("Жиганск", 66.77, 123.37),          # Жиганский улус
                ("Сангар", 63.93, 127.47),           # Кобяйский улус
                ("Ленск", 60.72, 114.93),            # Ленский район
                ("Мирный", 62.54, 113.96),           # Мирнинский район
                ("Хонуу", 66.47, 143.22),            # Момский улус
                ("Намцы", 62.7167, 129.6667),        # Намский улус
                ("Нерюнгри", 56.66, 124.71),         # Нерюнгринский район
                ("Черский", 68.75, 161.30),          # Нижнеколымский улус
                ("Нюрба", 63.28, 118.33),            # Нюрбинский улус
                ("Усть-Нера", 64.57, 143.20),        # Оймяконский улус
                ("Олёкминск", 60.38, 120.42),        # Олёкминский улус
                ("Оленёк", 68.50, 112.43),           # Оленёкский улус
                ("Среднеколымск", 67.45, 153.68),    # Среднеколымский улус
                ("Сунтар", 62.15, 117.63),           # Сунтарский улус
                ("Хандыга", 62.65, 135.60),          # Томпонский улус
                ("Усть-Мая", 60.35, 134.53),         # Усть-Майский улус
                ("Депутатский", 69.30, 139.90),      # Усть-Янский улус
                ("Покровск", 61.48, 129.13),         # Хангаласский улус
                ("Чурапча", 62.00, 132.43),          # Чурапчинский улус
                ("Батагай-Алыта", 68.62, 130.40),    # Эвено-Бытантайский улус
            ]

            # ------------------------------------------------------------
            # Кастомные палитры по образцам, присланным пользователем.
            # Цвета подобраны на глаз по скриншотам-образцам (MUCAPE,
            # 0-6km Bulk Shear, STP Cosmo-Ru) — не пиксель-в-пиксель точное
            # соответствие, а близкий визуальный аналог с теми же порогами.
            # ------------------------------------------------------------

            # --- CAPE (по образцу MUCAPE): голубой -> зелёный -> жёлтый -> оранжевый -> красный -> пурпурный
            # Шкала по образцу карт WRF-ARW: холодные тона до 500,
            # зелёный до 1000, жёлтый до 1500, оранжевый до 2000,
            # красный выше и пурпурный на экстремумах.
            CAPE_BOUNDS = [100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000, 4000]
            CAPE_COLORS = [
                '#cfe6f5',   # 100-250   бледно-голубой
                '#7cb8dd',   # 250-500   голубой
                '#a9d18e',   # 500-750   светло-зелёный
                '#4f9d43',   # 750-1000  зелёный
                '#f2e14c',   # 1000-1250 жёлтый
                '#f5c518',   # 1250-1500 золотой
                '#f39222',   # 1500-1750 оранжевый
                '#e8641c',   # 1750-2000 тёмно-оранжевый
                '#d32f2f',   # 2000-2500 красный
                '#a01c1c',   # 2500-3000 тёмно-красный
                '#e05ce0',   # 3000-4000 пурпурный
            ]
            CAPE_CMAP = ListedColormap(CAPE_COLORS)
            CAPE_CMAP.set_under('white')     # < 100 Дж/кг
            CAPE_CMAP.set_over('#7b2d8e')    # > 4000 Дж/кг
            CAPE_NORM = BoundaryNorm(CAPE_BOUNDS, CAPE_CMAP.N)

            # --- Сдвиг ветра (стиль "0-6 km Bulk Shear"): от бледно-голубого до тёмно-синего, бордовый сверху
            SHEAR_COLORS = ['#e6f2ff', '#b3d9ff', '#80bfff', '#4da6ff', '#1a75ff', '#003d99']
            SHEAR_CMAP = ListedColormap(SHEAR_COLORS)
            SHEAR_CMAP.set_under('white')
            SHEAR_CMAP.set_over('#660000')

            # --- SCP и STP (по образцу STP Cosmo-Ru): нелинейная шкала 1-10, чёрный сверху
            STP_BOUNDS = [1, 1.5, 2, 3, 4, 5, 6, 8, 10]
            STP_COLORS = [
                '#deeaf6', '#a6c8e8', '#5b9bd5', '#2e5c8a',
                '#1a6b1a', '#6db56d', '#ffcc00', '#ff6600',
            ]
            STP_CMAP = ListedColormap(STP_COLORS)
            STP_CMAP.set_under('white')      # < 1 — практически нулевой потенциал
            STP_CMAP.set_over('black')       # > 10 — экстремум
            STP_NORM = BoundaryNorm(STP_BOUNDS, STP_CMAP.N)

            # --- Outlook: наши четыре категории, оформление в духе SPC.
            # Заливка полигонами с обводкой более тёмным оттенком,
            # легенда в углу вместо цветовой шкалы сбоку.
            OUTLOOK_NAMES = OUTLOOK_NAMES_G
            OUTLOOK_COLORS = ['#f5f5f5',   # 1 фоновый
                              '#2ca02c',   # 2 локальный
                              '#ff8c00',   # 3 очаговый
                              '#d62728']   # 4 критический
            OUTLOOK_EDGES = ['#cfcfcf', '#1e7a1e', '#c26a00', '#a51d1e']
            OUTLOOK_BOUNDS = [0.5, 1.5, 2.5, 3.5, 4.5]

            # --- CIN: своя шкала. Значения отрицательные, цвет насыщается
            # к БОЛЬШЕМУ модулю — чем сильнее крышка, тем заметнее.
            # Границы совпадают с гейтами движка: -38 (снижение категории)
            # и -70 (обнуление), чтобы на карте было видно, где они сработают.
            CIN_BOUNDS = [-300, -200, -150, -100, -70, -50, -38, -25, -10, 0]
            CIN_COLORS = ['#4a0d0d', '#7b1414', '#a82020', '#cc4b37',
                          '#e08b6a', '#efc0a4', '#f7ded0', '#eaf0f5', '#dfe8ee']
            CIN_CMAP = ListedColormap(CIN_COLORS)
            CIN_CMAP.set_under('#2d0606')      # < -300: крышка непробиваемая
            CIN_CMAP.set_over('#ffffff')       # ~0: крышки практически нет
            CIN_NORM = BoundaryNorm(CIN_BOUNDS, CIN_CMAP.N)

            # --- Триггер: отдельная 3-уровневая шкала (не риск, а "рванёт ли")
            TRIGGER_NAMES = {1: "Нужен фронт/динамика", 2: "На грани",
                             3: "Прогрева достаточно"}
            TRIGGER_COLORS = ['#e8e8e8', '#9ecae1', '#08519c']
            TRIGGER_CMAP = ListedColormap(TRIGGER_COLORS)
            TRIGGER_BOUNDS = [0.5, 1.5, 2.5, 3.5]
            TRIGGER_NORM = BoundaryNorm(TRIGGER_BOUNDS, TRIGGER_CMAP.N)

            # Статичная сводка критериев (не зависит от данных прогона, одна и
            # та же таблица-легенда для любого прогона) — идёт в общий список
            # фигур, сохранится вместе с картами через обычный _save_maps().
            if (selected_params.get('outlook_tornado') or selected_params.get('outlook_overall')
                    or selected_params.get('outlook_trigger')):
                legend_fig = self._build_outlook_legend_figure()
                self.generated_figs.append(("OUTLOOK_LEGEND", legend_fig))

            # Размер холста под форму области. Жёсткие 10x8 годились для
            # почти квадратного участка вокруг Якутска, но на всей
            # республике (58° по долготе против 17° по широте) карта
            # сохраняет пропорции местности и ужимается в полоску,
            # а вокруг остаётся пустое поле.
            _lon_span = max(0.1, lon_max - lon_min)
            _lat_span = max(0.1, lat_max - lat_min)
            _mid_lat = (lat_min + lat_max) / 2.0
            _mid_lon = (lon_min + lon_max) / 2.0

            # ПРОЕКЦИЯ. PlateCarree приравнивает градус долготы к градусу
            # широты, а на 63° с.ш. он вдвое короче — Якутия целиком
            # растягивается по горизонтали больше чем вдвое и выглядит
            # «блином». На небольшом участке вокруг Якутска разница
            # незаметна, поэтому там оставляем PlateCarree: он проще и
            # даёт привычную прямоугольную сетку.
            # Для крупных областей берём коническую равновеликую проекцию —
            # именно в ней республика выглядит так, как на обычных картах.
            _wide = _lon_span > 20.0
            if _wide:
                _proj = ccrs.AlbersEqualArea(
                    central_longitude=_mid_lon,
                    central_latitude=_mid_lat,
                    standard_parallels=(lat_min + _lat_span * 0.2,
                                        lat_max - _lat_span * 0.2))
                # В проекции долгота сжата примерно как cos(широты),
                # отсюда и настоящее соотношение сторон.
                _ratio = _lat_span / (_lon_span * max(0.2, np.cos(np.radians(_mid_lat))))
            else:
                _proj = ccrs.PlateCarree()
                _ratio = _lat_span / _lon_span

            _w = float(np.clip(np.sqrt(96.0 / max(_ratio, 0.15)), 8.0, 15.0))
            _h = float(np.clip(_w * _ratio + 1.3, 5.0, 12.0))

            mask_geom = None
            if self.cb_mask_yakutia.get():
                mask_geom = get_yakutia_geometry()
                if mask_geom is None:
                    self.log("⚠️ Границы Якутии не найдены в Natural Earth — "
                             "карты рисуются без затемнения соседних регионов.")
                else:
                    self.log("🗺 Соседние регионы будут приглушены.")

            for param_key, data in grid_results.items():
                # add_axes/colorbar/title вызываем ЯВНО через объекты fig и ax,
                # а не через plt.*. Функции plt.* работают с «текущими осями»,
                # а plt.colorbar делает текущими СВОИ оси — после него
                # plt.title писал заголовок на цветовой шкале, а не на карте.
                fig = plt.figure(figsize=(_w, _h))
                # add_subplot, а не add_axes: с cartopy add_axes на части
                # версий matplotlib создаёт оси, которые не попадают в
                # обычный рендеринг — карта пропадает, а цветовая шкала,
                # живущая на СВОИХ осях, остаётся. Именно так выглядела
                # поломка: одни шкалы без карт.
                # Явное место под карту. По умолчанию matplotlib оставляет
                # широкие поля, а с bbox_inches="tight" они срезались —
                # теперь, когда обрезки нет, их надо задать самим, иначе
                # карта висит в пустоте.
                ax = fig.add_axes([0.055, 0.055, 0.83, 0.88], projection=_proj)

                # Порядок слоёв снизу вверх:
                #   2..6   заливки риска и параметров
                #   12     затемнение вне Якутии
                #   16     озёра, побережье, границы государств
                #   17     реки
                #   18     контур республики
                #   20     города, 25 — заголовок и легенда
                # Реки выше затемнения нарочно: по ним читается местность,
                # и терять их под серым нельзя.
                ax.add_feature(cfeature.LAKES, alpha=0.6, zorder=16)
                ax.add_feature(cfeature.COASTLINE, linewidth=1.2, zorder=16)
                ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.8, zorder=16)
                ax.add_feature(cfeature.RIVERS, alpha=0.8, linewidth=0.7, zorder=17)

                param_lower = param_key.lower()
                # Определяем ЗДЕСЬ: дальше на этот флаг опираются и сетка
                # координат, и линия фронта, и подписи городов. Раньше он
                # задавался только в блоке городов, то есть ниже первого
                # использования — отсюда «referenced before assignment».
                spc_style = ("outlook" in param_lower and "trigger" not in param_lower)

                if "srh" in param_lower:
                    # SRH может быть отрицательным (антициклоническая завихренность) —
                    # используем симметричный диапазон вокруг нуля и диверг. колормап,
                    # иначе отрицательные значения обрежутся до 0 и пропадут с карты.
                    # У RdBu_r ноль и так белый по центру палитры — отдельно красить не нужно.
                    abs_max = max(np.max(np.abs(data)), 50)
                    levels = np.linspace(-abs_max, abs_max, 21)
                    contour = ax.contourf(lon_grid, lat_grid, data, levels=levels, cmap="RdBu_r",
                                           transform=ccrs.PlateCarree(), extend='both')

                elif "cape" in param_lower:
                    # Фиксированная шкала по образцу MUCAPE (Дж/кг), одинаковая
                    # для всех CAPE-параметров, чтобы карты были сравнимы между собой.
                    contour = ax.contourf(lon_grid, lat_grid, data, levels=CAPE_BOUNDS,
                                           cmap=CAPE_CMAP, norm=CAPE_NORM,
                                           transform=ccrs.PlateCarree(), extend='both')

                elif "shear" in param_lower:
                    # Стиль/цвета по образцу "0-6 km Bulk Shear" (белый -> оттенки
                    # синего -> бордовый). Границы уровней — АДАПТИВНЫЕ под
                    # реальный максимум конкретной карты (0-1/0-3/0-6 км дают очень
                    # разный диапазон значений, фиксированная шкала 0-40 м/с из
                    # примера "убила" бы 0-1км карту в один сплошной цвет).
                    max_val = max(np.max(data), 10.0)
                    shear_bounds = np.linspace(0.3, max_val, len(SHEAR_COLORS) + 1)
                    shear_norm = BoundaryNorm(shear_bounds, SHEAR_CMAP.N)
                    contour = ax.contourf(lon_grid, lat_grid, data, levels=shear_bounds,
                                           cmap=SHEAR_CMAP, norm=shear_norm,
                                           transform=ccrs.PlateCarree(), extend='both')

                elif "cin" in param_lower:
                    contour = ax.contourf(lon_grid, lat_grid, data,
                                           levels=CIN_BOUNDS,
                                           cmap=CIN_CMAP, norm=CIN_NORM,
                                           transform=ccrs.PlateCarree(),
                                           extend='both')

                elif "outlook" in param_lower:
                    try:
                        from scipy.ndimage import gaussian_filter
                        smoothed = gaussian_filter(data, sigma=0.9)
                    except ImportError:
                        self.log("⚠️ scipy не найден — outlook рисуется без сглаживания "
                                 "(pip install scipy, если нужно сглаживание)")
                        smoothed = data

                    if "trigger" in param_lower:
                        # Тем же способом, что и карты риска: заливка
                        # полигонами, легенда в углу, БЕЗ цветовой шкалы.
                        # Шкала сбоку была единственным, чем эта карта
                        # отличалась от остальных outlook-карт, — и
                        # единственной, которая не отрисовывалась.
                        contour = None
                        for lv in (1, 2, 3):
                            if not np.any(np.round(data) >= lv):
                                continue
                            ax.contourf(lon_grid, lat_grid, smoothed,
                                        levels=[lv - 0.5, 3.6],
                                        colors=[TRIGGER_COLORS[lv - 1]],
                                        transform=ccrs.PlateCarree(), zorder=2 + lv)
                    else:
                        # Заливка + обводка каждой ступени отдельно: так
                        # получается вид полигонов, как на картах SPC,
                        # а не размытая растровая заливка.
                        contour = None
                        # Наличие категории проверяем по ТОМУ ЖЕ полю, по
                        # которому рисуем. Раньше проверка шла по исходным
                        # данным, а рисование по сглаженным: одиночные точки
                        # после сглаживания опускались ниже порога, заливка
                        # выходила нулевой ширины и невидимой, а контурная
                        # линия всё равно рисовалась — отсюда оранжевые
                        # линии посреди пустой карты.
                        # Плюс порог по площади: пятно меньше нескольких
                        # узлов сетки — это шум, а не очаг.
                        min_cells = max(3, int(smoothed.size * 0.0015))
                        for lv in range(2, 5):
                            mask = smoothed >= (lv - 0.5)
                            if int(mask.sum()) < min_cells:
                                continue
                            ax.contourf(lon_grid, lat_grid, smoothed,
                                        levels=[lv - 0.5, 4.6],
                                        colors=[OUTLOOK_COLORS[lv - 1]],
                                        transform=ccrs.PlateCarree(), zorder=2 + lv)
                            ax.contour(lon_grid, lat_grid, smoothed,
                                       levels=[lv - 0.5], colors=[OUTLOOK_EDGES[lv - 1]],
                                       linewidths=1.4,
                                       transform=ccrs.PlateCarree(), zorder=2 + lv)

                else:  # scp, stp
                    # Фиксированная шкала по образцу STP (COSMO-Ru), одинаковая
                    # для SCP и STP, как и просили.
                    contour = ax.contourf(lon_grid, lat_grid, data, levels=STP_BOUNDS,
                                           cmap=STP_CMAP, norm=STP_NORM,
                                           transform=ccrs.PlateCarree(), extend='both')

                # ---- Линии фронтов ----
                # Рисуем ПОВЕРХ заливок и затемнения (zorder 19), но под
                # городами: фронт — это привязка к синоптике, он должен
                # читаться на любой карте, включая outlook.
                if fronts is not None and self.cb_fronts_draw.get():
                    try:
                        f_lat, f_lon, tfp, gmag, adv, lvl_p = fronts
                        FLON, FLAT = np.meshgrid(f_lon, f_lat)
                        # Линия фронта: тёплая кромка бароклинной зоны.
                        # Требуем И заметный градиент, И положительный TFP,
                        # иначе линии полезут по любому слабому перепаду.
                        # Порог по градиенту — АДАПТИВНЫЙ, по процентилю
                        # внутри области. Фиксированные 2e-5 K/м (это всего
                        # 2 K на 100 км) в бароклинной обстановке проходит
                        # почти вся карта, и нулевой контур TFP рисуется
                        # по каждому изгибу поля — отсюда была сетка синих
                        # линий вместо линии фронта. Берём верхние 12%
                        # градиента, но не ниже физически осмысленного
                        # минимума в 3 K на 100 км.
                        finite = gmag[np.isfinite(gmag)]
                        if finite.size < 20:
                            raise ValueError("мало данных для фронтов")
                        thr = max(float(np.percentile(finite, 88)), 3.0e-5)
                        band = gmag > thr
                        # Отсев обрывков: фронт — протяжённая зона, а пятна
                        # в несколько узлов дают короткие «палочки», которые
                        # засоряют карту и ничего не означают.
                        from scipy.ndimage import label as _lbl
                        _l, _n = _lbl(band)
                        for _k in range(1, _n + 1):
                            if int((_l == _k).sum()) < 10:
                                band[_l == _k] = False
                        # Холодный и тёплый рисуем отдельно — по знаку
                        # адвекции на том же уровне.
                        for sign, colour, label in ((-1, '#1f5fbf', 'холодный'),
                                                     (1, '#c62828', 'тёплый')):
                            sel = np.where(band & ((adv * sign) > 0), tfp, np.nan)
                            # Требуем ощутимую зону: короткие обрывки —
                            # это шум поля, а не фронт.
                            if np.isfinite(sel).sum() < 25:
                                continue
                            cs = ax.contour(FLON, FLAT, sel, levels=[1.0e-10],
                                            colors=[colour], linewidths=2.8,
                                            transform=ccrs.PlateCarree(), zorder=19)

                            # Зубцы вдоль линии: треугольники у холодного
                            # фронта, полукруги у тёплого — как на
                            # синоптических картах. Ставим по длине пути
                            # через равные промежутки, а не по узлам сетки,
                            # иначе они сбиваются в кучу на изгибах.
                            try:
                                paths = [pp for col in cs.collections
                                         for pp in col.get_paths()]
                            except AttributeError:      # matplotlib 3.8+
                                paths = cs.get_paths()
                            for pth in paths:
                                v = pth.vertices
                                if len(v) < 8:
                                    continue          # слишком короткий кусок
                                seg = np.hypot(np.diff(v[:, 0]), np.diff(v[:, 1]))
                                arc = np.concatenate([[0], np.cumsum(seg)])
                                if arc[-1] < 1.5:
                                    continue          # короче ~1.5° — не рисуем
                                for dist in np.arange(0.7, arc[-1], 1.4):
                                    k = int(np.searchsorted(arc, dist))
                                    if k >= len(v):
                                        break
                                    ax.plot(v[k, 0], v[k, 1],
                                            marker='^' if sign < 0 else 'o',
                                            color=colour, markersize=7,
                                            markeredgecolor='white',
                                            markeredgewidth=0.6,
                                            transform=ccrs.PlateCarree(),
                                            zorder=19)
                        _fr_note = (" · учтены в риске"
                                    if self.cb_fronts_risk.get() else "")
                        ax.text(0.02, 0.02,
                                f"фронты по θ на {lvl_p:.0f} гПа "
                                f"(|∇θ| > {thr * 1e5:.1f} K/100км){_fr_note}",
                                transform=ax.transAxes, fontsize=7.5,
                                color='#333333', zorder=25,
                                bbox=dict(facecolor='white', alpha=0.8,
                                          edgecolor='none', pad=2))
                    except Exception as e:
                        self.log(f"⚠️ Линии фронтов не нарисовались: {type(e).__name__}")

                # Сетка координат с подписями — по ней читается положение
                # очага без привязки к городам.
                if not spc_style:
                    try:
                        gl = ax.gridlines(draw_labels=True, linewidth=0.4,
                                          color='#9aa4ad', alpha=0.5, linestyle=':')
                        gl.top_labels = False
                        gl.right_labels = False
                        gl.xlabel_style = {'size': 8}
                        gl.ylabel_style = {'size': 8}
                    except Exception:
                        pass

                # Подписи локальных максимумов: цифра рядом с очагом
                # читается быстрее, чем подбор оттенка по шкале.
                #
                # Порог свой для каждого типа параметра — иначе на картах
                # сдвига (единицы м/с) не подписалось бы ничего, а на SRH
                # с отрицательными значениями подписи ушли бы в мусор.
                _peak_cfg = None
                if "cape" in param_lower:
                    _peak_cfg = dict(thr=500, fmt="{:.0f}", signed=False)
                elif "shear" in param_lower:
                    # 12 м/с — примерно p75 вашей климатологии, ниже
                    # подписывать нечего.
                    _peak_cfg = dict(thr=12.0, fmt="{:.0f}", signed=False)
                elif "srh" in param_lower:
                    # SRH бывает и отрицательным (антициклоническое
                    # вращение), поэтому ищем экстремумы ПО МОДУЛЮ, а знак
                    # сохраняем в подписи: -180 не менее важно, чем +180.
                    _peak_cfg = dict(thr=80.0, fmt="{:.0f}", signed=True)
                elif "scp" in param_lower or "stp" in param_lower:
                    _peak_cfg = dict(thr=1.0, fmt="{:.1f}", signed=False)
                elif "cin" in param_lower:
                    # У CIN интересен не максимум, а МИНИМУМ — самая
                    # сильная крышка. Ищем по модулю, знак сохраняем.
                    _peak_cfg = dict(thr=50.0, fmt="{:.0f}", signed=True)

                if _peak_cfg is not None:
                    try:
                        from scipy.ndimage import maximum_filter
                        field = np.abs(data) if _peak_cfg['signed'] else data
                        peaks = ((field == maximum_filter(field, size=7))
                                 & (field >= _peak_cfg['thr']))
                        idx = sorted(np.argwhere(peaks),
                                     key=lambda ij: -field[ij[0], ij[1]])
                        taken = []
                        for i_, j_ in idx:
                            if len(taken) >= 8:
                                break
                            if any(abs(i_ - a) < 6 and abs(j_ - b) < 6
                                   for a, b in taken):
                                continue
                            taken.append((i_, j_))
                            ax.text(lons[j_], lats[i_],
                                    _peak_cfg['fmt'].format(data[i_, j_]),
                                    fontsize=8, fontweight='bold', color='white',
                                    ha='center', va='center', zorder=21,
                                    transform=ccrs.PlateCarree(),
                                    # обводка: белая цифра на светлой заливке
                                    # иначе не читается
                                    path_effects=[pe.withStroke(linewidth=2.2,
                                                                foreground='#333333')])
                    except Exception:
                        pass

                if "trigger" in param_lower:
                    from matplotlib.patches import Rectangle as _Rect
                    # ВНИМАНИЕ: не занимать имена _w/_h — это размеры холста,
                    # посчитанные до цикла и нужные всем следующим картам.
                    _trg_handles = [_Rect((0, 0), 1, 1,
                                          facecolor=TRIGGER_COLORS[k - 1],
                                          edgecolor='#666666', linewidth=0.8)
                                    for k in (3, 2, 1)]
                    _lg = ax.legend(_trg_handles,
                                    [TRIGGER_NAMES[k] for k in (3, 2, 1)],
                                    loc='lower right', fontsize=9, framealpha=0.95,
                                    handlelength=1.6, handleheight=1.1,
                                    borderpad=0.7, labelspacing=0.45,
                                    edgecolor='#333333')
                    _lg.set_zorder(25)
                elif "outlook" in param_lower:
                    # Легенда рисуется в углу карты ниже — цветовая шкала
                    # сбоку для ступенчатой категории только мешает.
                    pass
                else:
                    # Шкале выделяем СОБСТВЕННЫЕ оси заранее. Вызов
                    # fig.colorbar(..., ax=ax) отбирает место у исходных
                    # осей, переставляя их position — а с GeoAxes из
                    # cartopy это на части версий ломает отрисовку: карта
                    # исчезает, остаётся одна шкала. Ровно это и
                    # происходило со всеми картами параметров.
                    try:
                        cax = fig.add_axes([0.90, 0.14, 0.022, 0.70])
                        fig.colorbar(contour, cax=cax, orientation="vertical")
                    except Exception as e:
                        self.log(f"⚠️ Шкала не построилась ({type(e).__name__}), "
                                 f"карта сохранена без неё")

                # Населённые пункты. На outlook-картах — крестиком и без
                # подложки, как на картах SPC: заливка там сплошная, и
                # серые таблички её только рвут. На остальных картах
                # подложка нужна, иначе текст теряется на пёстром фоне.
                # На крупной области подписи городов налезают друг на друга.
                # Прореживаем жадно: пункт рисуется, только если рядом ещё
                # ничего не подписано. Порог привязан к размеру области,
                # поэтому на масштабе Якутска ничего не теряется.
                _min_sep = _lon_span * 0.035
                _label_fs = 6.5 if _lon_span < 20 else 7.5
                _drawn = []
                for name, slat, slon in SETTLEMENTS:
                    if lon_min <= slon <= lon_max and lat_min <= slat <= lat_max:
                        if any(abs(slon - dl) < _min_sep and abs(slat - da) < _min_sep * 0.6
                               for da, dl in _drawn):
                            continue
                        _drawn.append((slat, slon))
                        if spc_style:
                            ax.plot(slon, slat, marker='+', color='#222222',
                                    markersize=5, markeredgewidth=1.1,
                                    transform=ccrs.PlateCarree(), zorder=20)
                            ax.text(slon + _min_sep * 0.25, slat, name,
                                    fontsize=_label_fs,
                                    color='#111111', transform=ccrs.PlateCarree(),
                                    zorder=20, ha='left', va='center')
                        else:
                            ax.plot(slon, slat, marker='o', color='black',
                                    markersize=4, markeredgecolor='white',
                                    markeredgewidth=0.8,
                                    transform=ccrs.PlateCarree(), zorder=6)
                            ax.text(slon + 0.03, slat + 0.02, name, fontsize=5,
                                    color='#111111', transform=ccrs.PlateCarree(),
                                    zorder=6, ha='left', va='bottom',
                                    bbox=dict(facecolor='#d0d0d0', alpha=0.85,
                                              edgecolor='none', pad=1.0))

                ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

                # Приглушаем всё за пределами республики: вырезаем Якутию
                # из прямоугольника карты и заливаем остаток. zorder=12 —
                # выше заливок риска (2..6), но ниже рек и подписей, чтобы
                # соседние регионы оставались читаемыми, просто тусклыми.
                if mask_geom is not None:
                    try:
                        from shapely.geometry import box as _box
                        # Запас берём БОЛЬШОЙ. В конической проекции углы
                        # холста уходят далеко за пределы прямоугольника
                        # в градусах: при запасе в 2° там оставались белые
                        # клинья, которые нечем было залить. Лишняя площадь
                        # всё равно обрезается рамкой карты.
                        _pad_lon = max(45.0, _lon_span * 1.2)
                        _pad_lat = max(20.0, _lat_span * 1.2)
                        outside = _box(lon_min - _pad_lon,
                                       max(-89.0, lat_min - _pad_lat),
                                       lon_max + _pad_lon,
                                       min(89.0, lat_max + _pad_lat)).difference(mask_geom)
                        # Заливка НЕПРОЗРАЧНАЯ. При полупрозрачной сквозь
                        # неё просвечивал край сетки данных: область расчёта
                        # задана прямоугольником в градусах, а в конической
                        # проекции его края становятся дугами — и получался
                        # шов, похожий на вторую кривую маску. Сплошной цвет
                        # убирает это целиком.
                        ax.add_geometries([outside], crs=ccrs.PlateCarree(),
                                          facecolor='#c9d0d6', edgecolor='none',
                                          zorder=12)
                        ax.add_geometries([mask_geom], crs=ccrs.PlateCarree(),
                                          facecolor='none', edgecolor='#3d4750',
                                          linewidth=1.8, zorder=18)
                    except Exception as e:
                        self.log(f"⚠️ Маска Якутии не наложилась: {type(e).__name__}")

                if spc_style:
                    # --- заголовок в рамке слева вверху ---
                    if len(run_list) > 1:
                        valid = (f"{date_str} — сутки от {hour_str}:00 UTC "
                                 f"(+{run_list[0]}…+{run_list[-1]}ч)")
                    else:
                        valid = f"{date_str} {hour_str}:00 UTC +{step_str}ч"
                    what = ("Торнадо-риск" if "tornado" in param_lower
                            else "Опасные явления")
                    ax.text(0.02, 0.975, f"{what}\n{valid}",
                            transform=ax.transAxes, zorder=25,
                            fontsize=12, fontweight='bold', va='top', ha='left',
                            bbox=dict(facecolor='white', edgecolor='#333333',
                                      linewidth=0.9, pad=6))
                    ax.text(0.98, 0.975,
                            f"модель: {self.current_model}\n"
                            f"пороги: {'климатология' if 'клим' in CLIMO_SOURCE else 'SPC (США)'}",
                            transform=ax.transAxes, zorder=25, fontsize=7.5,
                            va='top', ha='right', color='#333333',
                            bbox=dict(facecolor='white', edgecolor='none',
                                      alpha=0.85, pad=3))

                    # --- легенда в правом нижнем углу ---
                    from matplotlib.patches import Rectangle
                    handles = [Rectangle((0, 0), 1, 1,
                                         facecolor=OUTLOOK_COLORS[k - 1],
                                         edgecolor=OUTLOOK_EDGES[k - 1], linewidth=1.0)
                               for k in range(4, 1, -1)]
                    labels = [f"{k}  {OUTLOOK_NAMES[k]}" for k in range(4, 1, -1)]
                    leg = ax.legend(handles, labels, loc='lower right',
                                    fontsize=9, framealpha=0.95, handlelength=1.6,
                                    handleheight=1.1, borderpad=0.7,
                                    labelspacing=0.45, edgecolor='#333333')
                    leg.set_zorder(25)

                title_names = {
                    'mlcin': 'MLCIN — крышка (перемешанный слой)',
                    'sbcin': 'SBCIN — крышка (приземная частица)',
                    'outlook_tornado': 'OUTLOOK: ТОРНАДО-РИСК',
                    'outlook_trigger': 'OUTLOOK: ТРИГГЕР (реальность инициации)',
                    'outlook_overall': 'OUTLOOK: OVERALL-РИСК',
                }
                title_param = title_names.get(param_key, param_key.upper())
                if not spc_style:
                    # В суточном режиме максимум берётся по ВСЕМ сеткам,
                    # включая CAPE и сдвиг, — значит и подпись должна
                    # говорить о сутках, а не об одном сроке. Иначе карта
                    # показывает максимум за день, а числится прогнозом
                    # на +12 ч.
                    if len(run_list) > 1:
                        stamp = (f"{date_str} — макс. за сутки от {hour_str}:00 "
                                 f"UTC (+{run_list[0]}…+{run_list[-1]}ч)")
                    else:
                        stamp = f"{date_str} {hour_str}:00 UTC +{step_str}h"
                    # Через fig.text, а не ax.set_title: заголовок на осях
                    # с подписями сетки координат уезжает за верхний край.
                    fig.text(0.5, 0.965,
                             f"{self.current_model} {title_param} | {stamp}",
                             fontsize=13, fontweight="bold", ha='center', va='top')

                self.generated_figs.append((param_key, fig))

            # Проверка: если оси карты не попали в фигуру, на выходе будет
            # файл с одной цветовой шкалой. Такое уже случалось, и молча
            # отдавать пустую картинку нельзя — лучше сказать сразу.
            _empty = []
            for _pk, _fig in self.generated_figs:
                if _pk.upper() == "OUTLOOK_LEGEND":
                    continue
                _mx = [a for a in _fig.axes if hasattr(a, 'projection')]
                if not _mx:
                    _empty.append(f"{_pk}: нет осей карты")
                elif not (_mx[0].collections or _mx[0].images or _mx[0].lines):
                    _empty.append(f"{_pk}: оси пустые")
            if _empty:
                self.log("⚠️ Карты без содержимого: " + "; ".join(_empty[:5]))
                self.log("   Похоже на несовместимость cartopy с этой версией "
                         "matplotlib — в файле останется только цветовая шкала.")

            self.log("✅ Расчет и построение карт успешно завершены!")
            self.status_label.config(text="Готово!")
            self.btn_save.config(state="normal")

        except Exception as e:
            self.log(f"❌ ОШИБКА: {str(e)}")
            messagebox.showerror("Ошибка", str(e))
            self.status_label.config(text="Ошибка выполнения")

        finally:
            self.btn_run.config(state="normal")

    def _build_outlook_legend_figure(self):
        """
        Статичная таблица критериев outlook-риска (не зависит от данных
        конкретного прогона — одни и те же пороги всегда). Два блока:
        сверху понятное объяснение "для всех", снизу техническая часть
        с формулами/порогами для тех, кому интересны детали.
        """
        import textwrap

        fig, axes = plt.subplots(2, 1, figsize=(13, 9), gridspec_kw={'height_ratios': [1, 1.1]})
        ax_plain, ax_tech = axes
        ax_plain.axis('off')
        ax_tech.axis('off')

        row_colors = ['#f5f5f5', '#2ca02c', '#ff8c00', '#d62728']

        # --- Блок 1: понятным языком ---
        plain_labels = ["Категория", "Что это значит", "Что может произойти"]
        plain_rows = [
            ["Фоновый",
             "Гроза в этот день/час маловероятна",
             "Обычная погода, опасных явлений не ожидается"],
            ["Локальный",
             "Возможны отдельные обычные грозы",
             "Кратковременный дождь/гроза локально, без серьёзной опасности"],
            ["Очаговый",
             "Вероятны сильные грозы в отдельных районах",
             "Шквалистый ветер, крупный град, при неудачном стечении — слабый смерч"],
            ["Критический",
             "Высокая вероятность опасной грозовой погоды",
             "Очень сильные шквалы, крупный/очень крупный град, повышенный риск смерча"],
        ]
        plain_rows_wrapped = [
            [cell if k == 0 else "\n".join(textwrap.wrap(cell, 34)) for k, cell in enumerate(row)]
            for row in plain_rows
        ]
        table1 = ax_plain.table(cellText=plain_rows_wrapped, colLabels=plain_labels,
                                 loc='center', cellLoc='left',
                                 colWidths=[0.17, 0.37, 0.46])
        table1.auto_set_font_size(False)
        table1.set_fontsize(11)
        table1.scale(1, 3.2)
        for r in range(len(plain_rows)):
            table1[(r + 1, 0)].set_facecolor(row_colors[r])
            table1[(r + 1, 0)].set_text_props(fontweight="bold")
        ax_plain.set_title(
            "Outlook — что означают категории (простыми словами)",
            fontsize=13, fontweight="bold", pad=14
        )

        # --- Блок 2: технические пороги (для тех, кому интересно) ---
        tech_labels = ["Категория", "STP (торнадо-карта)", "Overall (SigSvr / DCAPE / WNDG / SHERB)"]
        tech_rows = [
            ["Фоновый",     "STP = 0",
             "все сценарии ниже порога"],
            ["Локальный",   "0 < STP ≤ 1",
             "SigSvr / DCAPE / WNDG у нижнего порога климатологии"],
            ["Очаговый",    "1 < STP ≤ 2",
             "середина между нижним и верхним порогом; SHERB>1 при MUCAPE≤1000"],
            ["Критический", "STP > 2, доп: SRH(0-1км), либо SRH(0-3км) 100-150, либо эфф.SRH≥100",
             "сценарии у верхнего порога климатологии"],
        ]
        tech_rows_wrapped = [
            [cell if k == 0 else "\n".join(textwrap.wrap(cell, 40)) for k, cell in enumerate(row)]
            for row in tech_rows
        ]
        table2 = ax_tech.table(cellText=tech_rows_wrapped, colLabels=tech_labels,
                                loc='upper center', cellLoc='left',
                                colWidths=[0.15, 0.36, 0.49])
        table2.auto_set_font_size(False)
        table2.set_fontsize(8.5)
        table2.scale(1, 2.6)
        for r in range(len(tech_rows)):
            table2[(r + 1, 0)].set_facecolor(row_colors[r])

        # Подпись держим в две строки: длинный текст сжимает таблицы
        # и текст в ячейках перестаёт помещаться.
        ax_tech.set_title(
            "Технические пороги (для тех, кому интересны детали)\n"
            "Гейты: MLCIN < -70 → обнуление; MLCIN и SBCIN оба < -38 → минус категория; "
            "PWAT < 16мм → потолок Локальный\n"
            "Фронт (если включён) поднимает категорию на 1 в полосе ~150 км, "
            "но только там, где риск уже был: подъём не создаёт неустойчивость",
            fontsize=9.5, fontweight="bold", pad=10
        )
        # Происхождение порогов — отдельной строкой внизу, чтобы не
        # раздувать заголовок.
        fig.text(0.5, 0.012, CLIMO_SOURCE, fontsize=8.5, ha='center',
                 color='#555555')

        fig.tight_layout()
        return fig

    def _save_maps(self):
        base_output_dir = self.dir_entry.get().strip()

        date_str = self.date_entry.get().strip().replace("-", "")
        hour_str = self.hour_entry.get().strip().zfill(2)
        step_str = self.step_entry.get().strip().zfill(3)

        year_str = date_str[:4]
        month_str = date_str[4:6]

        # Структура: <Папка>\<МОДЕЛЬ>\<ГОД>\<МЕСЯЦ>\<ЧАС>\<Риски|Параметры>\<ИМЯ>_...png
        # Outlook-карты (и их легенда) — в "Риски", всё остальное — в "Параметры".
        base_dir = os.path.join(base_output_dir, self.current_model, year_str, month_str, hour_str)
        risk_dir = os.path.join(base_dir, "Риски")
        param_dir = os.path.join(base_dir, "Параметры")

        # Легенда одинакова для любого прогона — она описывает пороги, а не
        # погоду. Поэтому лежит ОДНИМ файлом с постоянным именем в корне
        # папки модели и переписывается при каждом расчёте. Иначе рядом с
        # каждой датой копилась бы её копия, и через месяц папка была бы
        # завалена одинаковыми картинками.
        legend_dir = os.path.join(base_output_dir, self.current_model)

        saved_count = 0
        used_dirs = set()
        for param_key, fig in self.generated_figs:
            key = param_key.upper()
            if key == "OUTLOOK_LEGEND":
                os.makedirs(legend_dir, exist_ok=True)
                filepath = os.path.join(legend_dir, "OUTLOOK_LEGEND.png")
            else:
                target_dir = risk_dir if key.startswith("OUTLOOK") else param_dir
                os.makedirs(target_dir, exist_ok=True)
                used_dirs.add(target_dir)
                filepath = os.path.join(
                    target_dir, f"{key}_{date_str}_{hour_str}_F{step_str}.png")

            # bbox_inches="tight" обрезает по видимому содержимому. Если на
            # карте что-то не отрисовалось, от файла остаётся одна цветовая
            # шкала — именно так и выглядела сломанная карта триггера.
            # Задаём явные поля: тогда пустая карта видна как пустая, а не
            # маскируется обрезкой.
            # Сохраняем БЕЗ bbox_inches="tight": эта обрезка режет картинку
            # по видимому содержимому, и если оси карты по какой-то причине
            # оказались за пределами области отрисовки, от файла остаётся
            # одна шкала. С фиксированными полями хотя бы видно, что именно
            # получилось, — пустая карта или сдвинутая.
            fig.savefig(filepath, dpi=200, facecolor="white")
            plt.close(fig)
            saved_count += 1

        self.log(f"💾 Сохранено карт: {saved_count} в {base_dir} (по подпапкам Риски/Параметры)")
        if any(k.upper() == "OUTLOOK_LEGEND" for k, _ in self.generated_figs):
            self.log(f"📋 Легенда обновлена: {os.path.join(legend_dir, 'OUTLOOK_LEGEND.png')}")
        if sys.platform.startswith("win") and used_dirs:
            os.startfile(base_dir)


if __name__ == "__main__":
    mp.freeze_support()  # безопасно и для обычного запуска, и для .exe (PyInstaller)
    app = GFSMapApp()
    app.mainloop()
