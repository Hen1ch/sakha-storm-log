#!/usr/bin/env python3
"""
sharppy_core — ОБЩЕЕ ЯДРО: загрузка данных, расчёт вердиктов, Skew-T.

Никакого интерфейса здесь нет: ни tkinter, ни telebot. Этот модуль
импортируют И десктопное приложение, И телеграм-бот, чтобы расчётная
логика существовала в ОДНОМ экземпляре. В этом проекте почти все баги
рождались из скопированного кода, который потом чинили в одном месте
и забывали в другом — здесь такого быть не должно.

Запускать нужно в окружении с установленной SHARPpy (sharppy_env).
"""

import os
import ssl
import re
import math
import threading
import traceback
from datetime import datetime, timezone, timedelta

import numpy as np
import requests

import matplotlib
matplotlib.use("Agg")  # без GUI: годится и для бота, и для десктопа
import matplotlib.pyplot as plt


from sharppy.sharptab import profile, params, winds, interp
try:
    from sharppy.sharptab import thermo
except ImportError:
    thermo = None

# geopy НЕ трогаем на импорте. Причина: на Python 3.7 под Windows создание
# Nominatim вызывает ssl.create_default_context(), который читает системное
# хранилище сертификатов и падает с "SSLError: not enough data". Раньше это
# роняло весь модуль ещё до запуска — вместе с /fact, которому geopy вообще
# не нужен. Теперь геокодер создаётся лениво, только когда реально нужен.
_geolocator = None
_geo_error = None


def _get_geolocator():
    """Создаёт геокодер при первом обращении, обходя проблему с
    сертификатами Windows."""
    global _geolocator, _geo_error
    if _geolocator is not None:
        return _geolocator
    if _geo_error is not None:
        raise RuntimeError(_geo_error)
    try:
        from geopy.geocoders import Nominatim
    except ImportError:
        _geo_error = ("Не установлен geopy — поиск по названию города недоступен.\n"
                      "Установите:  pip install geopy\n"
                      "Команда /fact работает и без него.")
        raise RuntimeError(_geo_error)

    # Собственный SSL-контекст вместо системного хранилища Windows.
    ctx = None
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        try:
            ctx = ssl._create_unverified_context()
        except Exception:
            ctx = None

    try:
        try:
            _geolocator = Nominatim(user_agent="sharppy_core", timeout=10,
                                    ssl_context=ctx)
        except TypeError:
            # у совсем старых версий geopy нет параметра ssl_context
            _geolocator = Nominatim(user_agent="sharppy_core", timeout=10)
    except Exception as e:
        _geo_error = (
            f"Геокодер не запустился ({type(e).__name__}: {e}).\n"
            "Это известная беда Python 3.7 под Windows с хранилищем "
            "сертификатов.\n\n"
            "Обход: указывайте координаты вместо названия —\n"
            "    /frcst 62.03,129.73 gfs 6\n"
            "Так geopy не нужен вообще.\n"
            "Либо поставьте certifi:  pip install certifi"
        )
        raise RuntimeError(_geo_error)
    return _geolocator

MISSING = -9999.0
KTS_TO_MS = 0.514444
MS_TO_KTS = 1.94384

# Модели Open-Meteo, у которых есть УРОВНИ ДАВЛЕНИЯ (без них зондирование
# не построить). Третий элемент — зона покрытия: None означает глобальную
# модель, строка — регион, за пределами которого модель вернёт пустоту.
#
# ВАЖНО ПРО ВЫСОКОЕ РАЗРЕШЕНИЕ: все модели 1-5 км по определению
# региональные — высокое разрешение достигается тем, что считается
# небольшой кусок планеты. Для Сибири и Дальнего Востока таких моделей
# не существует; лучшее доступное там — глобальные 10-11 км (UKMO, ICON).
MODELS = {
    # ---- Глобальные: работают в любой точке, включая Якутию ----
    'ukmo':   ('UKMO Global 10км', 'ukmo_seamless', None),
    'icon':   ('ICON Global 11км (DWD)', 'icon_global', None),
    'gfs':    ('GFS (NOAA)', 'gfs_global', None),
    'ecmwf':  ('ECMWF IFS 0.25°', 'ecmwf_ifs025', None),
    'gem':    ('GEM Global (Канада)', 'gem_global', None),
    'arpege': ('ARPEGE Global (Франция)', 'meteofrance_arpege_world', None),
    'jma':    ('JMA GSM (Япония)', 'jma_gsm', None),
    'bom':    ('BOM ACCESS-G (Австралия)', 'bom_access_global', None),

    # ---- Региональные высокого разрешения: ТОЛЬКО своя зона ----
    'icond2':  ('ICON-D2 2км', 'icon_d2', 'Германия и Центр. Европа'),
    'iconeu':  ('ICON-EU 7км', 'icon_eu', 'Европа'),
    # AROME France HD в список НЕ входит: по документации Open-Meteo у неё
    # только сокращённый набор приземных полей и НЕТ уровней давления,
    # то есть зондирование по ней не построить в принципе. То же касается
    # 15-минутных моделей.
    'aromefr': ('AROME France 2.5км', 'meteofrance_arome_france', 'Франция'),
    'cosmo':   ('COSMO 5км (ARPAE)', 'arpae_cosmo_5m', 'Италия'),
    'hrrr':    ('HRRR 3км', 'gfs_hrrr', 'США'),
    'knmi':    ('KNMI Harmonie 2.5км', 'knmi_seamless', 'Нидерланды и Сев. море'),
    'ukv':     ('UKMO UKV 2км', 'ukmo_uk_deterministic_2km', 'Британия'),
}

LEVELS = [1000, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500,
          450, 400, 350, 300, 275, 250, 225, 200, 175, 150, 125, 100]


# =====================================================================
#  Вспомогательное
# =====================================================================
def safe_val(val, default=0.0):
    try:
        if val is None or np.ma.is_masked(val):
            return default
        f = float(val)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _es(t_c):
    """Насыщающее давление пара, гПа (Магнус)."""
    return 6.112 * math.exp(17.67 * t_c / (t_c + 243.5))


def dewpoint_from_rh(t_c, rh):
    """Точка росы по Магнусу (не по правилу T-(100-RH)/5, которое врёт
    до 9 °C в сухом воздухе)."""
    if t_c is None or rh is None:
        return MISSING
    rh = min(max(float(rh), 0.1), 100.0)
    a, b = 17.625, 243.04
    g = math.log(rh / 100.0) + (a * t_c) / (b + t_c)
    return min((b * g) / (a - g), t_c)


def rh_from_t_td(t_c, td_c):
    """Относительная влажность из T и Td — честно, через давление пара.
    В боте здесь стояло rh = 100 - 5*(T-Td), то же кривое правило наоборот."""
    try:
        return max(0.0, min(100.0, 100.0 * _es(td_c) / _es(t_c)))
    except (ValueError, ZeroDivisionError, OverflowError):
        return 0.0


# =====================================================================
#  Тексты вердиктов на двух языках
#  Хранятся отдельно от логики: расчёт выдаёт КОДЫ, а рендер подставляет
#  нужный язык. Так перевод не зависит от разбора русских строк.
# =====================================================================
VERDICT_TEXT = {
    'PDS_TOR':     {'ru': "🌪  PDS TOR: Опасность торнадо!",
                    'en': "🌪  PDS TOR: Tornado danger!"},
    'TOR':         {'ru': "🌪  TOR: Риск торнадо",
                    'en': "🌪  TOR: Tornado risk"},
    'MRGL_TOR':    {'ru': "⚠️  MRGL TOR: Возможны торнадо",
                    'en': "⚠️  MRGL TOR: Tornadoes possible"},
    'SVR_HAIL':    {'ru': "☄️  SVR: Крупный град",
                    'en': "☄️  SVR: Large hail"},
    'SVR_WIND':    {'ru': "💨  SVR: Сильные шквалы",
                    'en': "💨  SVR: Severe wind gusts"},
    'GEN_TSTM':    {'ru': "⛈  GEN TSTM: Грозы",
                    'en': "⛈  GEN TSTM: Thunderstorms"},
    'FLASH_FLOOD': {'ru': "🌊  FLASH FLOOD: Ливни",
                    'en': "🌊  FLASH FLOOD: Torrential rain"},
    'BLIZZARD':    {'ru': "❄️  BLIZZARD: Метель",
                    'en': "❄️  BLIZZARD: Blizzard"},
    'NONE_CAP':    {'ru': "🚫  NONE: Инверсия (крышка держит)",
                    'en': "🚫  NONE: Capping inversion holds"},
    'NONE_CALM':   {'ru': "☀️  NONE: Спокойно",
                    'en': "☀️  NONE: Quiet"},
}

MODE_TEXT = {
    'NO_CONV': {'ru': "Конвекция маловероятна",        'en': "Convection unlikely"},
    'SINGLE':  {'ru': "Одноячейковые (Single Cell)",   'en': "Single Cell"},
    'MULTI':   {'ru': "Многоячейковые (Multi-cell)",   'en': "Multi-cell / Lines"},
    'SUPER':   {'ru': "Суперячейковые (Supercell)",    'en': "Supercell"},
    'MCS':     {'ru': "Организованные МКС (MCS)",      'en': "Organized MCS"},
}

PRECIP_TEXT = {
    'NONE': {'ru': "Нет",               'en': "None"},
    'RAIN': {'ru': "Дождь",             'en': "Rain"},
    'SNOW': {'ru': "Снег",              'en': "Snow"},
    'TSTM': {'ru': "Ливневые / Гроза",  'en': "Showers / Thunderstorm"},
}


# =====================================================================
#  Расчёт вердиктов (без простыни индексов)
# =====================================================================
def compute_verdicts(prof):
    """
    Возвращает словарь: режим конвекции, осадки, список вердиктов
    и небольшой набор величин, нужных для подписи диаграммы.
    """
    mupcl = params.parcelx(prof, flag=3)
    cape = safe_val(mupcl.bplus)
    cin = safe_val(mupcl.bminus)
    lcl_h = safe_val(mupcl.lclhght)
    el_h = safe_val(mupcl.elhght)

    # PWAT: precip_water() отдаёт ДЮЙМЫ, переводим в мм
    pw = safe_val(params.precip_water(prof)) * 25.4

    t_sfc = safe_val(prof.tmpc[prof.sfc])
    td_sfc = safe_val(prof.dwpc[prof.sfc])

    # SRH — только storm-relative, относительно вектора Бункерса.
    try:
        rstu, rstv, _, _ = winds.non_parcel_bunkers_motion(prof)
    except Exception:
        rstu = rstv = 0.0
    try:
        srh01 = safe_val(winds.helicity(prof, 0, 1000, stu=rstu, stv=rstv)[0])
    except Exception:
        srh01 = 0.0
    try:
        srh03 = safe_val(winds.helicity(prof, 0, 3000, stu=rstu, stv=rstv)[0])
    except Exception:
        srh03 = 0.0

    # Сдвиг: внутри SHARPpy ветер в УЗЛАХ, поэтому переводим явно.
    def _shear_kts(z_top):
        try:
            p_top = interp.pres(prof, interp.to_msl(prof, float(z_top)))
            su, sv = winds.wind_shear(prof, pbot=prof.pres[prof.sfc], ptop=p_top)
            return float(np.hypot(su, sv))
        except Exception:
            return 0.0

    shear06_kts = _shear_kts(6000)
    shear06_ms = shear06_kts * KTS_TO_MS
    shear03_ms = _shear_kts(3000) * KTS_TO_MS

    try:
        stp = safe_val(params.stp_fixed(cape, lcl_h, srh01, shear06_kts))
    except Exception:
        stp = 0.0
    try:
        scp = safe_val(params.scp(cape, srh03, shear06_kts))
    except Exception:
        scp = 0.0
    try:
        ship = safe_val(params.ship(prof))
    except Exception:
        ship = 0.0
    try:
        dcape = safe_val(params.dcape(prof)[0])
    except Exception:
        dcape = 0.0

    # ---- режим конвекции ----
    mode_code = 'NO_CONV'
    if cape > 100:
        if shear06_ms < 12:
            mode_code = 'SINGLE'
        elif shear06_ms < 20:
            mode_code = 'MULTI'
        else:
            mode_code = 'SUPER' if scp >= 2 else 'MCS'
    mode = MODE_TEXT[mode_code]['ru']

    # ---- осадки ----
    rh_sfc = rh_from_t_td(t_sfc, td_sfc)
    precip_code = 'NONE'
    if rh_sfc > 80 or pw > 30:
        precip_code = 'SNOW' if t_sfc <= 0 else 'RAIN'
        if cape > 500:
            precip_code = 'TSTM'
    precip = PRECIP_TEXT[precip_code]['ru']

    # ---- вердикты ----
    # Собираем КОДЫ, а не готовые строки: так один и тот же вердикт
    # можно вывести и по-русски, и по-английски без разбора текста.
    codes = []
    if stp >= 1 and cin > -50:
        codes.append('PDS_TOR' if (stp >= 3 and shear06_ms > 23 and lcl_h < 1000) else 'TOR')
    elif stp >= 0.5 and cin > -125:
        codes.append('MRGL_TOR')

    if scp >= 2 or ship >= 1 or cape > 1000:
        if ship >= 1.5:
            codes.append('SVR_HAIL')
        if dcape > 800 or shear06_ms > 18:
            codes.append('SVR_WIND')
        if not codes and cape > 0:
            codes.append('GEN_TSTM')

    if pw > 40:
        codes.append('FLASH_FLOOD')
    if t_sfc <= 0 and shear06_ms > 15:
        codes.append('BLIZZARD')

    if not codes:
        codes.append('NONE_CAP' if (cin < -150 and cape > 500) else 'NONE_CALM')

    verdicts = [VERDICT_TEXT[c]['ru'] for c in codes]

    # ---- величины только для подписи под диаграммой ----
    li_mu = safe_val(mupcl.li5)

    # 3CAPE: просим parcelx проинтегрировать частицу в слое до 3 км.
    # (Ручной цикл по mupcl.tmpc, который был в старом боте, не работал:
    # такого атрибута у Parcel нет, траектория лежит в ptrace/ttrace.)
    try:
        p3 = interp.pres(prof, interp.to_msl(prof, 3000.))
        cape_3km = max(0.0, safe_val(params.parcelx(
            prof, flag=1, pbot=prof.pres[prof.sfc], ptop=p3).bplus))
    except Exception:
        cape_3km = 0.0

    try:
        t850, td850 = interp.temp(prof, 850), interp.dwpt(prof, 850)
        t700, td700 = interp.temp(prof, 700), interp.dwpt(prof, 700)
        t500 = interp.temp(prof, 500)
        k_index = safe_val((t850 - t500) + td850 - (t700 - td700))
        tt_index = safe_val((t850 - t500) + (td850 - t500))
    except Exception:
        k_index = tt_index = 0.0

    # Зимние явления считаются всегда, но в вывод попадают, только если
    # что-то нашлось: летом модуль сразу возвращает пусто по температуре.
    winter = None
    try:
        from winter_hazards import winter_verdicts
        winter = winter_verdicts(prof, {'mucape': cape, 'cape': cape})
    except Exception:
        pass

    return {
        'mode': mode, 'precip': precip, 'verdicts': verdicts,
        'winter': winter,
        'parcel': mupcl,
        'cape': cape, 'cin': cin, 'lcl': lcl_h, 'el': el_h,
        'shear06_ms': shear06_ms, 'shear03_ms': shear03_ms,
        'srh01': srh01, 'srh03': srh03, 'pw': pw,
        'stp': stp, 'scp': scp, 'ship': ship, 'dcape': dcape,
        'li': li_mu, 'cape3': cape_3km, 'k': k_index, 'tt': tt_index,
        'lfc': safe_val(_try(lambda: mupcl.lfchght)),
        'brnshear': safe_val(_try(lambda: mupcl.brnshear), 0.0),
        'codes': codes, 'mode_code': mode_code, 'precip_code': precip_code,
    }


# =====================================================================
#  Skew-T диаграмма
# =====================================================================
SKEW = 45.0          # насколько «завалены» изотермы
P_BOT, P_TOP = 1050.0, 100.0


def _sx(t, p):
    """Смещение по X для skew-T: чем выше (меньше p), тем правее."""
    return np.asarray(t, dtype=float) + SKEW * np.log10(P_BOT / np.asarray(p, dtype=float))


def _sy(p):
    return -np.log10(np.asarray(p, dtype=float))


def make_profile(**kw):
    """
    Строит профиль, по возможности «convective» — именно он считает SARS
    (Sounding Analog Retrieval System) и кладёт результат в prof.matches
    (град) и prof.supercell_matches (суперячейки/торнадо).

    Если convective не получился (а он тяжелее и капризнее к вырожденным
    профилям) — молча откатываемся на обычный. Тогда блок SARS покажет
    «н/д», но всё остальное посчитается как раньше.
    """
    try:
        kw_conv = dict(kw)
        kw_conv['profile'] = 'convective'
        prof = profile.create_profile(**kw_conv)
        if prof is not None:
            return prof
    except Exception:
        pass
    kw_def = dict(kw)
    kw_def['profile'] = 'default'
    return profile.create_profile(**kw_def)


def _try(fn, default=None):
    """Вызов, который не должен ронять всю картинку.
    На вырожденных профилях отдельные функции SHARPpy кидают исключения —
    это нормально, в таблице просто появится «н/д»."""
    try:
        v = fn()
        if v is None:
            return default
        if isinstance(v, (int, float, np.floating)):
            f = float(v)
            return f if np.isfinite(f) else default
        return v
    except Exception:
        return default


def _f(val, digits=0, suffix=""):
    """Число для таблицы, либо «н/д»."""
    if val is None:
        return "н/д"
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "н/д"
    if not np.isfinite(v):
        return "н/д"
    return f"{v:.{digits}f}{suffix}"


def _parcel_rows(prof):
    """Таблица частиц: как блок PCL/CAPE/CINH/LCL/LI/LFC/EL в SHARPpy."""
    rows = []
    for label, flag in (("SFC", 1), ("ML", 4), ("FCST", 2), ("MU", 3)):
        pcl = _try(lambda f=flag: params.parcelx(prof, flag=f))
        if pcl is None:
            rows.append((label, "н/д", "н/д", "н/д", "н/д", "н/д", "н/д"))
            continue
        rows.append((
            label,
            _f(_try(lambda: pcl.bplus)),
            _f(_try(lambda: pcl.bminus)),
            _f(_try(lambda: pcl.lclhght)),
            _f(_try(lambda: pcl.li5), 1),
            _f(_try(lambda: pcl.lfchght)),
            _f(_try(lambda: pcl.elhght)),
        ))
    return rows


def _kinematics_rows(prof, rstu, rstv):
    """Кинематика по слоям: SRH / сдвиг / средний ветер / SR-ветер."""
    sfc_p = prof.pres[prof.sfc]

    def p_at(z_agl):
        return _try(lambda: interp.pres(prof, interp.to_msl(prof, float(z_agl))))

    eff = _try(lambda: params.effective_inflow_layer(prof), (None, None))
    try:
        eff_bot, eff_top = eff[0], eff[1]
        if np.ma.is_masked(eff_bot) or np.ma.is_masked(eff_top):
            eff_bot = eff_top = None
    except Exception:
        eff_bot = eff_top = None

    layers = [
        ("SFC-1км", sfc_p, p_at(1000), 0.0, 1000.0),
        ("SFC-3км", sfc_p, p_at(3000), 0.0, 3000.0),
        ("Эфф.слой", eff_bot, eff_top, None, None),
        ("SFC-6км", sfc_p, p_at(6000), 0.0, 6000.0),
        ("SFC-8км", sfc_p, p_at(8000), 0.0, 8000.0),
    ]

    rows = []
    for name, pb, pt, zb, zt in layers:
        if pb is None or pt is None:
            rows.append((name, "н/д", "н/д", "н/д", "н/д"))
            continue

        # SRH: по высотам AGL, если слой задан высотами; иначе по давлению
        if zb is not None:
            srh = _try(lambda: winds.helicity(prof, zb, zt, stu=rstu, stv=rstv)[0])
        else:
            zb2 = _try(lambda: interp.to_agl(prof, interp.hght(prof, pb)))
            zt2 = _try(lambda: interp.to_agl(prof, interp.hght(prof, pt)))
            srh = (_try(lambda: winds.helicity(prof, zb2, zt2, stu=rstu, stv=rstv)[0])
                   if (zb2 is not None and zt2 is not None) else None)

        shr = _try(lambda: float(np.hypot(*winds.wind_shear(prof, pbot=pb, ptop=pt))))
        mnw = _try(lambda: winds.mean_wind(prof, pbot=pb, ptop=pt))
        srw = _try(lambda: winds.sr_wind(prof, pbot=pb, ptop=pt, stu=rstu, stv=rstv))

        def _spd(vec):
            try:
                return float(np.hypot(vec[0], vec[1]))
            except Exception:
                return None

        rows.append((
            name,
            _f(srh),
            _f(shr * KTS_TO_MS if shr is not None else None, 1),
            _f(_spd(mnw) * KTS_TO_MS if _spd(mnw) is not None else None, 1),
            _f(_spd(srw) * KTS_TO_MS if _spd(srw) is not None else None, 1),
        ))
    return rows


def _thermo_rows(prof):
    """Термодинамика и градиенты — левый блок SHARPpy."""
    return [
        ("PW", _f(_try(lambda: params.precip_water(prof) * 25.4), 0, " мм")),
        ("DCAPE", _f(_try(lambda: params.dcape(prof)[0]), 0, " Дж/кг")),
        ("3CAPE 0-3км", _f(_try(lambda: params.parcelx(
            prof, flag=1, pbot=prof.pres[prof.sfc],
            ptop=interp.pres(prof, interp.to_msl(prof, 3000.))).bplus), 0, " Дж/кг")),
        ("K-index", _f(_try(lambda: (interp.temp(prof, 850) - interp.temp(prof, 500))
                            + interp.dwpt(prof, 850)
                            - (interp.temp(prof, 700) - interp.dwpt(prof, 700))), 1)),
        ("Totals", _f(_try(lambda: (interp.temp(prof, 850) - interp.temp(prof, 500))
                           + (interp.dwpt(prof, 850) - interp.temp(prof, 500))), 1)),
        ("ConvT", _f(_try(lambda: params.convective_temp(prof)), 1, " °C")),
        ("MaxT", _f(_try(lambda: params.max_temp(prof)), 1, " °C")),
        ("ГрадSFC-3км", _f(_try(lambda: params.lapse_rate(prof, 0, 3000, pres=False)), 1, " °C/км")),
        ("Град700-500", _f(_try(lambda: params.lapse_rate(prof, 700, 500, pres=True)), 1, " °C/км")),
        ("Град850-500", _f(_try(lambda: params.lapse_rate(prof, 850, 500, pres=True)), 1, " °C/км")),
    ]


def _sars_rows(prof):
    """
    SARS — поиск аналогов текущего зондирования среди архива реальных
    случаев сильной погоды (база SPC, Jewell et al.). Считается внутри
    ConvectiveProfile; если профиль обычный, атрибутов просто нет.

    Формат результата SARS: последний элемент — вероятность значимого
    события, предпоследние — счётчики совпадений, первые — даты и величины.
    """
    def _pack(matches, label):
        """Одной строкой: сколько точных аналогов и какая вероятность."""
        if matches is None:
            return (label, "н/д")
        try:
            n_quality = len(matches[0])
        except Exception:
            n_quality = "?"
        try:
            prob = f"{float(matches[-1]) * 100:.0f}%"
        except Exception:
            prob = "?"
        return (label, f"{n_quality} совп. / {prob}")

    return [
        _pack(_try(lambda: prof.matches), "Град (знач.)"),
        _pack(_try(lambda: prof.supercell_matches), "Торнадо (знач.)"),
    ]


def _local_sars(res):
    """
    Обращается к вашему архиву климатологии через sars_local.
    Возвращает (ранги, аналоги) или None, если модуля/базы нет —
    тогда панель просто не рисуется.
    """
    try:
        import sqlite3
        import sars_local
        path = os.path.join(os.path.expanduser("~"), "ERA5_Climatology",
                            "climatology.db")
        if not os.path.exists(path):
            return None
        conn = sqlite3.connect(path)
        out = sars_local.find_analogs(conn, sars_local.params_from_res(res), n=6)
        conn.close()
        return out
    except Exception:
        return None


def _draw_sars_panel(fig, prof, res, x, y, width=0.33):
    """
    Панель SARS: слева — насколько сегодняшние значения редки для Якутии,
    справа — похожие дни из архива и чем они кончились.

    Американский SARS из SHARPpy показываем одной строкой: для здешних
    зондирований он почти всегда «н/д», потому что его база собрана по
    проксимальным зондированиям severe-событий в США, а таких значений
    тут не бывает.
    """
    fig.text(x, y, "SARS — аналоги по архиву Якутии", fontsize=11.5,
             fontweight='bold', color='#333333')
    dy = 0.021
    yy = y - 0.020

    data = _local_sars(res)
    if not data or not data[0]:
        fig.text(x, yy, "База климатологии не найдена", fontsize=10,
                 color='#888888')
        fig.text(x, yy - dy, "(~/ERA5_Climatology/climatology.db)",
                 fontsize=9, color='#aaaaaa')
        return

    cur_rank, analogs = data
    names = {'mlcape': 'MLCAPE', 'mucape': 'MUCAPE', 'shear6': 'Сдвиг 0-6км',
             'shear3': 'Сдвиг 0-3км', 'srh1': 'SRH 0-1км', 'srh3': 'SRH 0-3км',
             'pwat': 'PW', 'dcape': 'DCAPE', 'lcl_ml': 'LCL', 'lr75': 'Град700-500'}

    top = sorted(cur_rank.items(), key=lambda kv: -kv[1]['rank'])[:3]
    fig.text(x, yy, "Насколько это редко здесь:", fontsize=9.5,
             color='#777777')
    yy -= dy
    for key, d in top:
        pct = 100.0 * d['rank']
        col = '#b30000' if pct >= 99 else ('#cc6600' if pct >= 95 else '#333333')
        fig.text(x, yy, names.get(key, key), fontsize=10, color='#555555',
                 va='center')
        fig.text(x + 0.115, yy, f"выше {pct:.0f}% сроков", fontsize=10,
                 va='center', family='monospace', fontweight='bold', color=col)
        yy -= dy

    yy -= 0.006
    fig.text(x, yy, "Похожие дни:", fontsize=9.5, color='#777777')
    yy -= dy
    if analogs:
        for a in analogs[:2]:
            ev = ""
            if a['outcomes']:
                ev = " — " + ", ".join(
                    sars_events().get(e, e) for e, _ in a['outcomes'])
            txt = f"{a['dt'][:10]}  {a['city']}  {a['similarity']:.0f}%{ev}"
            fig.text(x, yy, txt, fontsize=9.5, va='center',
                     family='monospace',
                     color='#1a6b1a' if ev else '#333333')
            yy -= dy
    if not any(a['outcomes'] for a in analogs):
        fig.text(x, yy, "исходы не размечены — sars_local.py --label",
                 fontsize=9, color='#999999', va='center')
        yy -= dy

    # американский SARS одной строкой, для сравнения
    def _n(m):
        try:
            return str(len(m[0]))
        except Exception:
            return "н/д"
    fig.text(x, yy - 0.004, f"Архив SPC (США): град {_n(_try(lambda: prof.matches))}, "
                            f"суперячейки {_n(_try(lambda: prof.supercell_matches))}",
             fontsize=9, color='#999999', va='center')


def sars_events():
    try:
        import sars_local
        return sars_local.EVENTS
    except Exception:
        return {}


def _composite_rows(prof, res):
    """Композитные индексы — правый нижний блок."""
    mupcl = res.get('parcel')
    mlpcl = _try(lambda: params.parcelx(prof, flag=4))
    return [
        ("STP (cin)", _f(_try(lambda: res.get('stp')), 1)),
        ("SCP", _f(_try(lambda: res.get('scp')), 1)),
        ("SHIP", _f(_try(lambda: res.get('ship')), 1)),
        ("SigSevere", _f(_try(lambda: params.sig_severe(prof, mlpcl=mlpcl)), 0)),
        ("DCP", _f(_try(lambda: params.dcp(prof)), 1)),
        ("MMP", _f(_try(lambda: params.mmp(prof)), 2)),
        ("WNDG", _f(_try(lambda: params.wndg(prof, mlpcl=mlpcl)), 1)),
        ("SHERB", _f(_try(lambda: params.sherb(prof)), 1)),
        ("MBURST", _f(_try(lambda: params.mburst(prof)), 1)),
    ]


def _draw_hodograph(ax, prof, rstu, rstv):
    """
    Годограф: кривая ветра по высоте, раскрашенная по слоям, как в SHARPpy
    (0-3 км красный, 3-6 зелёный, 6-9 жёлтый, 9-12 голубой), плюс векторы
    движения правого и левого суперячейкового шторма по Бункерсу.
    """
    u = np.asarray(prof.u, dtype=float)
    v = np.asarray(prof.v, dtype=float)
    h = np.asarray(prof.hght, dtype=float)
    ok = np.isfinite(u) & np.isfinite(v) & np.isfinite(h) & (np.abs(u) < 400) & (np.abs(v) < 400)
    u, v, h = u[ok], v[ok], h[ok]
    if len(u) < 2:
        ax.text(0.5, 0.5, "нет данных о ветре", ha='center', va='center',
                transform=ax.transAxes, fontsize=9, color='#888')
        return
    agl = h - h[0]

    lim = max(30.0, float(np.nanmax(np.hypot(u, v))) * 1.15)
    for ring in range(10, int(lim) + 20, 10):
        ax.add_artist(plt.Circle((0, 0), ring, fill=False,
                                 color='#cccccc', lw=0.6, zorder=1))
        ax.text(ring * 0.71, -ring * 0.71, str(ring), fontsize=8,
                color='#999999', ha='center', va='center', zorder=2)
    ax.axhline(0, color='#bbbbbb', lw=0.6, zorder=1)
    ax.axvline(0, color='#bbbbbb', lw=0.6, zorder=1)

    segs = [(0, 3000, '#d62728', '0-3 км'), (3000, 6000, '#2ca02c', '3-6 км'),
            (6000, 9000, '#e6b800', '6-9 км'), (9000, 12000, '#17becf', '9-12 км')]
    for lo, hi, col, lbl in segs:
        m = (agl >= lo) & (agl <= hi)
        if m.sum() > 1:
            ax.plot(u[m], v[m], color=col, lw=2.2, zorder=4, label=lbl)
        # стык сегментов, чтобы линия не рвалась
        idx = np.where(agl <= hi)[0]
        nxt = np.where(agl > hi)[0]
        if len(idx) and len(nxt):
            a, b = idx[-1], nxt[0]
            ax.plot([u[a], u[b]], [v[a], v[b]], color=col, lw=2.2, zorder=4)

    # метки высот
    for z in (1000, 3000, 6000, 9000):
        if agl.max() >= z:
            i = int(np.argmin(np.abs(agl - z)))
            ax.plot(u[i], v[i], 'o', ms=4, color='black', zorder=6)
            ax.text(u[i] + 1.5, v[i] + 1.5, f"{z // 1000}", fontsize=9.5,
                    zorder=6, fontweight='bold')

    if rstu is not None and np.isfinite(rstu):
        ax.plot(rstu, rstv, 'o', ms=8, mfc='none', mec='#b30000', mew=1.8,
                zorder=7, label='Бункерс R')
        ax.text(rstu + 2, rstv - 3, "R", color='#b30000', fontsize=10,
                fontweight='bold', zorder=7)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color('#aaaaaa')
    ax.set_title("Годограф (узлы)", fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.85, handlelength=1.4)


def _panel(fig, x, y, title, rows, col_w=(0.128, 0.075), fontsize=10.5):
    """Простая двухколоночная таблица «подпись — значение»."""
    fig.text(x, y, title, fontsize=11.5, fontweight='bold', color='#333333')
    dy = 0.0265
    for i, (label, value) in enumerate(rows):
        yy = y - 0.019 - i * dy
        fig.text(x, yy, label, fontsize=fontsize, color='#555555', va='center')
        fig.text(x + col_w[0], yy, value, fontsize=fontsize, va='center',
                 family='monospace', fontweight='bold')


def _table(fig, x, y, title, header, rows, widths, fontsize=10.5):
    """Многоколоночная таблица с шапкой (частицы, кинематика)."""
    fig.text(x, y, title, fontsize=11.5, fontweight='bold', color='#333333')
    dy = 0.0265
    for ci, htxt in enumerate(header):
        fig.text(x + widths[ci], y - 0.019, htxt, fontsize=9.5,
                 color='#777777', va='center', family='monospace')
    for ri, row in enumerate(rows):
        yy = y - 0.019 - (ri + 1) * dy
        for ci, cell in enumerate(row):
            fig.text(x + widths[ci], yy, str(cell), fontsize=fontsize, va='center',
                     family='monospace',
                     fontweight='bold' if ci == 0 else 'normal',
                     color='#333333' if ci == 0 else '#111111')


def draw_skewt(prof, res, title):
    """
    Полная панель: Skew-T слева, годограф справа, таблицы индексов снизу —
    по духу как окно SHARPpy, но своей отрисовкой на matplotlib.
    """
    fig = plt.figure(figsize=(17.0, 11.5))
    ax = fig.add_axes([0.045, 0.35, 0.40, 0.60])
    axw = fig.add_axes([0.455, 0.35, 0.035, 0.60], sharey=ax)
    ax_h = fig.add_axes([0.645, 0.575, 0.29, 0.365])

    p_line = np.arange(P_BOT, P_TOP - 1, -5.0)

    for t0 in range(-110, 61, 10):
        ax.plot(_sx(np.full_like(p_line, t0), p_line), _sy(p_line),
                color='#9ecae1', lw=0.6, zorder=1)
    for th in range(-40, 201, 10):
        t_dry = (th + 273.15) * (p_line / 1000.0) ** 0.2854 - 273.15
        ax.plot(_sx(t_dry, p_line), _sy(p_line),
                color='#fdae6b', lw=0.5, alpha=0.8, zorder=1)
    if thermo is not None and hasattr(thermo, 'wetlift'):
        p_m = np.arange(1000.0, P_TOP - 1, -25.0)
        for t0 in range(-20, 41, 5):
            try:
                tm = [thermo.wetlift(1000.0, float(t0), float(pp)) for pp in p_m]
                ax.plot(_sx(tm, p_m), _sy(p_m), color='#74c476', lw=0.5,
                        ls='--', alpha=0.8, zorder=1)
            except Exception:
                break
    if thermo is not None and hasattr(thermo, 'temp_at_mixrat'):
        p_w = np.arange(1000.0, 599.0, -25.0)
        for w in (0.4, 1, 2, 4, 7, 10, 16, 24, 32):
            try:
                tw = [thermo.temp_at_mixrat(float(w), float(pp)) for pp in p_w]
                ax.plot(_sx(tw, p_w), _sy(p_w), color='#c994c7', lw=0.5,
                        ls=':', alpha=0.9, zorder=1)
            except Exception:
                break

    p = np.asarray(prof.pres, dtype=float)
    t = np.asarray(prof.tmpc, dtype=float)
    td = np.asarray(prof.dwpc, dtype=float)
    m1 = np.isfinite(p) & np.isfinite(t) & (p >= P_TOP)
    ax.plot(_sx(t[m1], p[m1]), _sy(p[m1]), color='#d62728', lw=2.4, zorder=5, label='T')
    m2 = np.isfinite(p) & np.isfinite(td) & (td > -200) & (p >= P_TOP)
    ax.plot(_sx(td[m2], p[m2]), _sy(p[m2]), color='#2ca02c', lw=2.4, zorder=5, label='Td')

    pcl = res.get('parcel')
    try:
        pt = np.asarray(pcl.ptrace, dtype=float)
        tt = np.asarray(pcl.ttrace, dtype=float)
        mm = np.isfinite(pt) & np.isfinite(tt) & (pt >= P_TOP)
        if mm.sum() > 1:
            ax.plot(_sx(tt[mm], pt[mm]), _sy(pt[mm]), color='black', lw=1.6,
                    ls='--', zorder=6, label='Частица (MU)')
    except Exception:
        pass

    # LFC берём из объекта частицы: в res его отдельно нет
    lfc_h = None
    try:
        lfc_h = float(pcl.lfchght)
        if not np.isfinite(lfc_h):
            lfc_h = None
    except Exception:
        lfc_h = None

    for hval, lbl, col in ((res.get('lcl'), 'LCL', '#1f77b4'),
                           (lfc_h, 'LFC', '#e07b00'),
                           (res.get('el'), 'EL', '#7f7f7f')):
        try:
            if hval and hval > 0:
                pl = interp.pres(prof, interp.to_msl(prof, float(hval)))
                if np.isfinite(pl) and P_TOP <= pl <= P_BOT:
                    ax.axhline(_sy(pl), color=col, lw=1.2, ls='-.', alpha=0.75, zorder=4)
                    ax.text(0.015, _sy(pl), lbl, transform=ax.get_yaxis_transform(),
                            color=col, fontsize=10, va='bottom', fontweight='bold')
        except Exception:
            pass

    ax.set_ylim(_sy(P_BOT), _sy(P_TOP))
    ax.set_xlim(-40, 45)
    pticks = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100]
    ax.set_yticks([_sy(pp) for pp in pticks])
    ax.set_yticklabels([str(pp) for pp in pticks], fontsize=10)
    ax.tick_params(axis='x', labelsize=10)
    ax.set_ylabel("Давление, гПа", fontsize=11)
    ax.set_xlabel("Температура, °C (у поверхности)", fontsize=11)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.set_title(title, fontsize=13, fontweight='bold')

    # ветер сбоку от Skew-T
    try:
        u = np.asarray(prof.u, dtype=float)
        v = np.asarray(prof.v, dtype=float)
        mw = (np.isfinite(u) & np.isfinite(v) & np.isfinite(p) & (p >= P_TOP)
              & (np.abs(u) < 400) & (np.abs(v) < 400))
        idx = np.where(mw)[0]
        if len(idx) > 28:
            idx = idx[:: max(1, len(idx) // 28)]
        axw.barbs(np.zeros(len(idx)), _sy(p[idx]), u[idx], v[idx],
                  length=5.5, linewidth=0.6)
    except Exception:
        pass
    axw.set_xlim(-1, 1)
    axw.set_xticks([])
    for sp in ('top', 'right', 'bottom'):
        axw.spines[sp].set_visible(False)
    plt.setp(axw.get_yticklabels(), visible=False)

    # ---- годограф ----
    rstu = rstv = None
    try:
        rstu, rstv, _, _ = winds.non_parcel_bunkers_motion(prof)
    except Exception:
        pass
    _draw_hodograph(ax_h, prof, rstu, rstv)

    # ---- таблицы ----
    # add_artist, а не fig.lines.append: последний объявлен устаревшим и
    # в разных версиях matplotlib ведёт себя по-разному. Здесь окружение
    # на Python 3.7, то есть matplotlib не выше 3.5 — а add_artist
    # одинаково работает начиная с 3.0.
    from matplotlib.lines import Line2D as _L2D
    fig.add_artist(_L2D([0.03, 0.97], [0.325, 0.325],
                        transform=fig.transFigure, color='#999999', lw=0.8))

    _table(fig, 0.030, 0.300, "ЧАСТИЦЫ",
           ["", "CAPE", "CINH", "LCL", "LI", "LFC", "EL"],
           _parcel_rows(prof),
           widths=[0.0, 0.040, 0.083, 0.126, 0.169, 0.203, 0.246])

    _table(fig, 0.325, 0.300, "КИНЕМАТИКА  (SRH м²/с², остальное м/с)",
           ["", "SRH", "Сдвиг", "СрВет", "SR-вет"],
           _kinematics_rows(prof, rstu if rstu is not None else 0.0,
                            rstv if rstv is not None else 0.0),
           widths=[0.0, 0.062, 0.107, 0.152, 0.197])

    # BRN Shear — величина кинематическая, поэтому живёт здесь, а не
    # в термодинамике (та упиралась в нижнюю строку вердиктов).
    _brn = _f(_try(lambda: params.parcelx(prof, flag=3).brnshear), 0, " м²/с²")
    fig.text(0.325, 0.118, "BRN Shear:", fontsize=10.5, color='#555555', va='center')
    fig.text(0.453, 0.118, _brn, fontsize=10.5, va='center',
             family='monospace', fontweight='bold')

    _draw_sars_panel(fig, prof, res, x=0.615, y=0.545)

    _panel(fig, 0.585, 0.300, "ТЕРМОДИНАМИКА", _thermo_rows(prof))
    _panel(fig, 0.815, 0.300, "КОМПОЗИТЫ", _composite_rows(prof, res),
           col_w=(0.098, 0.06))

    _emoji = re.compile(r'[^\w\s:/№()\-.,°²³|·]+', flags=re.UNICODE)
    tags = [_emoji.sub('', v.split(':')[0]).strip() for v in res.get('verdicts', [])]
    tags = [x for x in tags if x]
    fig.text(0.5, 0.014,
             f"{res.get('mode', '')}   |   осадки: {res.get('precip', '')}"
             + (f"   |   {' · '.join(tags)}" if tags else ""),
             fontsize=11.5, ha='center', va='bottom', color='#222222')

    return fig


# =====================================================================
#  Источники данных
# =====================================================================
def fetch_actual(stn, date_str, hour):
    """Фактическое зондирование с актуального эндпоинта Вайоминга (ARCC).
    Старый weather.uwyo.edu/cgi-bin/sounding больше не работает."""
    y, m, d = date_str[:4], date_str[4:6], date_str[6:8]
    url = (f"https://weather.arcc.uwyo.edu/wsgi/sounding?"
           f"datetime={y}-{m}-{d}%20{hour}:00:00&id={stn}&src=UNKNOWN&type=TEXT:LIST")
    r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=30)
    if "<PRE>" not in r.text:
        raise RuntimeError(f"Данные для станции {stn} на {d}.{m}.{y} {hour}Z не найдены.")

    content = r.text.split("<PRE>")[1].split("</PRE>")[0]
    lines = content.split("\n")

    header = next((ln for ln in lines
                   if 'PRES' in ln and 'HGHT' in ln and 'TEMP' in ln), None)
    if header is None:
        raise RuntimeError("В ответе нет заголовка таблицы.")

    # Границы колонок берём из заголовка, а не зашиваем числами:
    # раскладка у нового эндпоинта своя.
    ends = [mm.end() for mm in re.finditer(r'\S+', header)]

    def col(*names):
        for nm in names:
            pos = header.find(nm)
            if pos >= 0:
                e = pos + len(nm)
                prior = [x for x in ends if x < e]
                return (max(prior) if prior else 0), e
        return None

    cols = {'pres': col('PRES'), 'hght': col('HGHT'), 'temp': col('TEMP'),
            'dewp': col('DWPT'), 'wdir': col('DRCT'), 'wspd': col('SPED', 'SKNT')}
    if any(v is None for v in cols.values()):
        raise RuntimeError("В таблице нет нужных колонок.")

    # SPED — новый эндпоинт, скорость в м/с. SHARPpy ждёт УЗЛЫ.
    wind_ms = 'SPED' in header

    data = {k: [] for k in ('pres', 'hght', 'temp', 'dewp', 'wdir', 'wspd')}
    for ln in lines:
        if not ln.strip() or '---' in ln or 'PRES' in ln:
            continue

        def g(key):
            a, b = cols[key]
            seg = ln[a:b].strip()
            try:
                return float(seg) if seg else None
            except ValueError:
                return None

        pp, hh, tt, dd = g('pres'), g('hght'), g('temp'), g('dewp')
        if pp is None or hh is None or tt is None:
            continue
        wd, ws = g('wdir'), g('wspd')
        if wd is None or ws is None:
            wd, ws = MISSING, MISSING
        elif wind_ms:
            ws *= MS_TO_KTS
        data['pres'].append(pp); data['hght'].append(hh)
        data['temp'].append(tt); data['dewp'].append(dd if dd is not None else MISSING)
        data['wdir'].append(wd); data['wspd'].append(ws)

    if len(data['pres']) < 5:
        raise RuntimeError("Недостаточно корректных уровней в зондировании.")

    # Чистка: убывающее давление, строго растущая высота — иначе SHARPpy
    # забракует профиль (DataQualityException).
    order = np.argsort(-np.asarray(data['pres']))
    cleaned = {k: [] for k in data}
    last_p = last_h = None
    for i in order:
        pp, hh = data['pres'][i], data['hght'][i]
        if last_p is not None and abs(pp - last_p) < 0.01:
            continue
        if last_h is not None and hh <= last_h:
            continue
        for k in data:
            cleaned[k].append(data[k][i])
        last_p, last_h = pp, hh

    prof_obj = make_profile(
        pres=cleaned['pres'], hght=cleaned['hght'],
        tmpc=cleaned['temp'], dwpc=cleaned['dewp'],
        wdir=cleaned['wdir'], wspd=cleaned['wspd'], missing=MISSING)
    when = datetime.strptime(f"{date_str}{hour}", "%Y%m%d%H")
    return prof_obj, f"ФАКТ: станция {stn} — {d}.{m}.{y} {hour}Z", when


def fetch_forecast(place, model_key, step, base_hour=None):
    """Прогностический профиль через Open-Meteo."""
    # Если вместо названия переданы координаты «широта,долгота» —
    # геокодер не нужен совсем. Это заодно рабочий обход, когда geopy
    # не стартует из-за сертификатов Windows.
    coord = re.match(r'^\s*(-?\d+(?:[.,]\d+)?)\s*[,;]\s*(-?\d+(?:[.,]\d+)?)\s*$', place)
    if coord:
        lat = float(coord.group(1).replace(',', '.'))
        lon = float(coord.group(2).replace(',', '.'))
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise RuntimeError(f"Координаты вне диапазона: {lat}, {lon}")
        loc = type('L', (), {'latitude': lat, 'longitude': lon, 'altitude': None})()
        place = f"{lat:.3f}, {lon:.3f}"
    else:
        loc = _get_geolocator().geocode(place)
        if not loc:
            raise RuntimeError(f"Локация «{place}» не найдена.")
    if model_key not in MODELS:
        raise RuntimeError(
            f"Неизвестная модель «{model_key}».\n\nДоступны:\n" +
            "\n".join(f"  {k} — {v[0]}" + (f"  [только {v[2]}]" if v[2] else "")
                      for k, v in MODELS.items()))
    m_name, m_code, m_area = MODELS[model_key]

    v = ['temperature_2m', 'relative_humidity_2m', 'surface_pressure',
         'wind_speed_10m', 'wind_direction_10m']
    for l in LEVELS:
        v += [f'temperature_{l}hPa', f'relative_humidity_{l}hPa',
              f'geopotential_height_{l}hPa', f'wind_speed_{l}hPa',
              f'wind_direction_{l}hPa']

    resp = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": loc.latitude, "longitude": loc.longitude,
        "hourly": ",".join(v), "models": m_code,
        "windspeed_unit": "ms", "timezone": "UTC",
        # Явно задаём высоту, иначе Open-Meteo подставит свою из модели
        # рельефа, и приземные величины будут относиться к другой высоте.
        "elevation": round(loc.altitude) if getattr(loc, 'altitude', None) else 0,
    }, timeout=40)
    resp.raise_for_status()
    h = resp.json()['hourly']

    # При timezone=UTC массив начинается с 00:00 текущих суток. Если база
    # не указана, шаг отсчитывается от этой полуночи — прежнее поведение.
    # Если указана, целевое время считается явно и ищется в массиве: так
    # не надо держать в уме, сколько часов прошло от полуночи.
    step_req = int(step)          # исходный шаг: ниже step станет индексом
    if base_hour is None:
        step = max(0, min(step_req, len(h['time']) - 1))
        target_label = f" (+{step_req}ч от 00z)"
    else:
        bh = int(base_hour)
        if not 0 <= bh <= 23:
            raise RuntimeError("База — час от 00 до 23")
        base_day = h['time'][0][:10]
        target = (datetime.strptime(f"{base_day} {bh:02d}", "%Y-%m-%d %H")
                  + timedelta(hours=int(step)))
        want = target.strftime("%Y-%m-%dT%H:00")
        try:
            step = next(i for i, t in enumerate(h['time']) if t.startswith(want))
        except StopIteration:
            raise RuntimeError(
                f"Срок {target.strftime('%d.%m %H:%M')} UTC вне прогноза модели.\n"
                f"Доступно с {h['time'][0][:16].replace('T', ' ')} "
                f"по {h['time'][-1][:16].replace('T', ' ')} UTC.")
        target_label = f" [база {bh:02d}z +{step_req}ч]"

    prof_obj = _build_profile_from_openmeteo(h, step, m_name, m_area)
    when = datetime.fromisoformat(h['time'][step].replace('Z', '+00:00'))
    return prof_obj, f"ПРОГНОЗ {m_name}: {place}{target_label}", when


def _build_profile_from_openmeteo(h, step, m_name, m_area):
    """
    Собирает профиль SHARPpy из ответа Open-Meteo на конкретный срок.
    Общая для оперативного прогноза и архива — иначе правки пришлось бы
    вносить в двух местах, а расхождение между ними заметили бы не сразу.
    """
    def _no_data(what):
        if m_area:
            return RuntimeError(
                f"Модель {m_name}: нет данных для этой точки.\n"
                f"Причины бывают две — точка вне зоны «{m_area}», либо у модели "
                f"нет уровней давления (у некоторых высокоразрешённых моделей "
                f"публикуются только приземные поля).\n"
                f"Для Франции берите arpege или iconeu, для Сибири — "
                f"ukmo, icon, gfs, ecmwf.")
        return RuntimeError(f"Нет {what} на этот срок "
                            "(возможно, срок за пределами прогноза модели).")

    sp = h['surface_pressure'][step]
    if sp is None:
        raise _no_data("приземного давления")
    t0 = h['temperature_2m'][step]
    if t0 is None:
        raise _no_data("приземной температуры")

    P, H, T, D, WD, WS = [], [], [], [], [], []
    P.append(float(sp)); H.append(10.0); T.append(float(t0))
    D.append(dewpoint_from_rh(t0, h['relative_humidity_2m'][step]))
    WD.append(float(h['wind_direction_10m'][step] or 0.0))
    # windspeed_unit=ms, а create_profile ждёт УЗЛЫ
    WS.append(float(h['wind_speed_10m'][step] or 0.0) * MS_TO_KTS)

    for l in LEVELS:
        if l >= sp:
            continue
        tl = h.get(f'temperature_{l}hPa', [None])[step]
        hl = h.get(f'geopotential_height_{l}hPa', [None])[step]
        if tl is None or hl is None:
            continue
        if hl <= H[-1]:
            continue
        P.append(float(l)); H.append(float(hl)); T.append(float(tl))
        D.append(dewpoint_from_rh(tl, h.get(f'relative_humidity_{l}hPa', [None])[step]))
        WD.append(float(h.get(f'wind_direction_{l}hPa', [0])[step] or 0.0))
        WS.append(float(h.get(f'wind_speed_{l}hPa', [0])[step] or 0.0) * MS_TO_KTS)

    if len(P) < 5:
        raise RuntimeError("Слишком мало уровней в ответе API.")

    return make_profile(pres=P, hght=H, tmpc=T, dwpc=D,
                        wdir=WD, wspd=WS, missing=MISSING)


# =====================================================================
#  Приложение
# =====================================================================


def fetch_forecast_historical(place, model_key, date_str, hour, step=0):
    """
    Архив ПРОГНОЗОВ Open-Meteo: как модель видела эту дату тогда.

    Это не реанализ: historical-forecast-api хранит то, что модели
    выдавали в своё время, начиная с 2021 года. Полезно для разбора
    случаев — видно, что показывал прогноз в тот день, а не что было
    на самом деле.
    """
    if model_key not in MODELS:
        raise RuntimeError(
            f"Неизвестная модель «{model_key}».\n\nДоступны:\n" +
            "\n".join(f"  {k} — {v[0]}" for k, v in MODELS.items()))
    m_name, m_code, m_area = MODELS[model_key]

    # Дату и час проверяем ПЕРВЫМИ: иначе при опечатке в дате сначала
    # отработает геокодер и ошибка будет не про то, что вы ошиблись.
    try:
        day = datetime.strptime(date_str, '%Y%m%d')
    except ValueError:
        raise RuntimeError(f"Дата «{date_str}» не в том формате.\n"
                           "Нужно ГГГГММДД без разделителей, например 20260617")
    try:
        hh = int(hour)
    except ValueError:
        raise RuntimeError(f"Час «{hour}» должен быть числом от 00 до 23")
    if not 0 <= hh <= 23:
        raise RuntimeError("Час от 00 до 23")
    if day.year < 2021:
        raise RuntimeError("Архив прогнозов Open-Meteo начинается с 2021 года.\n"
                           "Для более старых дат нужен реанализ ERA5, а не архив "
                           "прогнозов.")
    iso = day.strftime('%Y-%m-%d')

    coord = re.match(r'^\s*(-?\d+(?:[.,]\d+)?)\s*[,;]\s*(-?\d+(?:[.,]\d+)?)\s*$', place)
    if coord:
        lat = float(coord.group(1).replace(',', '.'))
        lon = float(coord.group(2).replace(',', '.'))
        elev = 0
        shown = f"{lat:.3f}, {lon:.3f}"
    else:
        loc = _get_geolocator().geocode(place)
        if not loc:
            raise RuntimeError(f"Локация «{place}» не найдена.")
        lat, lon = loc.latitude, loc.longitude
        elev = round(loc.altitude) if getattr(loc, 'altitude', None) else 0
        shown = place

    v = ['temperature_2m', 'relative_humidity_2m', 'surface_pressure',
         'wind_speed_10m', 'wind_direction_10m']
    for l in LEVELS:
        v += [f'temperature_{l}hPa', f'relative_humidity_{l}hPa',
              f'geopotential_height_{l}hPa', f'wind_speed_{l}hPa',
              f'wind_direction_{l}hPa']

    # При шаге, переваливающем за полночь, одного дня мало — просим и следующий.
    end_iso = (day + timedelta(days=1)).strftime('%Y-%m-%d') if int(step) > 0 else iso

    resp = requests.get("https://historical-forecast-api.open-meteo.com/v1/forecast",
                        params={"latitude": lat, "longitude": lon,
                                "start_date": iso, "end_date": end_iso,
                                "hourly": ",".join(v), "models": m_code,
                                "windspeed_unit": "ms", "timezone": "UTC",
                                "elevation": elev}, timeout=60)
    resp.raise_for_status()
    h = resp.json().get('hourly')
    if not h or not h.get('time'):
        raise RuntimeError(f"Архив не вернул данных на {iso}. "
                           "Архив прогнозов начинается с 2021 года.")

    target = day.replace(hour=hh) + timedelta(hours=int(step))
    want = target.strftime('%Y-%m-%dT%H:00')
    try:
        idx = next(i for i, t in enumerate(h['time']) if t.startswith(want))
    except StopIteration:
        raise RuntimeError(
            f"В архиве нет срока {target.strftime('%d.%m.%Y %H:%M')} UTC.\n"
            f"За {iso} доступно с {h['time'][0][11:16]} по {h['time'][-1][11:16]} UTC.")

    prof_obj = _build_profile_from_openmeteo(h, idx, m_name, m_area)
    when = target.replace(tzinfo=timezone.utc)
    tail = f" [база {hh:02d}z +{int(step)}ч]" if int(step) else f" {hh:02d}Z"
    return (prof_obj,
            f"АРХИВ {m_name}: {shown} — {target.strftime('%d.%m.%Y')}{tail}", when)


# =====================================================================
#  Единая точка входа для любого интерфейса
# =====================================================================
HELP_TEXT = (
    "Команды:\n"
    "  /fact <ID станции> [ГГГГММДД] [00|12]\n"
    "      фактическое зондирование (Вайоминг)\n"
    "      пример:  /fact 24959 20260801 12\n\n"
    "  /frcst <город | широта,долгота> [модель] [шаг, ч]\n"
    "      прогноз, шаг от 00z сегодня:  /frcst Yakutsk gfs 12\n"
    "  /frcst <город> <модель> <база, ЧЧ> <шаг, ч>\n"
    "      шаг от указанного часа:  /frcst Yakutsk gfs 12 6  -> 18:00 UTC\n"
    "      координаты работают без geopy:  /frcst 62.03,129.73 gfs 6\n"
    "\n"
    "  /frcst old <город> <модель> <ГГГГММДД> <ЧЧ> [шаг]\n"
    "      архив прогнозов с 2021 года — как модель видела ту дату\n"
    "      пример:  /frcst old ueee ukmo 20260617 12 6\n"
    "      глобальные (работают везде): "
    + ", ".join(k for k, v in MODELS.items() if v[2] is None) + "\n"
    "      региональные: "
    + ", ".join(k for k, v in MODELS.items() if v[2]) + "\n"
    "      пример:  /frcst Yakutsk gfs 6"
)


def run_command(text):
    """
    Разбирает команду и возвращает (prof, res, title, when).

    Одна и та же функция используется и десктопом, и ботом — чтобы
    поведение команд не разъезжалось между интерфейсами.
    """
    args = (text or "").strip().split()
    if not args:
        raise RuntimeError("Пустая команда.\n\n" + HELP_TEXT)

    # срезаем ведущий слэш и @имя_бота, если команда пришла из группы
    kind = args[0].lower().lstrip('/').split('@')[0]

    if kind in ("fact", "f"):
        if len(args) < 2:
            raise RuntimeError("Формат: /fact <ID станции> [ГГГГММДД] [00|12]\n"
                               "Пример: /fact 24959 20230627 12")
        now = datetime.now(timezone.utc)
        stn = args[1]

        # Частая ошибка: пропущен ID станции, и датой оказывается первый
        # аргумент. Без проверки код молча собирал бессмыслицу вроде
        # «станция 20130719, дата ..12» и выдавал непонятную ошибку.
        if len(stn) == 8 and stn.isdigit():
            raise RuntimeError(
                f"«{stn}» похоже на дату, а не на номер станции — "
                f"кажется, пропущен ID станции.\n"
                f"Формат: /fact <ID станции> <дата> <час>\n"
                f"Например: /fact 24959 {stn} " + (args[2] if len(args) > 2 else "12"))
        if not stn.isdigit() or not (4 <= len(stn) <= 6):
            raise RuntimeError(f"«{stn}» не похоже на номер метеостанции WMO — "
                               f"это 5 цифр, например 24959 (Якутск).")

        date_str = args[2] if len(args) > 2 else now.strftime("%Y%m%d")
        if len(date_str) != 8 or not date_str.isdigit():
            raise RuntimeError(f"Дата должна быть 8 цифрами ГГГГММДД, "
                               f"а не «{date_str}».\nНапример: 20230627")
        try:
            datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            raise RuntimeError(f"Такой даты не существует: {date_str}")

        hour = args[3].zfill(2) if len(args) > 3 else ("12" if now.hour >= 12 else "00")
        if not hour.isdigit() or not (0 <= int(hour) <= 23):
            raise RuntimeError(f"Час должен быть от 00 до 23, а не «{hour}».\n"
                               f"Зондирования обычно в 00 и 12 UTC.")

        prof_obj, title, when = fetch_actual(stn, date_str, hour)

    elif kind in ("frcst", "forecast", "fc"):
        if len(args) < 2:
            raise RuntimeError("Формат: /frcst <город> [модель] [шаг, ч]\n"
                               "Архив:  /frcst old <город> <модель> <ГГГГММДД> <ЧЧ>")

        # Архивная ветка: как модель видела ту дату тогда.
        if args[1].lower() == 'old':
            if len(args) < 5:
                raise RuntimeError(
                    "Формат архива:\n"
                    "  /frcst old <город> <модель> <ГГГГММДД> <ЧЧ>\n"
                    "  /frcst old <город> <модель> <ГГГГММДД> <ЧЧ> <шаг>\n"
                    "Пример: /frcst old ueee ukmo 20260617 12 6\n"
                    "        (база 12z, +6ч → 18:00 UTC)\n"
                    "Архив прогнозов доступен с 2021 года.")
            place, mkey, dstr = args[2], args[3].lower(), args[4]
            hh = args[5] if len(args) > 5 else '12'
            try:
                hstep = int(args[6]) if len(args) > 6 else 0
            except ValueError:
                raise RuntimeError("Шаг — число часов от базы, например: "
                                   "/frcst old ueee ukmo 20260617 12 6")
            prof_obj, title, when = fetch_forecast_historical(
                place, mkey, dstr, hh, hstep)
            res = compute_verdicts(prof_obj)
            return prof_obj, res, title, when

        place = args[1]
        mkey = args[2].lower() if len(args) > 2 else 'gfs'

        # Два вида записи:
        #   /frcst <город> <модель> <шаг>          — шаг от 00z сегодня
        #   /frcst <город> <модель> <база> <шаг>   — шаг от указанного часа
        # Второй нужен, чтобы не считать в уме, сколько прошло от полуночи.
        base_hour = None
        try:
            if len(args) > 4:
                base_hour = int(args[3])
                step = int(args[4])
                # Проверяем ДО обращения к сети: незачем ждать ответ API,
                # чтобы узнать про опечатку в часе.
                if not 0 <= base_hour <= 23:
                    raise RuntimeError(f"База «{base_hour}» — час от 00 до 23")
            else:
                step = int(args[3]) if len(args) > 3 else 0
        except ValueError:
            raise RuntimeError(
                "Числа не распознаны.\n"
                "  /frcst Yakutsk gfs 12       — +12ч от 00z сегодня\n"
                "  /frcst Yakutsk gfs 12 6     — база 12z, +6ч → 18:00 UTC")
        prof_obj, title, when = fetch_forecast(place, mkey, step, base_hour)

    else:
        raise RuntimeError(f"Неизвестная команда «{kind}».\n\n" + HELP_TEXT)

    res = compute_verdicts(prof_obj)
    return prof_obj, res, title, when


def format_verdict_text(res, title, when, lang='ru', both=True):
    """
    Компактная сводка: заголовок, таблица моноширинным блоком, вердикты.

    Формат таблицы выбран потому, что колонка значений выровнена — глазу
    не приходится бегать по строке в поисках числа. K-Index и Totals-Totals
    убраны намеренно: без навыка их чтения они только загромождают вывод.
    """
    def row(label, val, unit="", digits=0):
        if val is None:
            txt = "н/д"
        else:
            try:
                f = float(val)
                txt = "н/д" if not np.isfinite(f) else f"{f:.{digits}f}{unit}"
            except (TypeError, ValueError):
                txt = "н/д"
        return f"{label:<15}{txt:>14}"

    # Зимой composite-индексы опускаем: STP, SCP и SHIP при минусе всегда
    # нули, они построены на плавучести. Место лучше отдать зимнему блоку,
    # иначе сводка перестаёт влезать в подпись к фото.
    _w = res.get('winter') or {}
    winter_mode = bool(_w.get('codes')) and _w['codes'] != ['WINTER_QUIET']

    lines = [
        "=" * 32,
        f" {title}",
        "=" * 32,
        "",
        "```",
        row("CAPE (MU)", res.get('mucape') or res.get('cape'), " Дж/кг"),
        row("MLCAPE", res.get('mlcape'), " Дж/кг"),
        row("CIN (MU)", res.get('cin'), " Дж/кг"),
        row("LI (MU)", res.get('li'), "", 1),
        "-" * 29,
    ] + ([] if winter_mode else [
        row("STP", res.get('stp'), "", 1),
        row("SCP", res.get('scp'), "", 1),
        row("SHIP", res.get('ship'), "", 1),
        "-" * 29,
    ]) + [
        row("Сдвиг 0-6км", res.get('shear06_ms'), " м/с", 1),
        row("Сдвиг 0-3км", res.get('shear03_ms'), " м/с", 1),
        row("SRH 0-1км", res.get('srh01'), " м²/с²"),
        row("SRH 0-3км", res.get('srh03'), " м²/с²"),
        "-" * 29,
        row("PW (влага)", res.get('pw'), " мм"),
        row("DCAPE", res.get('dcape'), " Дж/кг"),
        "-" * 29,
        row("LCL", res.get('lcl'), " м"),
        row("LFC", res.get('lfc'), " м"),
        row("EL", res.get('el'), " м"),
        "```",
        "",
    ]

    def block(lg):
        mode = MODE_TEXT.get(res.get('mode_code', ''), {}).get(lg) or res.get('mode', '')
        precip = PRECIP_TEXT.get(res.get('precip_code', ''), {}).get(lg) or res.get('precip', '')
        codes = res.get('codes') or []
        verds = [VERDICT_TEXT[c][lg] for c in codes if c in VERDICT_TEXT] \
            or res.get('verdicts', [])
        if lg == 'en':
            head = ["Storm Mode:  " + mode, "Precip:      " + precip, "", "VERDICTS:"]
        else:
            head = ["Режим:   " + mode, "Осадки:  " + precip, "", "ВЕРДИКТЫ:"]
        return head + verds

    lines += block('ru')

    # Зимний блок: показывается ТОЛЬКО когда есть что показать. Летние
    # композиты при минусе бессмысленны — они построены на плавучести,
    # которой в холодной атмосфере почти нет, — поэтому зимние явления
    # считаются отдельным набором критериев.
    win = res.get('winter')
    if win and win.get('codes') and win['codes'] != ['WINTER_QUIET']:
        try:
            from winter_hazards import format_winter
            wtxt = format_winter(win, 'ru')
            if wtxt:
                lines += ["", "-" * 32, ""] + wtxt.split("\n")
        except ImportError:
            pass

    if both:
        lines += ["", "-" * 32, ""] + block('en')
    lines += ["", "-" * 32, f"Данные: {when.strftime('%d.%m.%Y %H:%M')} UTC"]
    return "\n".join(lines)