"""
ЗИМНИЕ ОПАСНЫЕ ЯВЛЕНИЯ для профиля SHARPpy.

Летние композиты — STP, SCP, SigSevere — зимой бесполезны: они построены
на плавучести, которой в холодной атмосфере почти нет. Здесь свои
критерии, для явлений, которые в Якутии реально опасны.

Что считается:

  МЕТЕЛЬ И БУРАН — перенос снега ветром. Главная беда Булунского улуса
  и всего арктического побережья: при −30 и ветре 15 м/с видимость
  падает до метров, и это опаснее любой летней грозы.

  ГОЛОЛЁД И ЛЕДЯНОЙ ДОЖДЬ — тёплая прослойка над мёрзлой землёй.
  Снег в ней подтаивает, а у поверхности капли снова замерзают.
  Классический признак — «тёплый нос» на профиле температуры.

  СНЕЖНАЯ ГРОЗА — редкость, но случается: неустойчивость при
  отрицательных температурах, обычно за холодным фронтом или над
  открытой водой.

  МОРОЗ С ВЕТРОМ — не явление само по себе, но при −40 и ветре
  обморожение наступает за минуты, и предупреждать об этом надо.

Пороги подобраны под Якутию, а не по учебникам умеренных широт: −30 °C
здесь обычная зимняя температура, а не бедствие, и порог мороза стоит
там, где начинается настоящая опасность.
"""

import numpy as np

# ---------------------------------------------------------------------
#  Пороги. Вынесены наверх, чтобы правились без чтения кода.
# ---------------------------------------------------------------------
WINTER = {
    # Метель: ветер у земли, м/с
    'blizzard_wind': 15.0,       # перенос снега становится сплошным
    'blizzard_wind_severe': 20.0,
    'drift_wind': 9.0,           # низовая позёмка

    # Температура, при которой снег остаётся сухим и переносится
    'snow_dry_max': -2.0,

    # Гололёд: тёплая прослойка
    'warm_nose_min': 0.5,        # °C, насколько прослойка выше нуля
    'warm_nose_depth': 300.0,    # м, минимальная толщина
    'sfc_freeze_max': -0.5,      # °C у земли, ниже — замерзает при ударе

    # Снежная гроза
    'thundersnow_cape': 100.0,   # Дж/кг, достаточно немного
    'thundersnow_tmax': 0.0,     # °C у земли

    # Мороз с ветром
    # Пороги по ОЩУЩАЕМОЙ температуре: она и определяет время до
    # обморожения. Для Якутии подняты против общепринятых — здесь
    # −30 рядовая зимняя величина, а не бедствие.
    'chill_extreme': -50.0,      # обморожение за 2-5 минут
    'chill_severe': -40.0,       # за 5-10 минут
    # Нижняя граница по САМОМУ воздуху. Без неё ощущаемые -40 набираются
    # уже при -24 и ветре 12 м/с, то есть половину зимы, и вердикт
    # перестаёт что-либо значить. Здесь -30 — это уже заметно холоднее
    # рядового якутского дня.
    'cold_air_floor': -30.0,
    'windchill_wind': 5.0,       # м/с, ниже ветер почти не усиливает
}

WINTER_TEXT = {
    'BLIZZARD_SEVERE': {
        'ru': "❄️🌪 БУРАН: сильная метель, видимость почти нулевая",
        'en': "❄️🌪 SEVERE BLIZZARD: near-zero visibility"},
    'BLIZZARD': {
        'ru': "❄️ МЕТЕЛЬ: перенос снега, видимость сильно снижена",
        'en': "❄️ BLIZZARD: blowing snow, reduced visibility"},
    'DRIFT': {
        'ru': "🌬 ПОЗЁМКА: низовой перенос снега",
        'en': "🌬 DRIFTING SNOW: low-level blowing snow"},
    'FREEZING_RAIN': {
        'ru': "🧊 ГОЛОЛЁД: ледяной дождь, налипание на провода и дороги",
        'en': "🧊 FREEZING RAIN: ice accretion"},
    'ICE_PELLETS': {
        'ru': "🧊 ЛЕДЯНАЯ КРУПА: тёплая прослойка тонкая",
        'en': "🧊 ICE PELLETS: shallow warm layer"},
    'THUNDERSNOW': {
        'ru': "⚡❄️ СНЕЖНАЯ ГРОЗА: неустойчивость при минусе",
        'en': "⚡❄️ THUNDERSNOW: instability below freezing"},
    'COLD_EXTREME': {
        'ru': "🥶 ЖЕСТОКИЙ МОРОЗ: обморожение за минуты",
        'en': "🥶 EXTREME COLD: frostbite in minutes"},
    'COLD_WIND': {
        'ru': "🥶 МОРОЗ С ВЕТРОМ: открытую кожу закрывать",
        'en': "🥶 COLD + WIND: cover exposed skin"},
    'COLD_STILL': {
        'ru': "🥶 СИЛЬНЫЙ МОРОЗ",
        'en': "🥶 SEVERE COLD"},
    'WINTER_QUIET': {
        'ru': "🌙 Зимняя погода без опасных явлений",
        'en': "🌙 Winter weather, no hazards"},
}


def _wind_chill(t_c, v_ms):
    """
    Ощущаемая температура по формуле, принятой в Канаде и России.

    Скорость переводится в км/ч: формула выведена именно для них.
    Ниже 5 км/ч ветер уже не усиливает охлаждение, там формула
    неприменима и возвращается сама температура.
    """
    v_kmh = v_ms * 3.6
    if v_kmh < 5.0 or t_c > 10.0:
        return t_c
    return (13.12 + 0.6215 * t_c - 11.37 * v_kmh ** 0.16
            + 0.3965 * t_c * v_kmh ** 0.16)


def _warm_nose(prof):
    """
    Ищет тёплую прослойку над мёрзлым приземным слоем.

    Возвращает (толщина в метрах, максимальная температура в ней).
    Это и есть механизм ледяного дождя: снег наверху тает в прослойке,
    а у земли капли переохлаждаются и намерзают при ударе.
    """
    try:
        h = np.asarray(prof.hght, dtype=float)
        t = np.asarray(prof.tmpc, dtype=float)
    except Exception:
        return 0.0, None

    ok = np.isfinite(h) & np.isfinite(t)
    h, t = h[ok], t[ok]
    if len(h) < 5:
        return 0.0, None

    h = h - h[0]                     # от поверхности
    # Смотрим только нижние 4 км: выше прослойка на осадки не влияет.
    m = h <= 4000.0
    h, t = h[m], t[m]
    if len(h) < 4 or t[0] > 0.0:
        return 0.0, None             # у земли не мороз — гололёда не будет

    warm = t > WINTER['warm_nose_min']
    if not warm.any():
        return 0.0, None

    # Толщина непрерывного тёплого слоя и его максимум
    idx = np.where(warm)[0]
    breaks = np.where(np.diff(idx) > 1)[0]
    groups = np.split(idx, breaks + 1)
    best_depth, best_t = 0.0, None
    for g in groups:
        if len(g) == 0:
            continue
        if len(g) == 1:
            # Один узел выше нуля — на модельной сетке с шагом в
            # сотни метров этого достаточно: толщину оцениваем по
            # половине расстояния до соседей, иначе гололёд на
            # грубых профилях просто не найдётся.
            i = g[0]
            lo = h[i - 1] if i > 0 else h[i]
            hi = h[i + 1] if i + 1 < len(h) else h[i]
            depth = float((hi - lo) / 2.0)
        else:
            depth = float(h[g[-1]] - h[g[0]])
        if depth > best_depth:
            best_depth, best_t = depth, float(np.max(t[g]))
    return best_depth, best_t


def winter_verdicts(prof, res=None):
    """
    Зимние вердикты по профилю.

    res — словарь от compute_verdicts, если он уже посчитан: оттуда
    берётся CAPE для снежной грозы. Без него CAPE считается нулевым.

    Возвращает словарь с кодами, пояснениями и величинами.
    """
    out = {'codes': [], 'details': {}}

    try:
        t_sfc = float(prof.tmpc[prof.sfc])
        td_sfc = float(prof.dwpc[prof.sfc])
        u0 = float(prof.u[prof.sfc])
        v0 = float(prof.v[prof.sfc])
    except Exception:
        return out

    # SHARPpy держит ветер в УЗЛАХ — переводим в м/с
    wind = float(np.hypot(u0, v0)) * 0.514444
    out['details']['t_sfc'] = round(t_sfc, 1)
    out['details']['wind_ms'] = round(wind, 1)

    if t_sfc > 5.0:
        return out                   # не зима

    # ---- мороз и ветер ----
    chill = _wind_chill(t_sfc, wind)
    out['details']['wind_chill'] = round(chill, 1)
    # Опасность определяет ОЩУЩАЕМАЯ температура, а не показания
    # термометра: при −32 и ветре 22 м/с ощущается −55, и обморожение
    # наступает быстрее, чем при −40 в штиль. Пороги — по шкале времени
    # обморожения: ниже −50 счёт идёт на минуты, ниже −40 на десяток.
    cold_enough = t_sfc <= WINTER['cold_air_floor'] + 0.01
    if chill <= WINTER['chill_extreme'] and cold_enough:
        out['codes'].append('COLD_EXTREME')
    elif chill <= WINTER['chill_severe'] and cold_enough:
        # Разводим два разных случая: холод сам по себе и холод,
        # усиленный ветром. Совет человеку в них разный — в первом
        # хватит одежды, во втором надо закрывать лицо.
        out['codes'].append('COLD_WIND' if wind >= WINTER['windchill_wind']
                            else 'COLD_STILL')

    # ---- метель ----
    # Нужен сухой снег: при оттепели он слипается и не переносится.
    snow_dry = t_sfc <= WINTER['snow_dry_max']
    if snow_dry:
        if wind >= WINTER['blizzard_wind_severe']:
            out['codes'].append('BLIZZARD_SEVERE')
        elif wind >= WINTER['blizzard_wind']:
            out['codes'].append('BLIZZARD')
        elif wind >= WINTER['drift_wind']:
            out['codes'].append('DRIFT')

    # ---- гололёд ----
    depth, t_warm = _warm_nose(prof)
    out['details']['warm_nose_m'] = round(depth)
    if t_warm is not None:
        out['details']['warm_nose_t'] = round(t_warm, 1)
    if depth > 0 and t_sfc <= WINTER['sfc_freeze_max']:
        # Толстая прослойка — снежинка успевает растаять полностью,
        # и выпадает переохлаждённая капля: гололёд.
        # Тонкая — тает не до конца, капля замерзает в падении:
        # ледяная крупа, она безопаснее.
        if depth >= WINTER['warm_nose_depth']:
            out['codes'].append('FREEZING_RAIN')
        else:
            out['codes'].append('ICE_PELLETS')

    # ---- снежная гроза ----
    cape = 0.0
    if res:
        for k in ('mucape', 'cape', 'mlcape'):
            v = res.get(k)
            if isinstance(v, (int, float)) and np.isfinite(v):
                cape = max(cape, float(v))
    out['details']['cape'] = round(cape)
    if (cape >= WINTER['thundersnow_cape']
            and t_sfc <= WINTER['thundersnow_tmax']):
        out['codes'].append('THUNDERSNOW')

    if not out['codes']:
        out['codes'].append('WINTER_QUIET')
    return out


def format_winter(w, lang='ru'):
    """Короткий блок для бота."""
    if not w or not w.get('codes'):
        return ""
    d = w.get('details', {})
    lines = ["ЗИМНИЕ ЯВЛЕНИЯ:" if lang == 'ru' else "WINTER HAZARDS:"]
    for c in w['codes']:
        lines.append("  " + WINTER_TEXT.get(c, {}).get(lang, c))
    if 't_sfc' in d:
        chill = d.get('wind_chill')
        tail = ""
        if chill is not None and abs(chill - d['t_sfc']) >= 2:
            tail = f", ощущается {chill:.0f}"
        lines.append(f"  ({d['t_sfc']:.0f} °C{tail}, "
                     f"ветер {d.get('wind_ms', 0):.0f} м/с)")
    if d.get('warm_nose_m'):
        lines.append(f"  тёплая прослойка {d['warm_nose_m']} м, "
                     f"до {d.get('warm_nose_t', 0):.1f} °C")
    return "\n".join(lines)
