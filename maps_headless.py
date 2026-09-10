#!/usr/bin/env python3
"""
Карты рисков БЕЗ ОКНА — для расписания на сервере.

Логика расчёта не дублируется: она берётся из gfs_sharppy_map_fixed
как есть. Вместо окна подставляется заглушка с теми же полями, что
читает метод расчёта, — так исключён разъезд между тем, что вы видите
на своей машине, и тем, что уходит в группу.

    python maps_headless.py --config maps.json
    python maps_headless.py --preset yakutsk --hours 12

Настройки — обычный JSON. Пример создаётся ключом --write-config.
"""

import os
import io
import sys
import json
import argparse
import datetime

import matplotlib
matplotlib.use("Agg")


# ---------------------------------------------------------------------
#  Заглушки вместо элементов окна
# ---------------------------------------------------------------------
class Val:
    """Подмена tk.BooleanVar и tk.StringVar: нужен только .get()."""

    def __init__(self, v):
        self._v = v

    def get(self):
        return self._v

    def set(self, v):
        self._v = v


class Entry(Val):
    """Подмена ttk.Entry — тот же .get(), но всегда строка."""

    def get(self):
        return str(self._v)


class Widget:
    """Заглушка для кнопок и полосы: расчёт их трогает, но результат
    никому не нужен."""

    def config(self, **kw):
        pass

    def __setitem__(self, k, v):
        pass

    def __getitem__(self, k):
        return 0

    def start(self, *a):
        pass

    def stop(self, *a):
        pass


DEFAULTS = {
    "region": {"lat_min": 55.0, "lat_max": 72.0,
               "lon_min": 105.0, "lon_max": 163.0, "step": 0.5},
    "source": "nomads",
    "om_model": "ICON Global 11км",
    "date": "today",
    "hour": "00",
    "step_h": "012",
    "workers": 0,
    "params": ["outlook_tornado", "outlook_overall", "outlook_trigger",
               "mucape", "shear6k"],
    "daily": True,
    "fronts": True,
    "fronts_draw": True,
    "fronts_risk": False,
    "mask_yakutia": True,
    "climo": "yakutia_thresholds.json",
    "out_dir": "",
}

#: Соответствие имён из конфига полям расчёта.
PARAM_FIELDS = {
    "sbcape": "cb_sbcape", "mlcape": "cb_mlcape", "mucape": "cb_mucape",
    "cape3k": "cb_cape3k", "mlcin": "cb_mlcin", "sbcin": "cb_sbcin",
    "shear1k": "cb_shear1k", "shear3k": "cb_shear3k", "shear6k": "cb_shear6k",
    "srh1k": "cb_srh1k", "srh3k": "cb_srh3k", "scp": "cb_scp", "stp": "cb_stp",
    "outlook_tornado": "cb_outlook_tornado",
    "outlook_overall": "cb_outlook_overall",
    "outlook_trigger": "cb_outlook_trigger",
}


class HeadlessApp:
    """
    Подставляется вместо окна. Держит те же поля, что читает расчёт,
    и складывает готовые фигуры в generated_figs — оттуда их забирает
    отправка.
    """

    def __init__(self, cfg, log=print):
        self._log = log
        self.generated_figs = []
        self.current_model = "GFS"

        r = cfg["region"]
        self.lat_min_entry = Entry(r["lat_min"])
        self.lat_max_entry = Entry(r["lat_max"])
        self.lon_min_entry = Entry(r["lon_min"])
        self.lon_max_entry = Entry(r["lon_max"])
        self.grid_step_entry = Entry(r["step"])

        self.source_var = Val(cfg["source"])
        self.om_model_var = Val(cfg["om_model"])

        date = cfg["date"]
        if date == "today":
            date = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        elif date == "yesterday":
            date = (datetime.datetime.utcnow()
                    - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        self.date_entry = Entry(date)
        self.hour_entry = Entry(str(cfg["hour"]).zfill(2))
        self.step_entry = Entry(str(cfg["step_h"]).zfill(3))
        self.workers_entry = Entry(cfg["workers"] or "")

        on = set(cfg["params"])
        for name, field in PARAM_FIELDS.items():
            setattr(self, field, Val(name in on))

        self.cb_daily = Val(cfg["daily"])
        self.cb_fronts = Val(cfg["fronts"])
        self.cb_fronts_draw = Val(cfg["fronts_draw"])
        self.cb_fronts_risk = Val(cfg["fronts_risk"])
        self.cb_mask_yakutia = Val(cfg["mask_yakutia"])

        self.btn_run = Widget()
        self.btn_save = Widget()
        self.progress = Widget()
        self.status_label = Widget()

    def log(self, text):
        self._log(text)

    # Эти два метода расчёт зовёт только для ERA5 — заимствуем их
    # у настоящего класса, чтобы не дублировать логику.
    def _download_era5_cds(self, *a, **kw):
        from gfs_sharppy_map_fixed import GFSMapApp
        return GFSMapApp._download_era5_cds(self, *a, **kw)

    @staticmethod
    def _era5_add_geopotential_height(ds):
        from gfs_sharppy_map_fixed import GFSMapApp
        return GFSMapApp._era5_add_geopotential_height(ds)

    def _build_outlook_legend_figure(self):
        from gfs_sharppy_map_fixed import GFSMapApp
        return GFSMapApp._build_outlook_legend_figure(self)


def load_config(path):
    cfg = json.loads(json.dumps(DEFAULTS))     # глубокая копия
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            user = json.load(f)
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="maps.json")
    ap.add_argument("--write-config", action="store_true",
                    help="создать образец настроек и выйти")
    ap.add_argument("--out", default=None, help="куда класть картинки")
    ap.add_argument("--hours", default=None, help="шаг прогноза, часов")
    ap.add_argument("--date", default=None)
    ap.add_argument("--hour", default=None)
    a = ap.parse_args()

    if a.write_config:
        with open(a.config, "w", encoding="utf-8") as f:
            json.dump(DEFAULTS, f, ensure_ascii=False, indent=2)
        print(f"Образец настроек: {a.config}")
        return 0

    cfg = load_config(a.config)
    if a.hours:
        cfg["step_h"] = a.hours
    if a.date:
        cfg["date"] = a.date
    if a.hour:
        cfg["hour"] = a.hour
    out_dir = a.out or cfg.get("out_dir") or os.path.join(
        os.getcwd(), "maps_out")

    root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, root)

    # tkinter нужен только чтобы модуль импортировался: класс окна
    # объявляется на уровне модуля, но мы его не создаём.
    try:
        import gfs_sharppy_map_fixed as M
    except Exception as e:
        sys.exit(f"Не импортируется модуль карт: {type(e).__name__}: {e}")

    # Пороги: тот же загрузчик, что в окне.
    climo = cfg.get("climo")
    if climo:
        if M.load_climo_thresholds(M._find_climo_file(climo)):
            print(f"Пороги: {M.CLIMO_SOURCE}")
        else:
            M.use_spc_thresholds()
            print(f"Файл {climo} не найден — считаю по порогам SPC.")

    app = HeadlessApp(cfg, log=lambda t: print(t, flush=True))

    print(f"Область: {cfg['region']}")
    print(f"Срок: {app.date_entry.get()} {app.hour_entry.get()}z "
          f"+{app.step_entry.get()}ч")
    print(f"Параметры: {', '.join(cfg['params'])}")
    print()

    t0 = datetime.datetime.now()
    M.GFSMapApp._worker_process(app)
    dt = (datetime.datetime.now() - t0).total_seconds()

    if not app.generated_figs:
        sys.exit("Ни одной карты не построено — смотрите вывод выше.")

    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for key, fig in app.generated_figs:
        name = f"{key.upper()}_{app.date_entry.get().replace('-', '')}" \
               f"_{app.hour_entry.get()}_F{app.step_entry.get()}.png"
        p = os.path.join(out_dir, name)
        fig.savefig(p, dpi=150, facecolor="white")
        saved.append(p)
        print(f"  {name}")

    print(f"\nГотово за {dt / 60:.1f} мин, карт: {len(saved)}")
    print(f"Папка: {out_dir}")

    # Список для следующего шага — отправки в группу.
    with open(os.path.join(out_dir, "maps.json"), "w", encoding="utf-8") as f:
        json.dump({"files": saved, "date": app.date_entry.get(),
                   "hour": app.hour_entry.get(),
                   "step": app.step_entry.get(),
                   "model": app.current_model}, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
