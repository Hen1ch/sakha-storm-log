#!/usr/bin/env python3
"""
Отправка готовых карт в группу.

Читает maps.json, оставленный maps_headless.py, и шлёт картинки
альбомом — так они приходят одним сообщением, а не засоряют ленту
десятком отдельных.

    python send_maps.py --dir maps_out
    python send_maps.py --dir maps_out --title "Прогноз на сегодня"
"""

import os
import sys
import json
import argparse

#: Порядок и подписи. Сначала главное — риски, потом параметры.
ORDER = [
    ("OUTLOOK_TORNADO", "Торнадо-риск"),
    ("OUTLOOK_OVERALL", "Опасные явления"),
    ("OUTLOOK_TRIGGER", "Реальность инициации"),
    ("MUCAPE", "MUCAPE"),
    ("MLCAPE", "MLCAPE"),
    ("SBCAPE", "SBCAPE"),
    ("CAPE3K", "3CAPE"),
    ("MLCIN", "MLCIN — крышка"),
    ("SBCIN", "SBCIN — крышка"),
    ("SHEAR6K", "Сдвиг 0-6 км"),
    ("SHEAR3K", "Сдвиг 0-3 км"),
    ("SHEAR1K", "Сдвиг 0-1 км"),
    ("SRH1K", "SRH 0-1 км"),
    ("SRH3K", "SRH 0-3 км"),
    ("STP", "STP"),
    ("SCP", "SCP"),
]


def sort_key(path):
    """Ставит карты в осмысленный порядок, а не по алфавиту."""
    name = os.path.basename(path).upper()
    for i, (key, _) in enumerate(ORDER):
        if name.startswith(key):
            return (i, name)
    return (len(ORDER), name)


def caption_for(path):
    name = os.path.basename(path).upper()
    for key, label in ORDER:
        if name.startswith(key):
            return label
    if "LEGEND" in name:
        return "Что означают категории"
    return os.path.splitext(os.path.basename(path))[0]


def send_album(token, chat, files, first_caption=""):
    """
    Шлёт до 10 картинок одним сообщением.

    Подпись у альбома может быть только у первой картинки — так
    устроен Телеграм. Поэтому общий заголовок ставится ей, а
    остальные подписываются по отдельности.
    """
    import requests

    media, handles = [], []
    try:
        for i, p in enumerate(files):
            f = open(p, "rb")
            handles.append(f)
            item = {"type": "photo", "media": f"attach://f{i}"}
            cap = first_caption if i == 0 and first_caption else caption_for(p)
            if cap:
                item["caption"] = cap[:1024]
                item["parse_mode"] = "HTML"
            media.append(item)
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMediaGroup",
            data={"chat_id": chat, "media": json.dumps(media)},
            files={f"f{i}": h for i, h in enumerate(handles)},
            timeout=120)
        r.raise_for_status()
        return r.json()
    finally:
        for h in handles:
            try:
                h.close()
            except OSError:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="maps_out")
    ap.add_argument("--title", default="")
    ap.add_argument("--chat", default=None)
    ap.add_argument("--max", type=int, default=10,
                    help="сколько карт слать (Телеграм берёт до 10 в альбом)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if not os.path.isdir(a.dir):
        # Обычно означает, что расчёт не дошёл до конца. Своей ошибкой
        # это не перекрываем: настоящая причина в шаге выше.
        print(f"Папки {a.dir} нет — карты не построились. "
              f"Смотрите шаг «Построить карты».")
        return 0

    meta_path = os.path.join(a.dir, "maps.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        files = [p for p in meta.get("files", []) if os.path.exists(p)]
    else:
        files = [os.path.join(a.dir, n) for n in os.listdir(a.dir)
                 if n.lower().endswith(".png")]
        meta = {}

    if not files:
        sys.exit(f"В {a.dir} нет карт.")

    files.sort(key=sort_key)
    # Легенду не шлём каждый раз: она не меняется, а место в альбоме
    # занимает. Люди посмотрят её один раз в закреплённом сообщении.
    files = [p for p in files if "LEGEND" not in os.path.basename(p).upper()]
    files = files[:a.max]

    title = a.title
    if not title and meta:
        title = (f"<b>{meta.get('model', 'GFS')}</b>  "
                 f"{meta.get('date', '')} {meta.get('hour', '')}z "
                 f"+{meta.get('step', '')}ч")

    print(f"Карт к отправке: {len(files)}")
    for p in files:
        print(f"  {caption_for(p):<28} {os.path.basename(p)}")

    if a.dry_run:
        print("\n(пробный прогон, ничего не отправлено)")
        return 0

    token = os.environ.get("TELEGRAM_TOKEN")
    chat = a.chat or os.environ.get("TELEGRAM_CHAT")
    if not token or not chat:
        sys.exit("Нет TELEGRAM_TOKEN или TELEGRAM_CHAT.")

    send_album(token, chat, files, title)
    print("Отправлено.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
