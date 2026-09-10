#!/usr/bin/env python3
"""
Ежедневная публикация зондирования в группу.

Запускается по расписанию на GitHub Actions. Считает профиль, рисует
панель Skew-T и отправляет её в Телеграм.

Выключатель: файл PAUSED в корне репозитория. Пока он есть, публикация
пропускается. Это единственный способ остановить расписание без
постоянно работающего процесса — Actions ничего не слушает, он только
просыпается по времени.

    python daily_post.py --command "/fact ueee 12"
    python daily_post.py --command "/frcst ueee ukmo 6" --title "Прогноз"
"""

import os
import io
import sys
import argparse
import datetime

import matplotlib
matplotlib.use("Agg")


def paused(root):
    """
    Проверяет выключатель.

    Файл, а не команда из Телеграма: у расписания нет процесса, который
    мог бы слушать сообщения. Файл создаётся через веб-интерфейс GitHub
    или его приложение на телефоне — это доступно откуда угодно.
    """
    for name in ("PAUSED", "PAUSED.txt", "paused"):
        p = os.path.join(root, name)
        if os.path.exists(p):
            try:
                why = open(p, encoding="utf-8").read().strip()
            except OSError:
                why = ""
            return why or "без пояснения"
    return None


def send_photo(token, chat_id, buf, caption, parse_mode="HTML"):
    import requests
    r = requests.post(
        f"https://api.telegram.org/bot{token}/sendPhoto",
        data={"chat_id": chat_id, "caption": caption[:1024],
              "parse_mode": parse_mode},
        files={"photo": ("sounding.png", buf, "image/png")},
        timeout=60)
    r.raise_for_status()
    return r.json()


def send_message(token, chat_id, text, parse_mode="HTML"):
    import requests
    r = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data={"chat_id": chat_id, "text": text[:4096],
              "parse_mode": parse_mode},
        timeout=60)
    r.raise_for_status()
    return r.json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--command", required=True,
                    help="команда для sharppy_core, например /fact ueee 12")
    ap.add_argument("--title", default="",
                    help="подпись перед сводкой")
    ap.add_argument("--chat", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="посчитать и сохранить файл, но не отправлять")
    a = ap.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    why = paused(root)
    if why:
        print(f"Публикация приостановлена: {why}")
        print("Уберите файл PAUSED из репозитория, чтобы возобновить.")
        return 0

    token = os.environ.get("TELEGRAM_TOKEN")
    chat = a.chat or os.environ.get("TELEGRAM_CHAT")
    if not a.dry_run and (not token or not chat):
        sys.exit("Нет TELEGRAM_TOKEN или TELEGRAM_CHAT в переменных "
                 "окружения. На GitHub они задаются в Secrets.")

    sys.path.insert(0, root)
    import sharppy_core as core

    print(f"Команда: {a.command}")
    try:
        prof, res, title, when = core.run_command(a.command)
    except Exception as e:
        # Молчать нельзя: если данных нет, об этом лучше сообщить в
        # группу, чем оставить людей гадать, почему сегодня пусто.
        msg = (f"⚠️ Не удалось построить зондирование\n\n"
               f"<code>{a.command}</code>\n\n"
               f"{type(e).__name__}: {e}")
        print(msg)
        if not a.dry_run:
            try:
                send_message(token, chat, msg)
            except Exception as e2:
                print(f"и сообщить не вышло: {e2}")
        return 1

    fig = core.draw_skewt(prof, res, title)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor="white")
    buf.seek(0)

    text = core.format_verdict_text(res, title, when, both=False)

    # Тот же перевод в HTML, что у бота: таблица должна остаться
    # моноширинной, иначе колонки разъедутся.
    import html as _html
    out, in_code = [], False
    for line in text.split("\n"):
        if line.strip() == "```":
            out.append("</pre>" if in_code else "<pre>")
            in_code = not in_code
            continue
        out.append(_html.escape(line))
    if in_code:
        out.append("</pre>")
    caption = "\n".join(out)
    if a.title:
        caption = f"<b>{_html.escape(a.title)}</b>\n" + caption

    if a.dry_run:
        p = os.path.join(root, "daily_preview.png")
        with open(p, "wb") as f:
            f.write(buf.getvalue())
        print(f"\nСохранено: {p}")
        print(f"Подпись ({len(caption)} знаков):\n")
        print(text)
        return 0

    if len(caption) <= 1000:
        send_photo(token, chat, buf, caption)
    else:
        send_photo(token, chat, buf, a.title or title)
        send_message(token, chat, caption)
    print("Отправлено.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
