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


def check_token(token, chat_id):
    """
    Проверяет, что бот существует и группа ему доступна.

    Телеграм отвечает 404 на неверный токен — это чаще всего означает,
    что при копировании в Secrets потерялась часть строки или добавился
    пробел.
    """
    import requests
    try:
        r = requests.get(f"https://api.telegram.org/bot{token}/getMe",
                         timeout=30)
        if r.status_code == 404:
            return False, ("Телеграм не знает такого бота (404). Токен в "
                           "TELEGRAM_TOKEN неверный или скопирован не "
                           "целиком — он выглядит как 1234567890:AAE... "
                           "и содержит двоеточие.")
        r.raise_for_status()
        name = (r.json().get("result") or {}).get("username", "?")
    except Exception as e:
        return False, f"Не удалось проверить токен: {type(e).__name__}: {e}"

    try:
        r = requests.get(f"https://api.telegram.org/bot{token}/getChat",
                         params={"chat_id": chat_id}, timeout=30)
        if r.status_code != 200:
            return False, (f"Бот @{name} существует, но группа {chat_id} "
                           f"ему недоступна ({r.status_code}). Проверьте "
                           f"номер группы и что бот в неё добавлен.")
    except Exception as e:
        return False, f"Группа не проверилась: {type(e).__name__}: {e}"

    print(f"Бот @{name}, группа {chat_id} — доступны.")
    return True, ""


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

    token = os.environ.get("TELEGRAM_TOKEN", "").strip()
    chat = (a.chat or os.environ.get("TELEGRAM_CHAT", "")).strip()
    if not a.dry_run:
        if not token or not chat:
            sys.exit("Нет TELEGRAM_TOKEN или TELEGRAM_CHAT в переменных "
                     "окружения. На GitHub они задаются в Secrets.")
        # Проверяем токен ДО расчёта: иначе профиль считается несколько
        # минут, а потом выясняется, что отправить его некуда. Заодно
        # 404 здесь сразу означает неверный токен, а не сбой сети.
        ok, why = check_token(token, chat)
        if not ok:
            sys.exit(why)

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
