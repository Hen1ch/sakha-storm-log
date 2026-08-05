#!/usr/bin/env python3
"""
Поиск кандидатов в архив по новостным лентам.

ВАЖНО ПРО УСТРОЙСТВО: скрипт НИЧЕГО НЕ ДОБАВЛЯЕТ в архив. Он собирает
находки и открывает Issue со списком, который человек проверяет вручную.

Так сделано намеренно. Архив нужен в том числе для калибровки прогноза,
а туда попадание выдуманного события испортит калибровку незаметно —
разницу между «агент нашёл и записал» и «агент нашёл и предложил»
видно не сразу, но она отделяет полезный архив от свалки правдоподобного
текста.

Работает без ключей и без внешних зависимостей: RSS Google Новостей
плюс стандартная библиотека Python.
"""

import os
import re
import json
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

UA = "Mozilla/5.0 (compatible; sakha-storm-log/1.0)"
FEED = "https://news.google.com/rss/search?q={q}&hl=ru&gl=RU&ceid=RU:ru"

# Запросы: явление + привязка к региону. Без привязки лента забивается
# смерчами со всего мира.
REGION = ['Якутия', 'Якутске', 'Саха', 'улусе']
PHENOMENA = [
    ('tornado', ['смерч', 'торнадо', 'вихрь']),
    ('squall', ['шквал', 'ураган', 'повалило деревья', 'сорвало крышу']),
    ('hail', ['град']),
    ('thunderstorm', ['ливень подтопил', 'гроза затопила', 'ливневый паводок']),
    ('fire', ['сухая гроза', 'молния пожар']),
]

# Слова, по которым отсекаем явно чужое: спорт, кино, переносные значения.
NOISE = re.compile(
    r'\b(футбол|хоккей|матч|сериал|фильм|игр[аеы]\b|акци[ия]|курс|'
    r'биржа|котировк|аниме|манга)', re.I)

DAYS_BACK = int(os.environ.get('SCAN_DAYS', '14'))


def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def parse_feed(xml_bytes):
    out = []
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return out
    for item in root.iterfind('.//item'):
        title = (item.findtext('title') or '').strip()
        link = (item.findtext('link') or '').strip()
        pub = (item.findtext('pubDate') or '').strip()
        src_el = item.find('source')
        src = (src_el.text or '').strip() if src_el is not None else ''
        out.append({'title': title, 'link': link, 'pub': pub, 'src': src})
    return out


def pubdate_to_iso(pub):
    for fmt in ('%a, %d %b %Y %H:%M:%S %Z', '%a, %d %b %Y %H:%M:%S %z'):
        try:
            return datetime.strptime(pub, fmt).strftime('%Y-%m-%d')
        except ValueError:
            continue
    return ''


def load_known(path):
    """Уже внесённые случаи — чтобы не предлагать их снова."""
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        return {(e['date'], e['cat']) for e in data.get('events', [])}, data
    except Exception:
        return set(), {'events': []}


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    archive_path = os.path.join(root, 'data', 'archive.json')
    known, _ = load_known(archive_path)

    cutoff = (datetime.now(timezone.utc) - timedelta(days=DAYS_BACK)).strftime('%Y-%m-%d')
    seen_links, found = set(), []

    for cat, words in PHENOMENA:
        for w in words:
            for reg in REGION:
                q = urllib.parse.quote(f'{w} {reg}')
                try:
                    items = parse_feed(fetch(FEED.format(q=q)))
                except Exception as e:
                    print(f"  запрос «{w} {reg}» не удался: {e}", file=sys.stderr)
                    continue
                for it in items:
                    if it['link'] in seen_links:
                        continue
                    if NOISE.search(it['title']):
                        continue
                    date = pubdate_to_iso(it['pub'])
                    if not date or date < cutoff:
                        continue
                    if (date, cat) in known:
                        continue
                    seen_links.add(it['link'])
                    found.append({'cat': cat, 'date': date, 'query': f'{w} {reg}',
                                  **it})

    found.sort(key=lambda x: x['date'], reverse=True)
    print(f"Кандидатов за последние {DAYS_BACK} дн.: {len(found)}")

    if not found:
        print("Ничего нового — Issue не создаётся.")
        return

    CAT_RU = {'tornado': 'Смерч', 'squall': 'Шквал', 'hail': 'Град',
              'thunderstorm': 'Гроза/ливень', 'fire': 'Сухая гроза → пожар'}

    lines = [
        f"Автоматический обзор новостей за последние {DAYS_BACK} дней.",
        "",
        "**Это кандидаты, а не записи.** Ничего не попало в архив — "
        "проверьте каждый пункт и внесите вручную то, что подтвердилось.",
        "",
        "Ложные срабатывания здесь ожидаемы: поиск идёт по словам, "
        "а слово «шквал» бывает и в переносном смысле.",
        "",
    ]
    for f in found[:60]:
        lines.append(f"- [ ] **{f['date']}** · {CAT_RU.get(f['cat'], f['cat'])} — "
                     f"[{f['title']}]({f['link']}) · _{f['src'] or 'источник не указан'}_")
    if len(found) > 60:
        lines.append(f"\n…и ещё {len(found) - 60} — сузьте период через SCAN_DAYS.")

    body = "\n".join(lines)
    title = f"Кандидаты в архив · {datetime.now(timezone.utc).strftime('%Y-%m-%d')} · {len(found)} шт."

    token = os.environ.get('GITHUB_TOKEN')
    repo = os.environ.get('GITHUB_REPOSITORY')
    if not token or not repo:
        print("\n--- GITHUB_TOKEN не задан, печатаю в консоль ---\n")
        print(title)
        print(body)
        return

    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/issues",
        data=json.dumps({'title': title, 'body': body,
                         'labels': ['кандидаты']}).encode(),
        headers={'Authorization': f'Bearer {token}',
                 'Accept': 'application/vnd.github+json',
                 'User-Agent': UA, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=30) as r:
        print("Issue создан:", json.loads(r.read())['html_url'])


if __name__ == '__main__':
    main()
