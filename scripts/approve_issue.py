#!/usr/bin/env python3
"""
Переносит одобренный Issue в архив.

Запускается автоматически, когда на Issue вешают метку «одобрено».
Решение принимает человек — скрипт только избавляет от ручного
редактирования JSON, где легко потерять запятую и сломать весь файл.

Читает поля формы из тела Issue, дописывает запись в data/archive.json,
коммитит и закрывает Issue с комментарием.
"""

import os
import re
import sys
import json
import urllib.request
from datetime import datetime, timezone

UA = "sakha-storm-log/1.0"

CAT_MAP = {
    'смерч': 'tornado', 'шквал': 'squall', 'град': 'hail',
    'гроза/ливень': 'thunderstorm', 'гроза': 'thunderstorm',
    'паводок': 'flood', 'сухая гроза': 'fire',
}
CRED_MAP = {'подтверждено': 'green', 'частично': 'yellow', 'требует проверки': 'red'}


def parse_issue_form(body):
    """
    Разбирает тело Issue, созданного по форме.

    GitHub раскладывает ответы как «### Заголовок поля» и следом текст.
    Незаполненные поля он помечает как _No response_.
    """
    fields, cur = {}, None
    for line in (body or '').split('\n'):
        m = re.match(r'^###\s+(.+?)\s*$', line)
        if m:
            cur = m.group(1).strip()
            fields[cur] = []
            continue
        if cur is not None:
            fields[cur].append(line)
    out = {}
    for k, v in fields.items():
        txt = '\n'.join(v).strip()
        out[k] = '' if txt in ('_No response_', '_Не указано_') else txt
    return out


def pick(fields, *names):
    for n in names:
        for k, v in fields.items():
            if n.lower() in k.lower():
                return v
    return ''


def main():
    token = os.environ['GITHUB_TOKEN']
    repo = os.environ['GITHUB_REPOSITORY']
    number = os.environ['ISSUE_NUMBER']
    body = os.environ.get('ISSUE_BODY', '')
    author = os.environ.get('ISSUE_AUTHOR', 'неизвестно')

    f = parse_issue_form(body)
    print("Разобранные поля:", json.dumps(f, ensure_ascii=False)[:500])

    date = pick(f, 'Дата').strip()
    if not re.match(r'^\d{4}-\d{2}-\d{2}$', date):
        fail(token, repo, number,
             f"Не понял дату «{date}». Нужен формат ГГГГ-ММ-ДД, например 2026-06-17. "
             "Поправьте в описании Issue и снимите/верните метку «одобрено».")

    cat_raw = pick(f, 'Тип явления').strip().lower()
    cat = next((v for k, v in CAT_MAP.items() if k in cat_raw), None)
    if not cat:
        fail(token, repo, number,
             f"Не понял тип явления «{cat_raw}». Допустимые: " +
             ", ".join(CAT_MAP.keys()))

    cred_raw = pick(f, 'Достоверность').strip().lower()
    cred = next((v for k, v in CRED_MAP.items() if k in cred_raw), 'red')

    approx = 'x' in pick(f, 'Точность даты').lower()

    loc = pick(f, 'Место').strip()
    detail = pick(f, 'Что произошло', 'Описание').strip()
    src_raw = pick(f, 'Источник').strip()
    sources = [x.strip() for x in re.split(r'[,;\n]', src_raw) if x.strip()]

    title = os.environ.get('ISSUE_TITLE', '').replace('[случай]', '').strip()
    if not title:
        title = detail.split('.')[0][:80] or f"{cat} в {loc}"

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, 'data', 'archive.json')
    with open(path, encoding='utf-8') as fh:
        data = json.load(fh)

    # Защита от повторного добавления: тот же день, тип и место
    for e in data['events']:
        if e['date'] == date and e['cat'] == cat and e['loc'].strip() == loc:
            fail(token, repo, number,
                 f"Такая запись уже есть в архиве (id `{e['id']}`). "
                 "Если это другой случай, уточните место в описании.")

    n = sum(1 for e in data['events'] if e['date'] == date) + 1
    rec = {
        "id": f"{date}-{cat}-{n:02d}",
        "date": date,
        "approx_date": approx,
        "cat": cat,
        "title": title,
        "loc": loc,
        "detail": detail,
        "sources": sources,
        "cred": cred,
        "added": datetime.now(timezone.utc).strftime('%Y-%m-%d'),
        "author": author,
        "issue": int(number),
    }
    data['events'].append(rec)
    data['events'].sort(key=lambda e: e['date'], reverse=True)
    data['updated'] = datetime.now(timezone.utc).strftime('%Y-%m-%d')

    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write('\n')

    print("Добавлена запись:", json.dumps(rec, ensure_ascii=False, indent=1))
    print(f"Всего записей: {len(data['events'])}")

    comment(token, repo, number,
            f"Добавлено в архив: `{rec['id']}`\n\n"
            f"- дата: {date}{' (примерная)' if approx else ''}\n"
            f"- тип: {cat}\n"
            f"- место: {loc}\n"
            f"- достоверность: {cred}\n"
            f"- источники: {', '.join(sources) or 'не указаны'}\n\n"
            "Сайт обновится через минуту. Спасибо!")
    close_issue(token, repo, number)


def api(token, url, payload=None, method=None):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode() if payload else None,
        headers={'Authorization': f'Bearer {token}',
                 'Accept': 'application/vnd.github+json',
                 'User-Agent': UA, 'Content-Type': 'application/json'},
        method=method)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read() or b'{}')


def comment(token, repo, number, text):
    api(token, f"https://api.github.com/repos/{repo}/issues/{number}/comments",
        {'body': text})


def close_issue(token, repo, number):
    api(token, f"https://api.github.com/repos/{repo}/issues/{number}",
        {'state': 'closed'}, method='PATCH')


def fail(token, repo, number, msg):
    """Пишет в Issue, что не так, и останавливает перенос."""
    print("ОТКЛОНЕНО:", msg, file=sys.stderr)
    try:
        comment(token, repo, number, f"Не удалось добавить в архив.\n\n{msg}")
        api(token, f"https://api.github.com/repos/{repo}/issues/{number}/labels/"
            + urllib.parse.quote('одобрено'), method='DELETE')
    except Exception as e:
        print("не смог прокомментировать:", e, file=sys.stderr)
    sys.exit(1)


if __name__ == '__main__':
    import urllib.parse
    main()
