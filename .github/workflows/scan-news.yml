name: Поиск кандидатов в архив

on:
  workflow_dispatch:
    inputs:
      days:
        description: 'За сколько дней искать'
        required: false
        default: '14'
  schedule:
    - cron: '0 6 * * 1'

permissions:
  issues: write
  contents: read

jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Проверить файлы
        run: |
          ls -la
          test -f scripts/scan_news.py && echo "OK" || (echo "нет scripts/scan_news.py"; exit 1)

      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Искать новости
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          GITHUB_REPOSITORY: ${{ github.repository }}
          SCAN_DAYS: ${{ github.event.inputs.days || '14' }}
        run: python scripts/scan_news.py
