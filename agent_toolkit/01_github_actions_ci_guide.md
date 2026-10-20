# 01. Руководство для Агента: Управление GitHub Actions (`gh`) и мониторинг в реальном времени

Это подробная шпаргалка со всеми командами `git` и `gh`, которые позволяют ИИ-агенту использовать GitHub Actions как свой удалённый вычислительный узел.

---

## 1. Проверка статуса репозитория и текущей ветки

Всегда проверяй, на какой ветке ты находишься, и пушь только в неё:

```bash
# Узнать текущую ветку и статус файлов
git branch --show-current
git status -s
```

---

## 2. Запуск GitHub Actions (по пушу или вручную)

### Вариант А: Автоматический запуск через `git push`
Если в `.github/workflows/*.yml` настроен триггер:
```yaml
on:
  workflow_dispatch:
  push:
    branches:
      - "**"
```
то любой пуш без тега `[skip ci]` автоматически запускает раннер:

```bash
git add -A
git commit -m "Test new optimization in CI"
git push origin HEAD
```

### Вариант Б: Ручной запуск без коммита (`gh workflow run`)
Если нужно перезапустить workflow без новых изменений в коде:

```bash
gh workflow run fetch_baseline.yml --ref $(git branch --show-current)
```

---

## 3. Просмотр списка запусков и получение `RUN_ID`

Сразу после `git push` подожди `3–5 секунд`, чтобы GitHub зарегистрировал запуск, и запроси список последних ранов:

```bash
# Красивая таблица последних 5 запусков на текущей ветке
gh run list --branch $(git branch --show-current) --limit 5
```

Пример вывода:
```text
STATUS  TITLE                                     WORKFLOW                         BRANCH                             EVENT  ID           ELAPSED  AGE
*       Fetch Baseline and Inspect Data           Fetch Baseline and Inspect Data  arena/8f28ae4f-3d-dit-distillation push   39493718947  14s      less than a minute ago
✓       ci: record teacher weight stats [skip ci] Fetch Baseline and Inspect Data  arena/8f28ae4f-3d-dit-distillation push   39490636656  1m26s    18 minutes ago
```

Чтобы программно вытащить `RUN_ID` самого свежего запуска в переменную bash:

```bash
RUN_ID=$(gh run list --branch $(git branch --show-current) --limit 1 --json databaseId --jq '.[0].databaseId')
echo "Latest Run ID: $RUN_ID"
```

---

## 4. Живой мониторинг выполнения (`gh run watch`)

Команда `gh run watch` работает через `api.github.com` (она **не** обращается к заблокированному `results-receiver.actions.githubusercontent.com`), поэтому идеально работает в песочнице!

```bash
sleep 4
RUN_ID=$(gh run list --branch $(git branch --show-current) --limit 1 --json databaseId --jq '.[0].databaseId')
echo "Watching CI run $RUN_ID"
gh run watch "$RUN_ID" --exit-status
```

Что ты увидишь в терминале в реальном времени:
```text
Watching CI run 39493718947
Refreshing run status every 3 seconds. Press Ctrl+C to quit.

* arena/8f28ae4f-3d-dit-distillation Fetch Baseline and Inspect Data · 39493718947
Triggered via push less than a minute ago

JOBS
* verify (ID 118873211811)
  ✓ Set up job
  ✓ Checkout repository
  ✓ Set up Python
  * Install dependencies
  * Build and verify submission and run unit tests
  * Run teacher verification and solver benchmark
  * Commit and push results
```

Когда все шаги завершатся, `gh run watch` автоматически выйдет и покажет финальный статус:
```text
✓ arena/8f28ae4f-3d-dit-distillation Fetch Baseline and Inspect Data · 39493718947
Triggered via push about a minute ago

✓ Verify in 1m24s (ID 118873211811)
  ✓ Set up job
  ✓ Checkout repository
  ✓ Set up Python
  ✓ Install dependencies
  ✓ Build and verify submission and run unit tests
  ✓ Run teacher verification and solver benchmark
  ✓ Commit and push results
  ✓ Post Set up Python
  ✓ Post Checkout repository
  ✓ Complete job
```

*(Важно: в инструменте `bash` ставь `timeout: 300`, чтобы `gh run watch` спокойно дождался окончания установки пакетов и прогона тестов).*

---

## 5. Как получить полный лог выполнения (`fetched_info/ci_run.log`)

Почему `gh run view $RUN_ID --log` не работает в песочнице?
Потому что GitHub CLI при запросе `--log` делает редирект на `https://results-receiver.actions.githubusercontent.com/...` или `*.blob.core.windows.net`, который блокируется файрволом песочницы (возвращает `EOF`).

### Наше решение (работает безотказно на 100%):
1. Внутри `.github/workflows/fetch_baseline.yml` все команды пишут вывод в `fetched_info/ci_run.log`:
   ```bash
   python3 tools/build_submission.py 2>&1 | tee fetched_info/ci_run.log
   python3 tools/test_solution.py 2>&1 | tee -a fetched_info/ci_run.log
   python3 .github/scripts/verify_teacher_and_solvers.py 2>&1 | tee -a fetched_info/ci_run.log
   ```
2. Последний шаг workflow имеет условие `if: always()` (выполняется даже тогда, когда предыдущий шаг упал с ошибкой!) и пушит `fetched_info/ci_run.log` обратно в ветку с пометкой `[skip ci]`.
3. Сразу после завершения `gh run watch` агент в песочнице выполняет:
   ```bash
   git pull --rebase origin $(git branch --show-current)
   cat fetched_info/ci_run.log
   ```
   И видит **полный побайтовый traceback и stdout** всех тестов!

---

## 6. Проверка деталей шагов через JSON API (если нужно без `git pull`)

Если нужно узнать, на каком именно шаге упал job и его время работы прямо через API:

```bash
gh run view "$RUN_ID" --json status,conclusion,jobs --jq '.jobs[] | {name: .name, conclusion: .conclusion, steps: [.steps[] | {name: .name, conclusion: .conclusion}]}'
```
Эта команда тоже ходит только на `api.github.com` и никогда не блокируется.
