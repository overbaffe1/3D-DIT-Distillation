# Источники цифр и границы доказательств

Этот файл отвечает на один вопрос: **откуда взялась каждая цифра в витрине и можно ли её повторить**. Портфолио-кейс намеренно разделяет три слоя: код, воспроизводимый CPU-протокол и архивные заявки площадки.

**Связанные файлы:** [кейс](CASE_STUDY.md) · [README](../README.md) · [техника](TECHNICAL.md) · [запуск](REPRODUCE.md)

---

## 1. Карта артефактов

| Файл | Что содержит | Статус |
| :--- | :--- | :--- |
| [`../fetched_info/solver_experiments.json`](../fetched_info/solver_experiments.json) | equivalence + сравнение солверов | **воспроизводимый протокол** |
| [`../fetched_info/ci_run.log`](../fetched_info/ci_run.log) | формат посылки, тесты, AOT PTX, замеры | **журнал конкретного CI-прогона** |
| [`../.github/scripts/verify_teacher_and_solvers.py`](../.github/scripts/verify_teacher_and_solvers.py) | код, который породил JSON | исходник протокола |
| [`../tools/test_solution.py`](../tools/test_solution.py) | numerical equivalence + AOT-проверка | тест |
| [`../lib/solver_config.json`](../lib/solver_config.json) | текущий солвер: `4` шага, `3+1`, `neg_reuse` | конфигурация сдачи |
| [`../qaoa_state.json`](../qaoa_state.json) | generation `153`, внутренние метрики оптимизатора | **внутренний state**, не бенчмарк |
| [`../fetched_info/report.json`](../fetched_info/report.json) | описание выданных архивов: `4 974` записей, `500` файлов `public_test` | метаданные датасета |
| [`../fetched_info/weights_stats.json`](../fetched_info/weights_stats.json) | статистика `13` тензоров `flow.blocks.0.*` | частичная выгрузка, см. warning в CI-log |
| [`../submission.zip`](../submission.zip), [`../submission_final_q015.zip`](../submission_final_q015.zip) | проверенные архивы сдачи | артефакт, не метрика |

---

## 2. Воспроизводимый CPU-стенд

[`verify_teacher_and_solvers.py`](../.github/scripts/verify_teacher_and_solvers.py) собирает **уменьшенную 2-блочную модель** (`depth = 2`, `cond_depth = 1`). Реальные веса `flow.blocks.0.*` читаются из удалённого ZIP по HTTP Range; остальные параметры инициализируются фиксированным seed. Используется один сэмпл из `fetched_info/sample_test_data/public_test`:

- equivalence: первые `384` вокселя и полные `4 101` DINO-токенов;
- solver benchmark: `128` вокселей и `69` cond-токенов;
- устройство: CPU, без A100;
- учитель: `50` шагов Euler внутри той же уменьшенной модели.

Именно поэтому `10,07×` в обложке подписан как **ускорение одной безусловной ветви**, а `2,70×` / `3,27×` в солверной таблице — как **относительная динамика на стенде**, а не финальный competition score.

### Ключевые значения

| Метрика | Значение | Где в JSON |
| :--- | :--- | :--- |
| `pos_pred` max abs diff | `9,5367431640625e-07` | `equivalence.pos_max_abs_diff` |
| `pos_pred` cosine | `1,000000238418579` | `equivalence.pos_cosine_sim` |
| `neg_pred` max abs diff | `3,2186508178710938e-06` | `equivalence.neg_1token_max_abs_diff` |
| `neg_pred` cosine | `1.0` | `equivalence.neg_1token_cosine_sim` |
| `neg_pred` reference | `2,993171174 s` | `equivalence.ref_neg_sec` |
| `neg_pred` 1-token | `0,297149551 s` | `equivalence.fast_neg_1token_sec` |
| branch speedup | `10,072945303×` | `equivalence.neg_speedup` |
| teacher trajectory | `1,357896891 s` | `teacher_time_sec` |

Полный список солверов: [`solver_experiments.json`](https://github.com/overbaffe1/3D-DIT-Distillation/blob/d65823d79066eb6e67a722d8a3db2f4153739ac7/fetched_info/solver_experiments.json#L2-L12) · краткая распечатка в [`ci_run.log`](../fetched_info/ci_run.log).

### Что именно проверяет CI

1. `tools/build_submission.py` — корень и `submission.zip` соответствуют `submission_format.py`; `weights/` может отсутствовать, тогда оценка берёт веса организатора.
2. `tools/test_solution.py` — `pos_pred` / `neg_pred` equivalence, контракт `generate()`, warmup.
3. AOT-компиляция шести `@triton.jit` ядер в PTX для `sm_80`: `_encode_hilbert_4_kernel`, `_fused_gate_add_ln_modulate_kernel`, `_fused_out_modulate_unpatchify_kernel`, `_fused_packed_swiglu_kernel`, `_fused_qknorm_rope3d_gather_kernel`, `_fused_qknorm_rope2d_cond_inplace_kernel`.
4. `verify_teacher_and_solvers.py` — описанный выше CPU-стенд.

В журнале есть `ConnectionResetError` при RemoteZip: часть статистики весов может быть неполной. На equivalence-тест это не влияет, потому что он сравнивает reference и fast path на уже загруженных тензорах.

---

## 3. Формулы оценки хакатона

| Скрипт | Время | Порог качества | Итог на сэмпл |
| :--- | :--- | :--- | :--- |
| [`score.py`](../baseline_repo/3dgen-main/scripts/hackathon/score.py) | `t_gt / t_pred` | `q_min = 0.25`, `res = 512` | `quality × t_gt / t_pred` |
| [`score_final.py`](../score_final.py) | `1 + log₂(t_gt / t_pred)` | `q_min = 0.15`, `res = 1024` | `quality × (1 + log₂(S))` |

В обоих случаях `quality = 0`, если LPIPS выше порога, иначе `1 − LPIPS`; усредняется по всем GT-сэмплам, отсутствующие/упавшие предсказания дают нулевой вклад. LPIPS считается по четырём рендерам `512×512` (в финальном скрипте `res = 1024`).

---

## 4. Архивный слой: что не используется как доказательство

| Источник | Что там написано | Почему вынесено отдельно |
| :--- | :--- | :--- |
| [PR #1](https://github.com/overbaffe1/3D-DIT-Distillation/pull/1) и исходный README | публичные баллы `15.5618`, `56.7039`, `116.0542`, `>125×` | в ветке нет логов площадки; числа сохранены как историческая заявка |
| [`../qaoa_state.json`](../qaoa_state.json) | `best_metrics.public_score = 1629.83`, `cos_sim = 1`, `eff_nfe ≈ 0.054` | внутренняя метрика оптимизатора, не независимый прогон `score.py` |
| комментарии и старые бейджи | «#1 among honest GPU inference solutions» | утверждение о месте не проверяется репозиторием и в новом оформлении не повторяется |

В [`../README.md`](../README.md) архивные баллы спрятаны в `<details>` с этой оговоркой. В [`CASE_STUDY.md`](CASE_STUDY.md) и в SVG-схемах используются только величины из `solver_experiments.json` и `lib/solver_config.json`.

**Факт участия в хакатоне** подтверждается контрактом `LatDitGenerator`, participant kit в `baseline_repo/`, `submission*.zip` и CI-протоколом. Призовое место или ранг в этом описании не заявляются.

---

## 5. Происхождение кода и данных

- `baseline_repo/3dgen-main/` — slim extract кода организаторов; в выжимке нет файла лицензии. Она нужна только как эталон протокола и не выдаётся за авторский код проекта.
- Полный архив весов `kandi3d_weights_aijc_kit.zip` и `public_test` в Git не хранятся. [`../tools/download_data.py`](../tools/download_data.py) и workflow [`../.github/workflows/fetch_baseline.yml`](../.github/workflows/fetch_baseline.yml) описывают загрузку.
- Один сэмпл в `fetched_info/sample_test_data/` оставлен для офлайн smoke-тестов.
- Все SVG в [`assets/portfolio/`](assets/portfolio/) и [`manifest.json`](assets/portfolio/manifest.json) генерятся [`../tools/build_portfolio_assets.py`](../tools/build_portfolio_assets.py) из двух JSON-файлов выше.

---

## 6. Как перепроверить

```bash
# только стандартная библиотека: ссылки, XML/SVG, локальные ассеты страницы
make docs-check

# пересобрать схемы и social preview из текущих JSON
make docs-assets

# полный тест требует torch/triton и внешних данных
python3 tools/test_solution.py
python3 .github/scripts/verify_teacher_and_solvers.py
```

Если внешний ZIP с весами недоступен, `verify_teacher_and_solvers.py` всё равно завершит сравнение на инициализированной модели, но `weights_stats.json` будет неполным — это тот самый warning из CI-log.

---

[← Кейс](CASE_STUDY.md) · [Техническое устройство →](TECHNICAL.md)
