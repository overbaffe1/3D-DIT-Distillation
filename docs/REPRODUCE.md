# Запуск и воспроизведение

Документ отделяет **лёгкую проверку оформления** (без GPU и внешних данных) от **полного хакатонского прогона** (нужны веса, датасет и CUDA-стек).

**См. также:** [README](../README.md) · [кейс](CASE_STUDY.md) · [источники цифр](EVIDENCE.md) · [техника](TECHNICAL.md)

---

## 0. Что можно проверить сразу

```bash
make docs-check     # ссылки, XML/SVG, локальные ассеты HTML, отсутствие localhost в разметке
make docs-assets    # пересобрать SVG + social-preview.png из JSON-протоколов
make docs-serve     # статическая витрина на 0.0.0.0:4173
```

`docs-serve` поднимает `python3 -m http.server` от корня репозитория. Локально откройте `http://localhost:4173/docs/index.html`; в Arena тот же сервер доступен через автоматический preview-домен. Страница использует только относительные пути к `docs/assets/`.

Для `docs-check` и `docs-assets` достаточно Python 3.10+ без сторонних пакетов. `social-preview.png` экспортируется через ImageMagick, если он установлен; иначе остаётся переносимый [`assets/portfolio/social-preview.svg`](assets/portfolio/social-preview.svg).

---

## 1. Окружение

Базовый список зависимостей лежит в [`../baseline_repo/3dgen-main/requirements.txt`](../baseline_repo/3dgen-main/requirements.txt). CI дополнительно ставит:

```bash
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install remotezip requests numpy safetensors einops triton transformers diffusers omegaconf trimesh pandas
```

Для полного eval на GPU дополнительно нужны CUDA-расширения из participant kit: `flash-attn`, `spconv`, `nvdiffrast`, `torch_scatter`, `libsparse`/`curope`/`libmesh` и совместимая сборка PyTorch. Без них запускаются только тесты и CPU-стенд.

> Ветка не содержит полный архив `kandi3d_weights_aijc_kit.zip` (~7,80 GB) и весь `public_test`. В Git лежат конфиги, один smoke-сэмпл и загрузчики.

---

## 2. Данные

```bash
# веса teacher + lat_vae; флаг скопирует lat_dit-веса в ./weights/ для автономной посылки
python3 tools/download_data.py \
  --target weights \
  --out-dir ./data \
  --populate-submission-weights

# публичный тестовый сплит
python3 tools/download_data.py \
  --target public_test \
  --out-dir ./data/public_test

# опционально: train-чанки
python3 tools/download_data.py \
  --target train --train-chunks 1 2 \
  --out-dir ./data/train
```

Ожидаемая раскладка после загрузки:

```text
data/
├── kandi3d_weights_aijc_kit/
│   ├── lat_dit/   # diffusion_pytorch_model.safetensors + config.json
│   └── lat_vae/
└── public_test/
    └── <id>/cache/
        ├── vox_idxi.npz
        └── 4view_000_dinov3.npz
```

`data/` и `eval_runs/` добавлены в `.gitignore`.

---

## 3. Лёгкая посылка и тесты

```bash
# архив без гигабайтных весов: оценка возьмёт --lat-dit-dir организатора
python3 tools/build_submission.py --output submission.zip

# или собрать автономную посылку с весами
python3 tools/build_submission.py \
  --lat-dit-dir ./data/kandi3d_weights_aijc_kit/lat_dit \
  --output submission_full.zip

# формат, контракт, numerical equivalence, AOT PTX sm_80
python3 tools/test_solution.py
```

`submission_format.py` допускает `solution.py`, `weights/`, `lib/` и запрещает корневые `kandi3d_image23d/`, `scripts/`. Отсутствие непустого `weights/` — не ошибка, а режим «использовать веса организатора».

---

## 4. Калибровка / дистилляция (опционально)

```bash
python3 tools/train_distill.py \
  --cache-dir ./data/public_test \
  --lat-dit-dir ./data/kandi3d_weights_aijc_kit/lat_dit \
  --out-weights-dir ./weights \
  --mode calibrate \
  --student-steps 16 \
  --cfg-stride 2 \
  --num-samples 32 \
  --epochs 300 \
  --lr 1e-2
```

`--mode lora_distill` сохраняет `student_adapter.pt`; [`../solution.py`](../solution.py) вливает адаптер в базовые веса в `from_pretrained()`, поэтому во время `generate()` нет runtime-надбавки LoRA. Текущий [`../lib/solver_config.json`](../lib/solver_config.json) уже содержит настроенные `4` шага; обучение нужно только для эксперимента с другой траекторией.

---

## 5. GT, student и score

Используйте `--out-dir` (не `--output-dir`) — именно так называется аргумент в `run_lat_dit_eval.py`.

```bash
# 1. Учитель / GT
python3 baseline_repo/3dgen-main/scripts/hackathon/run_lat_dit_eval.py \
  --cache-dir ./data/public_test \
  --out-dir ./eval_runs/gt \
  --lat-dit-dir ./data/kandi3d_weights_aijc_kit/lat_dit \
  --lat-vae-dir ./data/kandi3d_weights_aijc_kit/lat_vae

# 2. Наше решение как unpacked submission
python3 baseline_repo/3dgen-main/scripts/hackathon/run_lat_dit_eval.py \
  --cache-dir ./data/public_test \
  --out-dir ./eval_runs/student \
  --submission-dir . \
  --lat-dit-dir ./data/kandi3d_weights_aijc_kit/lat_dit \
  --lat-vae-dir ./data/kandi3d_weights_aijc_kit/lat_vae

# 3. Публичный score.py: quality × t_gt/t_pred, q_min=0.25
python3 baseline_repo/3dgen-main/scripts/hackathon/score.py \
  --gt-csv ./eval_runs/gt/results_summary.csv \
  --pred-csv ./eval_runs/student/results_summary.csv \
  --out-dir ./eval_runs/score

# 4. Финальный score_final.py: quality × (1 + log2(S)), q_min=0.15
PYTHONPATH=baseline_repo/3dgen-main:baseline_repo/3dgen-main/scripts/hackathon \
python3 score_final.py \
  --gt-csv ./eval_runs/gt/results_summary.csv \
  --pred-csv ./eval_runs/student/results_summary.csv \
  --out-dir ./eval_runs/score_final
```

`score_final.py` импортирует `_bootstrap` и `rendering` из participant kit, поэтому для него нужен `PYTHONPATH`. Оба скрипта пишут `summary.csv`, `Quality_factor.txt`, `Time_factor.txt`, `Total_factor.txt` в `--out-dir`.

---

## 6. CPU-протокол из CI

```bash
# создаёт/обновляет fetched_info/solver_experiments.json и weights_stats.json
python3 .github/scripts/verify_teacher_and_solvers.py 2>&1 | tee fetched_info/ci_run.log

# только проверка решений без пересбора статистики
python3 tools/test_solution.py
```

Скрипт читает реальные `flow.blocks.0.*` из удалённого ZIP по HTTP Range. Если сеть обрывается, в логе остаётся `ConnectionResetError`, а equivalence-сравнение всё равно выполняется на уже загруженных/инициализированных весах. Подробности и оговорки — в [EVIDENCE.md](EVIDENCE.md).

---

## 7. GitHub Pages / CDN

`docs/index.html` самодостаточен: CSS и SVG лежат рядом, внешних шрифтов и JS нет. Варианты публикации:

- GitHub Pages: source = `main`, folder = `/docs` (в репозитории есть `docs/.nojekyll`);
- CDN для быстрой демонстрации: `https://rawcdn.githack.com/overbaffe1/3D-DIT-Distillation/main/docs/index.html`;
- локально: `make docs-serve`.

Для превью репозитория подготовлен [`assets/portfolio/social-preview.png`](assets/portfolio/social-preview.png) — загрузите его в **Settings → Social preview**.

---

[← Техническое устройство](TECHNICAL.md) · [Наверх в README](../README.md)
