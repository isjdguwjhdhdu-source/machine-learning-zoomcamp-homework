# ML Zoomcamp 2026 — Homework 5: Deploying Machine Learning Models

Полное решение пятой домашней работы: загрузка готовой модели, предсказание регистрации клиента, HTTP API на FastAPI и запуск в Docker.

Ответы и результаты проверки находятся в [answers.md](answers.md), оригинальное условие — в [homework.md](homework.md).

## Состав проекта

- `q3.py` — решение Q3: загрузка модели и прогноз для первого клиента.
- `predict.py` — API курса: `GET /health` и `POST /predict`.
- `model.py` — подготовка признаков и обработка пропусков.
- `pipeline.bin` — готовый замороженный pipeline: DictVectorizer + LogisticRegression.
- `feature_defaults.json`, `model_metadata.json` — значения для заполнения пропусков и контрольные суммы.
- `examples/q3.json`, `examples/q4.json` — запросы из задания.
- `check_api.py` — проверка работающего API: health, контрольные суммы и ответ Q4/Q6.
- `q6_test.py`, `smoke_test.py` — исходные проверки курса.
- `Dockerfile`, `pyproject.toml`, `uv.lock`, `.python-version` — зафиксированное окружение курса.
- `train.py`, `data/course_lead_scoring_2026.csv` — исходный скрипт обучения и датасет; для сдачи используется уже готовый `pipeline.bin`.

## Подготовка

Нужен `uv`. Для контейнерной части также нужен работающий Docker Engine или Docker Desktop.
Все команды ниже выполняются из папки `05-deployment`.

```bash
uv --version
uv sync --locked
uv run python smoke_test.py
```

Ожидаемый вывод smoke-теста:

```text
{'conversion_probability': 0.769799, 'conversion': True}
```

В `uv.lock` зафиксированы Scikit-Learn 1.7.2 и NumPy 2.3.3. Файл `.python-version` задаёт Python 3.11.15. Не пересоздавайте lockfile для выполнения задания.

Проверка модели на macOS:

```bash
shasum -a 256 pipeline.bin
```

На Linux можно использовать `sha256sum pipeline.bin`. Ожидаемая сумма:

```text
1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1
```

## Q1 и Q2: версии окружения

```bash
uv --version
uv run python -c "import sklearn; print(sklearn.__version__)"
```

Q1 — локальная версия `uv`, этот вопрос не оценивается. Q2 — `1.7.2`.

## Q3: загрузка модели

```bash
uv run python q3.py
```

Ожидаемый результат:

```text
conversion_probability: 0.532994418643
Q3: 0.533
```

Скрипт проверяет контрольную сумму, загружает `pipeline.bin` через pickle, читает первого клиента из `examples/q3.json` и вызывает `predict_proba`.
Индекс `[0, 1]` выбирает вероятность класса 1 для первой строки. Это оценка вероятности регистрации клиента.

## Q4: локальный HTTP API

В первом терминале:

```bash
uv run uvicorn predict:app --host 127.0.0.1 --port 9696
```

Во втором терминале, из той же папки:

```bash
curl -sS http://127.0.0.1:9696/predict \
  -H 'Content-Type: application/json' \
  --data-binary @examples/q4.json
```

Ожидаемый ответ:

```json
{"conversion_probability": 0.769799, "conversion": true}
```

Ответ Q4: **0.770**. Поле `conversion` сравнивает исходную вероятность с порогом 0.5.

Проверка API:

```bash
uv run python check_api.py
```

Скрипт проверяет `/health`, обе контрольные суммы, допустимый диапазон вероятности, тип поля `conversion` и результат предсказания с допуском задания.

## Q5 и Q6: Docker

Первая строка `Dockerfile` задаёт базовый образ **python:3.11.15-slim-bookworm**, дополнительно закреплённый digest. Это ответ Q5.

Остановите локальный Uvicorn через Ctrl+C, чтобы освободить порт 9696. Затем:

```bash
docker build -t zoomcamp-model:2026-hw5 .
docker run --rm -p 127.0.0.1:9696:9696 zoomcamp-model:2026-hw5
```

Во втором терминале:

```bash
uv run python q6_test.py
uv run python check_api.py
curl -sS http://127.0.0.1:9696/health
```

Ожидаемый ответ Q6: **0.770**. Запрос и замороженная модель те же, что в Q4.
Health должен возвращать `status=ok`, `model_sha256` и `feature_defaults_sha256`, совпадающие с `model_metadata.json`.

Модель и HTTP API проверены в зафиксированном окружении. Сборка и запуск Docker-контейнера в среде подготовки архива не выполнялись, поскольку Docker отсутствовал.

## Дополнительно: обучение отдельной копии модели

Для Q1–Q6 переобучение не требуется. Исходный `train.py` оставлен без изменений; в отдельной папке проекта путь к датасету передаётся явно. Чтобы не заменить замороженную модель, сохраняйте эксперимент в `retrained`:

```bash
mkdir -p retrained
uv run python train.py \
  --data data/course_lead_scoring_2026.csv \
  --model retrained/pipeline.bin
```

Результаты повторного обучения могут иметь другую контрольную сумму. Для оцениваемой части задания используйте исходный `pipeline.bin` в корне проекта.

## Источники

Исходный проект: [DataTalksClub/machine-learning-zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp), каталог `cohorts/2026/homework/05-deployment`.

Решение подготовлено по предоставленному архиву курса, commit `5356b3b16e5c87f8f9fe0c3f8fdb2183d34b18fb`.
Модель, исходный API, скрипт обучения, Dockerfile и lockfile сохранены без изменений. Добавлены решение Q3, примеры запросов, проверка API и документация.

Форма сдачи: https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw05
