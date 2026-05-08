# Landmark Recognition – Data‑Centric Pipeline

Работа посвящена задаче определения достопримечательностей по фотографиям, взят за основу датасет 
Google Landmarks Dataset v2 с соревнования Kaggle Landmark Recognition 2021.
Основная идея — проверить на практике, насколько data‑centric подход эффективнее бесконечной погони 
за сложными архитектурами. Мы последовательно очищаем данные, учим компактные эмбеддинги с помощью 
ArcFace/PartialArcFace и GeM‑пулинга, ищем ближайшие эталоны через FAISS и на каждом шаге замеряем, 
как это улучшает GAP@1.

## Структура репозитория

### Датасет
- `train.csv` – исходная разметка тренировочной выборки (id, landmark_id)
- `sample_submission.csv` – файл для предсказаний на соревновании
- `train_label_to_category.csv` – landmark_id → читаемое название (Wikimedia URL)

### Places365 (классификация indoor/outdoor сцен)
- `categories_places365.txt` – список 365 категорий сцен
- `categories_places365_extended.csv` – расширенная аннотация категорий (включая indoor/outdoor‑метки)
- `IO_places365.txt` – бинарные метки indoor (1) / outdoor (2) для категорий

### Train после фильтраций
- `train_after_size_filter.csv` – после удаления изображений с минимальной стороной <128 px
- `train_after_dup_removal.csv` – после удаления дубликатов (phash + связные компоненты)
- `train_final_landmark.csv` – финальный очищенный тренировочный набор (после Places365 + YOLOv5)

### Train/val/test
- `train_split.csv` – обучающая выборка (без дистракторов)
- `val_split.csv` – валидационная выборка (с дистракторами, помеченными `landmark_id = -1`)
- `val_gt.csv` – ground truth для валидации (только релевантные запросы)
- `test_landmark_clean.csv` – очищенная тестовая выборка (после фильтрации Places365 + YOLOv5)

### Ноутбуки
- `1_eda.ipynb` – разведочный анализ (распределение классов, размеры, цветность, размытость)
- `2_data_preparation.ipynb` – полный цикл очистки: фильтрация размеров, дубликатов, Places365, YOLOv5, создание сплитов
- `3_training_baseline.ipynb` – baseline‑модель (ResNet18 + Softmax) на сырых данных
- `4_intermediate_model.ipynb`– улучшенная модель (EfficientNet‑B3 + PartialArcFace) на сырых данных
- `5_training_improved.ipynb` – улучшенная модель (EfficientNet‑B3 + PartialArcFace) на очищенных данных

### Прочее
- `README.md` – текущий файл с описанием репозитория
- `first_prediction.ipynb` – первоначальный запуск baseline‑модели
- `second_prediction.ipynb` – результаты улучшенной модели
- `result.ipynb` – финальная версия пайплайна

## Основные результаты

| Конфигурация | Данные | GAP@1 (val) |
|-------------|--------|-------------|
| Baseline (ResNet18 + Softmax) | сырые, топ-10k классов | 0.705 (GAP@5) |
| Промежуточная (EfficientNet‑B3 + ArcFace) | сырые, 81k классов | 0.425 |
| **Финальная (EfficientNet‑B3 + PartialArcFace + очищенные данные)** | **~60k классов** | **0.865** |


## Ссылки

- [Соревнование Kaggle Landmark Recognition 2021](https://www.kaggle.com/competitions/landmark-recognition-2021)
- [Google Landmarks Dataset v2 (GLDv2)](https://github.com/cvdfoundation/google-landmark)
- [Places365](https://github.com/CSAILVision/places365)
- [ArcFace (Deng et al., CVPR 2019)](https://arxiv.org/abs/1901.090109)
- [FAISS](https://github.com/facebookresearch/faiss)

## Лицензия

Проект выполнен в учебных целях. 
