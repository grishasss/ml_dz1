"""Create the readable experiment notebook; execution is a separate step."""
from pathlib import Path
import sys
import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md(r"""
# Домашнее задание 1: персональные рекомендации музыкальных инструментов

**ФИО студента:** Воронин Григорий Алексеевич.

**Группа:** Б05-326в.

**Датасет:** Amazon Review Data (2018), **Musical Instruments, 5-core**.
[Первоисточник и описание полей](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/index.html).
Рекомендуем товары: инструменты, аксессуары и музыкальное оборудование. Отзыв — наблюдаемое
взаимодействие, оценка 4 или 5 — положительный сигнал. Релевантность на validation/test для
**всех** моделей: ещё не известный пользователю товар с будущей оценкой ≥ 4.

Реализованы ItemKNN, iALS, EASE, FM с метаданными и ALS с градуированным таргетом.
Все обучение, scoring, top-10, негативный сэмплинг и метрики реализованы в репозитории.
NumPy/SciPy используются для линейной алгебры; PyTorch — для autograd и Adam внутри FM.

## Загрузка и запуск

Python 3.13, CPU; точные версии — в `requirements.txt`. Из корня репозитория:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/run_notebook.py
```
Либо открыть `experiments.ipynb` и выполнить **Restart Kernel and Run All**.
Первая загрузка автоматически скачивает **два gzip-архива с сервера UCSD** в `cache/raw/`.
SHA256 сверяются с закреплёнными значениями в коде; данные и модели в GitHub не входят.
Ноутбук каждый раз заново обучает модели; готовые метрики из `results/` для вычислений не читает.
Оценка EASE требует квадратичной памяти: ориентир 16 ГБ RAM. Точное время зависит от CPU/BLAS.
""")
code(r"""
import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['VECLIB_MAXIMUM_THREADS'] = '2'
import json, random, platform
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from threadpoolctl import threadpool_limits, threadpool_info
from IPython.display import display, Markdown
from recsys.data import load_data, build_protocol
from recsys.experiment import select_hyperparameters, final_comparison, save_results
from models.item_knn import ItemKNN
from models.ials import IALS
from models.ease import EASE
from models.fm import FM
from models.custom_als import CustomALS

CONFIG = json.loads(Path('config.json').read_text())
SEED = CONFIG['seed']
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(CONFIG['threads'])
torch.use_deterministic_algorithms(True)
THREAD_LIMIT = threadpool_limits(limits=CONFIG['threads'])
sns.set_theme(style='whitegrid', context='notebook')
plt.rcParams.update({'figure.dpi': 110, 'figure.figsize': (10, 4)})
print('Python:', platform.python_version(), '| PyTorch:', torch.__version__, '| seed:', SEED)
display(pd.DataFrame(threadpool_info())[['internal_api', 'num_threads', 'version']])
""")
md(r"""
## 1. Данные и смысл полей

Источник: Jianmo Ni, Jiacheng Li, Julian McAuley, *Justifying recommendations using
distantly-labeled reviews and fine-grained aspects*, EMNLP 2019.
На странице источника выбираются **Musical Instruments / 5-core** и соответствующий файл **metadata**.
Полная категория у источника содержит 1 512 530 отзывов; используемый авторский 5-core — 231 392.
5-core сформирован авторами: каждый пользователь и товар имел не менее пяти отзывов до нашей очистки.
Не делаем дополнительную случайную подвыборку и не отбираем популярные товары по test.

| Исходное поле | Использование |
|---|---|
| `reviewerID`, `asin` | ID пользователя и товара |
| `overall` | Оценка 1…5; обучение и общий критерий релевантности |
| `unixReviewTime` | Дата отзыва, глобальное временное разбиение |
| `title`, `category` из metadata | Название и категории товара, признаки FM |
| `brand` | Анализ полноты метаданных, в FM не включён |

Не используем будущий текст отзыва, `vote`, текущие цены, sales rank или графы `also_buy`.
Ни оценка, ни дата будущего отзыва не подаются в FM при рекомендации.
""")
code(r"""
reviews, metadata, data_audit = load_data('cache')
assert 100_000 <= len(reviews) <= 2_000_000
assert reviews.user_id.nunique() >= 1_000 and reviews.item_id.nunique() >= 1_000
display(pd.DataFrame([
    {'stage': 'source reviews', 'interactions': data_audit['raw_rows']},
    {'stage': 'invalid rows removed', 'interactions': data_audit['invalid_rows']},
    {'stage': 'duplicate user-item rows removed', 'interactions': data_audit['duplicate_user_item_rows']},
    {'stage': 'clean unique interactions', 'interactions': len(reviews)},
]))
display(pd.Series({k: data_audit[k] for k in ['n_users', 'n_items', 'metadata_source_rows',
    'metadata_matched_items', 'metadata_missing_items']}).to_frame('value'))
display(pd.DataFrame(data_audit['files']).T)
print('Period:', pd.to_datetime(reviews.timestamp.min(), unit='s', utc=True), '—',
      pd.to_datetime(reviews.timestamp.max(), unit='s', utc=True))
""")
md(r"""
## 2. EDA и подготовка

Удаляем строки без ID, даты или корректной оценки 1…5. Повторные отзывы на одну пару
user–item заменяем **самым ранним**: поздний отзыв не переписывает обучающий таргет задним числом.
При совпадении даты сохраняем первую строку исходного файла. Это также запрещает попадание
одного товара пользователя в несколько частей. Пропуски метаданных не удаляют взаимодействия:
FM получает отдельный missing-токен. После дедупликации строгий 5-core может нарушиться;
повторный k-core по всей истории не строим, чтобы не усиливать зависимость отбора от будущего.
""")
code(r"""
display(pd.Series(data_audit['missing_fields'], name='missing review fields').to_frame())
display(pd.Series({k: data_audit[k] for k in ['malformed_rows', 'invalid_rows',
    'malformed_metadata_rows', 'missing_title_items', 'missing_category_items', 'missing_brand_items']},
    name='count').to_frame())
activity = reviews.groupby('user_id').size()
popularity = reviews.groupby('item_id').size()
display(pd.DataFrame({'user activity': activity.describe(), 'item popularity': popularity.describe()}))
print(f'Overall sparsity: {1 - len(reviews) / (len(activity) * len(popularity)):.8%}')
fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
axes[0].hist(activity, bins=np.geomspace(1, activity.max() + 1, 35))
axes[0].set(xscale='log', yscale='log', title='Reviews per user', xlabel='Reviews', ylabel='Users')
axes[1].hist(popularity, bins=np.geomspace(1, popularity.max() + 1, 35))
axes[1].set(xscale='log', yscale='log', title='Reviews per item', xlabel='Reviews', ylabel='Items')
reviews.rating.value_counts().sort_index().plot.bar(ax=axes[2], color='#3b82a0')
axes[2].set(title='Observed ratings', xlabel='Rating', ylabel='Reviews')
plt.tight_layout(); plt.show()
""")
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
monthly = reviews.assign(month=pd.to_datetime(reviews.timestamp, unit='s', utc=True).dt.strftime('%Y-%m'))
monthly.groupby('month').size().plot(ax=axes[0], color='#3b82a0')
axes[0].set(title='Interactions over time', xlabel='Month', ylabel='Reviews')
categories = metadata.category.explode().dropna().value_counts().head(12).sort_values()
categories.plot.barh(ax=axes[1], color='#d78b47')
axes[1].set(title='Most frequent metadata categories', xlabel='Items', ylabel='Category')
plt.tight_layout(); plt.show()
display(pd.DataFrame({
    'field': ['title', 'category', 'brand'],
    'present_items': [metadata.title.ne('').sum(), metadata.category.map(len).gt(0).sum(),
                      metadata.brand.ne('').sum()],
    'share_of_clean_items': [metadata.title.ne('').sum()/data_audit['n_items'],
                             metadata.category.map(len).gt(0).sum()/data_audit['n_items'],
                             metadata.brand.ne('').sum()/data_audit['n_items']]
}))
""")
md(r"""
## 3. Протокол оценки и метрики, выбранные до подбора

Основная метрика **NDCG@10**: правильные рекомендации на верхних позициях полезнее.
Вторая метрика **Recall@10**: доля релевантных будущих товаров, найденных в десяти рекомендациях.
Дополнительно показываем HitRate@10 и Coverage@10. Усреднение — по пользователям, а не отзывам.

Для пользователя $u$ с множеством релевантных товаров $R_u$:
$$Recall@10(u)=\frac{|Top10_u\cap R_u|}{|R_u|},\quad
NDCG@10(u)=\frac{\sum_{j=1}^{10}\mathbf1[i_j\in R_u]/\log_2(j+1)}
{\sum_{j=1}^{\min(10,|R_u|)}1/\log_2(j+1)}.$$

Глобальные границы времени — 80-й и 90-й процентили очищенных timestamp, `method='nearest'`.
Все отзывы одной даты находятся с одной стороны границы. Train: $t<t_v$;
validation: $t_v\le t<t_s$; test: $t\ge t_s$. История будущих пользователей не попадает в train.

Словари строим **только по train**, каталог фиксируем для всего эксперимента. Кандидаты — все
товары train за вычетом всей известной истории пользователя. Сэмплирования кандидатов нет.
Новые пользователи/товары вне словарей исключаются из warm-start оценки с отдельным отчётом
о потерях. Пользователь участвует, если у него есть хотя бы один допустимый положительный
товар в соответствующей части; набор пользователей каждой части одинаков для всех моделей.
Validation и test могут включать разные наборы пользователей: это разные временные периоды.

Настройки выбираем исключительно на validation по NDCG@10; при равенстве — первый элемент сетки.
Затем обучаем заново на train+validation с тем же train-каталогом и один раз оцениваем test.
На test исключаем всё известное из train+validation, включая низкие оценки.
Критерий релевантности для пятой модели остаётся `rating >= 4`.
""")
code(r"""
assert CONFIG['k'] == 10 and CONFIG['rating_threshold'] == 4
assert CONFIG['split_quantiles'] == [0.8, 0.9]
protocol = build_protocol(reviews, metadata,
    train_fraction=CONFIG['split_quantiles'][0],
    validation_fraction=CONFIG['split_quantiles'][1]-CONFIG['split_quantiles'][0])
assert protocol.train.timestamp.max() < protocol.validation.timestamp.min()
assert protocol.validation.timestamp.max() < protocol.test.timestamp.min()
assert protocol.val_relevance.multiply(protocol.seen_train).nnz == 0
assert protocol.test_relevance.multiply(protocol.seen_final).nnz == 0
assert 100_000 <= protocol.audit['retained_rows_all_splits'] <= 2_000_000
print('Validation starts:', pd.to_datetime(protocol.audit['validation_cutoff_timestamp'], unit='s', utc=True))
print('Test starts:', pd.to_datetime(protocol.audit['test_cutoff_timestamp'], unit='s', utc=True))
print('Fixed train users/items:', protocol.n_users, protocol.n_items)
print('Train sparsity:', protocol.audit['train_sparsity'])
display(pd.DataFrame(protocol.audit['splits']).T)
print('Max EASE float64 matrix size, GiB:', round(protocol.n_items**2 * 8 / 2**30, 2))
""")
md(r"""
## 4. Пять моделей и основные формулы

Пусть $O$ — наблюдаемые отзывы, $r_{ui}\in\{1,\ldots,5\}$,
$A_{ui}=\mathbf1[(u,i)\in O\land r_{ui}\ge4]$. Неизвестное взаимодействие не означает
наблюдаемый негатив, но в iALS служит слабым нулевым сигналом.

### ItemKNN — `models/item_knn.py`

Бинарная item-based коллаборативная фильтрация. $n_{ij}=\sum_u A_{ui}A_{uj}$,
$n_i=\sum_u A_{ui}$. Сходство — cosine с подавлением ненадёжных совпадений:
$$s_{ij}=\frac{n_{ij}}{\sqrt{n_i n_j}}\frac{n_{ij}}{n_{ij}+\gamma},\quad s_{ii}=0.$$
Оставляем $L$ лучших соседей каждого исходного товара; оценка $\hat R=AS$.
Нулевые сходства и совпадающие scores разрешаются по возрастанию ASIN.

### iALS — `models/ials.py`

Для отзывов $p_{ui}=\mathbf1[r_{ui}\ge4]$, для остальных пар $p_{ui}=0$.
$c_{ui}=1+\alpha\mathbf1[(u,i)\in O]$ — наблюдаемая низкая оценка получает уверенный ноль.
$$\min_{X,Y}\sum_{ui}c_{ui}(p_{ui}-x_u^T y_i)^2+
\lambda(\|X\|_F^2+\|Y\|_F^2).$$
Точное попеременное решение нормальных уравнений:
$$x_u=(Y^TY+\alpha Y_{O_u}^TY_{O_u}+\lambda I)^{-1}
       (1+\alpha)Y_{O_u}^Tp_{O_u};$$
для $Y$ симметрично. Инверсию для обновления факторов не вычисляем: решаем SPD-систему.

### EASE — `models/ease.py`

$$\min_B\|A-AB\|_F^2+\lambda\|B\|_F^2,\quad diag(B)=0;\quad
P=(A^TA+\lambda I)^{-1},\quad B_{ij}=-P_{ij}/P_{jj},\ B_{jj}=0.$$
Scoring $\hat R=AB$. Реализовано через Cholesky; ограничение нулевой диагонали запрещает
тривиальное копирование входа. Память $O(|I|^2)$, разложение $O(|I|^3)$.

### FM с метаданными — `models/fm.py`

Признаки: one-hot user ID, one-hot item ID, категории и слова названия товара.
Категории передают тип/назначение оборудования; названия — тип, производителя и совместимость.
Для каждого из двух полей: до 16 различных токенов, стабильный SHA256 hashing в 512 отдельных
корзин; веса нормированы так, чтобы квадрат нормы поля был равен 1. Отсутствие — missing-токен.
Категории и названия считаем статическими свойствами карточки, известными при размещении товара.
Словари user/item обучены на train; хеширование не обучается на validation/test.
$$f(z)=w^Tz+\frac12\sum_f[(\sum_jv_{jf}z_j)^2-\sum_jv_{jf}^2z_j^2].$$
Сами поля не добавляются как независимый обучаемый алгоритм: реализована полная FM второго
порядка. Обучение BPR: $\sum_{(u,i^+,i^-)}softplus[-(f(u,i^+)-f(u,i^-))]$ с L2.
Положительные — известные оценки ≥4; по одному равномерному неизвестному негативу на положительный
за эпоху, исключая **все** наблюдаемые отзывы. Сэмплер видит только обучающую историю своей фазы.
Adam из PyTorch обновляет вручную заданную FM; число эпох выбирается по validation.

### CustomALS — `models/custom_als.py`

Разработанный таргет — степень удовлетворённости:
$$p^*_{ui}=\begin{cases}\max(0,(r_{ui}-2)/3),&(u,i)\in O;\\0,&\text{иначе}.\end{cases}$$
Таким образом, оценки 1/2/3/4/5 переходят в 0/0/⅓/⅔/1 вместо 0/0/0/1/1.
Предположение: нейтральный отзыв содержит слабый интерес; 5 сильнее 4.
Меняется **целевая переменная**, включая вклад 3-звёздочных отзывов в правую часть системы.
Матрица confidence, регуляризация и алгоритм совпадают с iALS; левая часть нормальных
уравнений остаётся той же при фиксированном $Y$. Это не перенастройка confidence.
Проверим гипотезу на общем бинарном test, а не на собственном градуированном критерии.

Для причинной интерпретации проводим два заранее заданных контроля: FM без метаданных с
настройками выбранной FM и CustomALS с настройками выбранного базового iALS. Эти настройки
также замораживаются до test. Если сетки выбрали одинаковые настройки iALS/CustomALS,
основная строка CustomALS одновременно является контролем.
""")
code(r"""
print('Primary selection metric:', CONFIG['primary_metric'])
print('Fixed seed:', CONFIG['seed'], '| K:', CONFIG['k'])
display(pd.DataFrame([
    {'model': name, 'trial': j, **params}
    for name, grid in CONFIG['search'].items() for j, params in enumerate(grid)
]).fillna('—'))
""")
md(r"""
Ограниченная вычислительная сетка задана в `config.json` до обращения к test.
Для каждого из пяти методов проверяем два варианта. У iALS и CustomALS одинаковые сетки,
факторы, число итераций, confidence и seed. Для FM фиксированы batch=2048 и 512 hash bins
на поле, сравниваются learning rate, L2 и число эпох. GPU не используется.
Результаты ниже получены обучением в этой ячейке.
""")
code(r"""
validation_results, selected = select_hyperparameters(protocol, CONFIG)
display(validation_results.sort_values(['model', 'NDCG@10'], ascending=[True, False]))
display(pd.DataFrame([{'model': name, **params} for name, params in selected.items()]).fillna('—'))
Path('results').mkdir(exist_ok=True)
Path('results/selected.json').write_text(json.dumps(selected, indent=2) + '\n')
""")
md(r"""
## 5. Финальное сравнение на test

Следующая ячейка заново обучает выбранные конфигурации на train+validation. Все модели
получают одинаковые test-пользователей, кандидатов и историю для исключения. После просмотра
test сетка и признаки не меняются. Время обучения приведено отдельно от времени оценки;
это однократное измерение, а не аппаратно-независимый benchmark.
""")
code(r"""
test_results, controls, example_recommendations, learning_curves = final_comparison(protocol, selected, CONFIG)
main_names = list(CONFIG['search'])
main_results = test_results[test_results.model.isin(main_names)].copy()
comparison = main_results.merge(
    validation_results.sort_values('NDCG@10', ascending=False).drop_duplicates('model')[['model', 'NDCG@10']],
    on='model', suffixes=('', '_validation'))
display(comparison.sort_values('NDCG@10', ascending=False)[[
    'model', 'NDCG@10_validation', 'NDCG@10', 'Recall@10', 'HitRate@10', 'Coverage@10',
    'fit_seconds', 'eval_seconds', 'n_users', 'params']])
save_results(validation_results, selected, test_results, controls, data_audit, protocol, CONFIG, learning_curves)
""")
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, metric in zip(axes, ['NDCG@10', 'Recall@10']):
    frame = main_results.sort_values(metric, ascending=False)
    ax.bar(frame.model, frame[metric], color=sns.color_palette('muted', len(frame)))
    ax.set(title=f'Test {metric}', ylabel=metric)
    ax.tick_params(axis='x', rotation=20)
plt.tight_layout(); plt.show()
display(test_results[['model', 'NDCG@10', 'Recall@10', 'Coverage@10', 'params']])
display(controls[['comparison', 'metric', 'difference', 'ci_low', 'ci_high', 'n_resamples']])
""")
md(r"""
Для контрольных сравнений выше показан **парный bootstrap по одним и тем же пользователям**:
2 000 повторных выборок с возвращением, seed=42, процентильный 95% интервал среднего изменения.
Интервал относится к вариативности пользователей при фиксированном обучении; он не учитывает
все возможные случайные инициализации и не является доказательством причинности.
""")
code(r"""
fig, ax = plt.subplots(figsize=(8, 3))
for name, values in learning_curves.items():
    ax.plot(np.arange(1, len(values)+1), values, marker='.', label=name)
ax.set(xlabel='Epoch', ylabel='Mean sampled BPR loss', title='Final FM training (no test feedback)')
ax.legend(); plt.tight_layout(); plt.show()
display(example_recommendations[example_recommendations.model.isin(main_names)].pivot(
    index='rank', columns='model', values='title'))
print('Example uses first eligible encoded test user; titles come from public item metadata.')
""")
md(r"""
## 6. Выводы по полученным результатам
""")
code(r"""
best = main_results.sort_values('NDCG@10', ascending=False).iloc[0]
best_recall = main_results.sort_values('Recall@10', ascending=False).iloc[0]
display(Markdown(f'**Лучшая NDCG@10: {best["model"]} — {best["NDCG@10"]:.5f}**, '
    f'Recall@10 = {best["Recall@10"]:.5f}. Настройки: `{best["params"]}`. '
    f'Максимальная Recall@10 у {best_recall["model"]}: {best_recall["Recall@10"]:.5f}.'))
for _, row in controls[controls.metric.eq('NDCG@10')].iterrows():
    direction = 'увеличил' if row.difference > 0 else 'снизил' if row.difference < 0 else 'не изменил'
    evidence = ('Интервал включает ноль; устойчивое преимущество не установлено.'
                if row.ci_low <= 0 <= row.ci_high else 'Интервал не включает ноль при фиксированном обучении.')
    display(Markdown(f'**{row["comparison"]}:** контроль {direction} NDCG@10 на '
        f'{row.difference:+.5f}; 95% CI [{row.ci_low:+.5f}, {row.ci_high:+.5f}]. {evidence}'))
display(Markdown('У iALS и CustomALS сопоставлены одинаковые confidence и настройки в контрольной '
    'строке: разность отражает выбранный таргет. У FM отключались только поля категорий и '
    'названий; неизменны разбиения, позитивы, число эпох и протокол негативов. '
    'Смена числа параметров FM означает, что это контроль всей конструкции признаков, '
    'а не изолированный эффект конкретного слова или категории.'))
display(Markdown(f'После очистки осталось **{len(reviews):,}** взаимодействий, '
    f'в warm-start протокол вошло **{protocol.audit["retained_rows_all_splits"]:,}**; '
    f'test содержит **{len(protocol.test_users):,}** оцениваемых пользователей '
    f'и **{protocol.n_items:,}** товаров-кандидатов.'))
""")
md(r"""
В выполненном запуске iALS и CustomALS выбрали $\lambda=10$ при 32 факторах,
$\alpha=20$ и 10 итерациях; EASE — $\lambda=500$; ItemKNN — 150 соседей и shrink=30.
У FM выбран lr=0.01, L2=$10^{-5}$, 10 эпох. Это лучшие настройки **в заданных двухточечных
сетках**, а не глобально оптимальные параметры.

Градуированный таргет увеличил Recall@10 с 0.02819 до 0.02878, но парный 95% интервал
разности [-0.00145, +0.00269] включает ноль. По основной NDCG@10 iALS немного выше;
убедительного преимущества нового таргета в этом эксперименте нет.

У FM добавление категорий и слов названий практически не изменило NDCG@10 (0.01486 → 0.01484),
а Recall@10 снизилось с 0.02562 до 0.02108; 95% интервал разности [-0.00828, -0.00095]
не включает ноль при фиксированном обучении. Полезность метаданных в этой конструкции
не подтверждается. Общие user/item-ID факторы контрольной и основной FM имели одинаковую
инициализацию; дополнительно изменилась ёмкость модели, а не только смысл признаков.

ItemKNN охватил 65.5% каталога, EASE — 39.3%, iALS — 6.3%: улучшение точности здесь
сопровождается меньшим охватом. На validation наилучшая NDCG@10 была у CustomALS (0.02025),
на test — у iALS (0.01758). Это показывает, почему validation и test следует разделять;
по test новые параметры не выбирались. Все числа этого абзаца относятся к сохранённому
запуску с `config.json`, seed=42; динамические таблицы и выводы выше вычисляются заново.

**Ограничения.** Отзыв — запаздывающий и смещённый индикатор потребления: нет экспозиций,
неизвестны покупки без отзывов. Оценки преимущественно высокие; выводы о градуированном
сигнале зависят от этой асимметрии. Преобразование оценок предполагает условно порядковую
шкалу и конкретное расстояние между уровнями, что требует отдельной проверки на других данных.

Готовый 5-core выбирался авторами по всей истории и исключает редких пользователей и товары;
временной split внутри него не устраняет этот selection bias. Результаты относятся к
warm-start пользователям с будущим позитивным отзывом. Доли исключённых cold-start событий
показаны в таблице протокола; качество на них не измерено. Каталог заморожен на train и
поэтому не оценивает появление новых товаров.

Метаданные — снимок карточек, а не исторический журнал изменений. Название и категория
обычно существуют до отзыва; это принятое здесь допущение доступности контента, но точные
прошлые версии источник не даёт. Запрещённые динамические поля и будущие отзывы не использованы.
Хеширование создаёт коллизии и ограничивает текст первыми 16 токенами.

Использован один seed и небольшая заранее заданная сетка; различия нельзя обобщать на
любые настройки этих семейств. Bootstrap не отражает вариативность обучения по seed.
EASE требует $O(|I|^2)$ памяти. Для production также нужны оценка cold-start, экспозиций,
разнообразия, доверительных интервалов по повторным обучениям и онлайн-проверка.

Агрегированные результаты, выбранные параметры, SHA256, версии библиотек и полный аудит
сохраняются в `results/`. Проверки формул, исключения известных товаров, временного
разбиения, метрик и FM находятся в `tests/`; запуск: `python -m pytest -q`.
""")

notebook = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.13.2'},
})
destination = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'experiments.ipynb'
nbf.write(notebook, destination)
print(f'Created {len(cells)} cells')
