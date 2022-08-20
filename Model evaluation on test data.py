"""

Краткое введение

В качестве исходного набора данных используется предварительно обработанный датасет Task_1_prepprocessed.csv, далее строится модель для определения, является ли сообщение спамом.

Удаление строк, содержащих пропущенные значения, разделение датасета на тренировочный и тестовый набор данных с параметрами test_size = 0.35, random_state = 30. Обучение трех классификаторов RandomForestClassifier, использующих данные, полученные различными алгоритмами векторизации, при n_estimators = 9, n_jobs = 10 и random_state = 30 на обучающей выборке и произведение оценки полученной модели на тестовой.

Текст в STR взят из файла txt_for_test.txt

"""

TEST_SIZE = 0.35
RANDOM_STATE_1 = 30

N_ESTIMATORS = 9
N_JOBS = 10
RANDOM_STATE = 30

STR = "Market Price Volatility May 1-3, 2002 o Houston, TX Click Here To Download A Complete Conference Brochure http://www.pmaconference.com/mpv5_pma.pdf THIS IN-DEPTH TECHNICAL PROGRAM WILL SHOW YOU HOW TO: o How to Measure, Model, and Estimate Volatility o How to Model Price Volatility Using Financial Models o How to Use Blended Models Incorporating Fundamental Drivers o How to Apply Volatility Modeling to Today's Market Conditions: Full Requirements Contracts, Operating Performance of Generation Portfolios, Impacts of ISO Market Mitigation Infocast's highly regarded Market Price Volatility is a tightly-focused program specifically designed to attack the problems of modeling volatility in today's energy markets. It will provide you with the tools and insights you'll need to get and maintain an edge in assessing and managing volatility. The program will first provide you with an in-depth examination of sound market-based analytical processes and modeling techniques to accurately represent volatility, then will show you how these techniques are being applied to solving advanced energy market problems. TOPICS AND SPEAKERS INCLUDE: Estimating and Modeling Electricity and Fuel Price Volatility: A Comparison of Approaches Richard L. Carlson, Ph.D., Consulting Project Manager, Henwood Energy Services, Inc. Modeling Volatility Using Multi-Factor Models: A Practitioner's Approach Ionel Birgean, Director, Quantitative Analysis, Risk Management, PG&E National Energy Group Market-Based Price Forecasts: Integrating Fundamental and Market Components Sandra L. Ringelstetter Ennis, Executive Vice President, e-Acumen Advisory Services Are Price Spikes in Electricity Markets Predictable? Yumei Ning, Manager, Quantitative Analysis, Calpine Corporation Working Towards a Realistic Model to Price Generation Assets and Electricity Derivatives Michael Pierce, Ph.D., Financial Engineer, FEA Modeling Volatility: Mirant's Approaches Vance C. Mullis, Director of Market Evaluation Tools, Mirant Americas Summer 2001 Price Volatility in New England: Market Rules and Remedies Robert Ethier, Manager, Market Monitoring and Mitigation, ISO New England, Inc. An Integrated Approach to Modeling Price Uncertainty Mike King, Managing Partner, PA Consulting GROUP PRECONFERENCE WORKSHOP: Measuring, Modeling and Estimating Price Volatility Wednesday, May 1, 2002 o 8:00 AM-5:00 PM -Defining and Measuring Volatility -Modeling Volatility -Estimating Volatility -Issues in Modeling Volatility -Roundtable on Volatility Click Here To Download A Complete Conference Brochure http://www.pmaconference.com/mpv5_pma.pdf Presented By Infocast This email has been sent to michelle.lokay@enron.com, by PowerMarketers.com. Visit our Subscription Center to edit your interests or unsubscribe. http://ccprod.roving.com/roving/d.jsp?p=oo&m=1000838503237&ea=michelle.lokay@enron.com View our privacy policy: http://ccprod.roving.com/roving/CCPrivacyPolicy.jsp Powered by Constant Contact(R) www.constantcontact.com"

avg_1 = 'macro'
avg_2 = 'macro'
avg_3 = 'macro'

"""Генерация датасетов"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
import scipy as sp

"""Считывание обработанного датасета, удаление всех строк с пропусками"""

df = pd.read_csv('Task_1_prepprocessed.csv')
df = df.dropna()
df

"""Сравнение результаты, полученных при векторизации:

Для базового алгоритма из загруженного датасета берется только колонка class с обозначением класса сообщения и колонка body.
"""

x_train, x_test, y_train, y_test = train_test_split(
    df.iloc[:, 5], df.iloc[:,0], test_size=TEST_SIZE, random_state=RANDOM_STATE_1)

"""Глубокая копия данных для их дальнейшего преобразования. Конкатенация колонок subject и body. При разделении копии на тренировочный и тестовый наборы используется random_state как для основого алгоритма."""

df_copy = df.copy(deep = True)

for i in range(len(df_copy)):
    if df.iat[i, 4] != '':
        try:
            df_copy.iat[i, 5] = df_copy.iat[i, 4] + ' ' + df_copy.iat[i, 5]
        except:
            print("Something got wrong!")
            
x_train_b, x_test_b, _, _ = train_test_split(
    df_copy.iloc[:, [1, 5]], df_copy.iloc[:,0], test_size=TEST_SIZE, random_state=RANDOM_STATE_1)

# y_train_b и y_test_b совпадают с y_train, y_test. Хранить их отдельно надобности нет.

"""Генерация признаков

Составление словаря для базового алгоритма из тренировочного набора данных. Словарь используется для векторизации и генерации признаков TF-IDF. Затем этот же словарь используется в функции трансформации тестового набора. Предлагаемый класс TfidfVectorizer используется с параметрами по умолчанию.
"""

vectorizer_a = TfidfVectorizer()

x_train_a = vectorizer_a.fit_transform(x_train)
x_test_a = vectorizer_a.transform(x_test)

"""Для модифицированного алгоритма, где используется день недели, векторизация производится на колонке конкатенированных темы письма и содержания. К результату выполнения векторизации добавляется колонка индексов дней недели. Векторизатор также инициализируется с параметрами по умолчанию."""

vectorizer_b = TfidfVectorizer()
# Добавление к полученной в результате векторизации разреженной матрице столбец со значениями дней недели
x_train_b = sp.sparse.hstack((vectorizer_b.fit_transform(x_train_b.iloc[:, 1]), x_train_b.iloc[:, 0].values.reshape(len(x_train_b.iloc[:, 0]),1)))
x_test_b = sp.sparse.hstack((vectorizer_b.transform(x_test_b.iloc[:, 1]), x_test_b.iloc[:, 0].values.reshape(len(x_test_b.iloc[:, 0]),1)))

"""Третий сценарий, вместо слов используются биграммы для генерации признаков TF-IDF, при инициализации векторизатора указывается параметр ngram_range = (2, 2)."""

vectorizer_c = TfidfVectorizer(ngram_range = (2, 2))

x_train_c = vectorizer_c.fit_transform(x_train)
x_test_c = vectorizer_c.transform(x_test)

strs = STR

arr = pd.Series(strs)

x_test_d = vectorizer_c.transform(arr)

"""Классификация

Инициализация классификаторов с одинаковыми параметрами.
"""

rfc_a = RandomForestClassifier(random_state = RANDOM_STATE, n_jobs=N_JOBS, n_estimators=N_ESTIMATORS)
rfc_b = RandomForestClassifier(random_state = RANDOM_STATE, n_jobs=N_JOBS, n_estimators=N_ESTIMATORS)
rfc_c = RandomForestClassifier(random_state = RANDOM_STATE, n_jobs=N_JOBS, n_estimators=N_ESTIMATORS)

rfc_a.fit(x_train_a, y_train)
rfc_b.fit(x_train_b, y_train)
rfc_c.fit(x_train_c, y_train)

"""Расчет метрик

Предсказания на тестовых данных, оценка полученных моделей для трех случаев.
"""

y_pred_a = rfc_a.predict(x_test_a)
y_pred_b = rfc_b.predict(x_test_b)
y_pred_c = rfc_c.predict(x_test_c)


print(classification_report(y_test, y_pred_a, digits=3))
print(classification_report(y_test, y_pred_b, digits=3))
print(classification_report(y_test, y_pred_c, digits=3))

"""Пример: сравение первых двух сценириев с точки зрения FPR и precision. Для получения FPR достаточно сгенерировать confusion matrix и рассчитать его на основе значений из матрицы, в то время как precision может быть получен с помощью отдельной функции."""

tn_a, fp_a, _, _  = confusion_matrix(y_test, y_pred_a).ravel()
tn_b, fp_b, _, _ = confusion_matrix(y_test, y_pred_b).ravel()

fpr_a = fp_a / (fp_a + tn_a)
fpr_b = fp_b / (fp_b + tn_b)

pr_a = precision_score(y_test, y_pred_a)
pr_b = precision_score(y_test, y_pred_b)

print(f'Difference in FPR: {fpr_b - fpr_a}')
print(f'Difference in precision: {pr_b - pr_a}')

"""Оценка модели на тестовых данных

Использование TfidfVectorizer() на столбце body. Вывод метрик:

* Precision (macro avg)
* Recall (macro avg)
* F-score (macro avg)
"""

from sklearn import metrics

macro_precision, macro_recall, macro_f1, _support =\
    metrics.precision_recall_fscore_support(y_test, y_pred_a, average=avg_1)

print("Macro avg\n\tprecision\t{}\n\trecall\t\t{}\n\tf1 score\t{}\n\n".format(round(macro_precision, 3), round(macro_recall, 3), round(macro_f1, 3)))

"""Использование TfidfVectorizer() на объединенных в результате конкатенации столбцах subject и body. Дополение полученных предикторов столбцом с метками дней недели. Вывод метрик:

* Precision (macro avg)
* Recall (macro avg)
* F-score (macro avg)
"""

macro_precision, macro_recall, macro_f1, _support =\
    metrics.precision_recall_fscore_support(y_test, y_pred_b, average=avg_2)

print("Macro avg\n\tprecision\t{}\n\trecall\t\t{}\n\tf1 score\t{}\n\n".format(round(macro_precision, 3), round(macro_recall, 3), round(macro_f1, 3)))

"""Использование TfidfVectorizer() на колонке body с параметром ngram_range = (2, 2). Вывод метрик:

* Precision (macro avg)
* Recall (macro avg)
* F-score (macro avg)
"""

macro_precision, macro_recall, macro_f1, _support =\
    metrics.precision_recall_fscore_support(y_test, y_pred_c, average=avg_3)

print("Macro avg\n\tprecision\t{}\n\trecall\t\t{}\n\tf1 score\t{}\n\n".format(round(macro_precision, 3), round(macro_recall, 3), round(macro_f1, 3)))

"""Выполнение предсказания для текста txt_for_test.txt, с использованием последнего алгоритма."""

print(rfc_c.predict(x_test_d)[0])

ans = round(rfc_c.predict_proba(x_test_d)[:, 1][0], 3)

print(ans if ans > 0.5 else 1 - ans)