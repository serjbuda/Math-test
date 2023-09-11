import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Загрузка данных и объединение
data_folder = r'C:\Users\Сергій\Documents\Code\Current_task\Math-test\homework\data'
csv_files = []

for root, dirs, files in os.walk(data_folder):
    for file in files:
        if file.endswith('.csv'):
            csv_files.append(os.path.join(root, file))

# Создание пустого DataFrame
data = pd.DataFrame()

# Загрузка и объединение данных из CSV-файлов
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    df['activity'] = os.path.basename(csv_file).split('.')[0]  # Используйте имя файла как метку активности
    data = pd.concat([data, df])

# 2. Извлечение временных признаков (подставьте свои фичи)
# Пример: среднее значение по осям X, Y, Z
data['mean_x'] = data.groupby('activity')['accelerometer_X'].transform('mean')
data['mean_y'] = data.groupby('activity')['accelerometer_Y'].transform('mean')
data['mean_z'] = data.groupby('activity')['accelerometer_Z'].transform('mean')

# 3. Разделение данных на обучающую и тестовую выборки
X = data[['mean_x', 'mean_y', 'mean_z']]
y = data['activity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Масштабирование признаков
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Обучение моделей
# SVM
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# 6. Оценка производительности моделей
# SVM
svm_predictions = svm_classifier.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_report = classification_report(y_test, svm_predictions)

# Random Forest
rf_predictions = rf_classifier.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_report = classification_report(y_test, rf_predictions)

# 7. Вывод результатов
print("SVM Accuracy:", svm_accuracy)
print("SVM Classification Report:\n", svm_report)

print("\nRandom Forest Accuracy:", rf_accuracy)
print("Random Forest Classification Report:\n", rf_report)

