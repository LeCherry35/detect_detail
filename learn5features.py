import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
import joblib
import matplotlib.pyplot as plt
from utils.countAngle import calculate_angle_from_points


def get_data(data_folder):
    # Список для хранения данных
    data = []

    # Чтение файлов
    for filename in os.listdir(data_folder):
        if filename.endswith(".txt"):  # Читаем только текстовые файлы
            with open(os.path.join(data_folder, filename), "r") as file:
                for line in file:
                    # Разделяем строку на название детали и координаты
                    parts = line.strip().split(";")
                    part_name = parts[0]
                    coordinates = []
                    for point in parts[1:]:
                        # Преобразуем координаты в числа
                        coords = list(map(float, point.split(",")))
                        coordinates.append(
                            coords
                        )  # Добавляем координаты в общий список
                    # Если 3 точки, копируем последнюю
                    if len(coordinates) < 4:
                        coordinates.append(coordinates[2])
                    # Вычисляем углы
                    angle12 = calculate_angle_from_points(
                        coordinates[1], coordinates[0], coordinates[1], coordinates[2]
                    )
                    angle23 = calculate_angle_from_points(
                        coordinates[2], coordinates[1], coordinates[2], coordinates[3]
                    )
                    angle13 = calculate_angle_from_points(
                        coordinates[0], coordinates[1], coordinates[2], coordinates[3]
                    )
                    angles = [angle12, angle23, angle13]
                    data.append(
                        {
                            "part_name": part_name,
                            "coordinates": coordinates,
                            "angles": angles,
                        }
                    )

    # Преобразуем в DataFrame
    df = pd.DataFrame(data)
    return df

def prep_data_and_learn(df):
    # Кодируем названия деталей в числовые метки
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["part_name"])

    # Преобразуем в numpy-массив
    X = np.array(df["angles"].tolist())
    y = np.array(df["label"])

    # Разделяем данные перед нормализацией
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42
    )

    # Создаем MinMaxScaler и обучаем только на train
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)

    # Применяем только transform() к валидации и тесту
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Создаем модель
    model = Sequential(
        [
            # Входной слой
            Dense(64, activation="relu", input_shape=(3,)),  # 3 входных признака
            Dropout(0.2),
            Dense(32, activation="relu", kernel_regularizer=l2(0.001)),
            Dropout(0.2),
            # Выходной слой
            Dense(
                5, activation="softmax"
            ),  # 5 выходных нейрона (по количеству классов)
        ]
    )

    optimizer = SGD(learning_rate=0.01, momentum=0.9)
    # Компилируем модель
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",  # Функция потерь для целочисленных меток
        metrics=["accuracy"],
    )

    # Выводим информацию о модели
    model.summary()

    # Обучаем модель
    history = model.fit(
        X_train,
        y_train,  # Обучающие данные
        epochs=5,  # Количество эпох
        batch_size=50,  # Размер батча
        validation_data=(X_val, y_val),  # Валидационные данные
    )

    # Оценка модели на тестовых данных
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Точность на тестовых данных: {test_accuracy:.4f}")

    return model, history, scaler, label_encoder


def create_graph_and_save(history, folder_path):

    # График точности
    plt.plot(history.history["accuracy"], label="Обучающая выборка")
    plt.plot(history.history["val_accuracy"], label="Валидационная выборка")
    plt.title("Точность модели")
    plt.xlabel("Эпохи")
    plt.ylabel("Точность")
    plt.legend()
    plt.savefig(
        os.path.join(folder_path, "accuracy_plot.png"), dpi=300, bbox_inches="tight"
    )
    plt.show()

    # График потерь
    plt.plot(history.history["loss"], label="Обучающая выборка")
    plt.plot(history.history["val_loss"], label="Валидационная выборка")
    plt.title("Потери модели")
    plt.xlabel("Эпохи")
    plt.ylabel("Потери")
    plt.legend()
    plt.savefig(
        os.path.join(folder_path, "loss_plot.png"), dpi=300, bbox_inches="tight"
    )
    plt.show()


def save_model(model, label_encoder, scaler, folder_path):
    # Сохраняем модель
    model.save(os.path.join(folder_path, "model.h5"))
    print("Модель сохранена.")

    # Сохраняем MinMaxScaler
    joblib.dump(scaler, os.path.join(folder_path, "min_max_scaler_wse.pkl"))
    print("MinMaxScaler сохранен.")

    # Сохраняем LabelEncoder
    joblib.dump(label_encoder, os.path.join(folder_path, "label_encoder_wse.pkl"))
    print("LabelEncoder сохранен.")


def main():
    # Путь к папке с файлами
    data_folder = input("Введите путь к папке с обучающими данными: ")
    df = get_data(data_folder)

    model, history, scaler, label_encoder = prep_data_and_learn(df)

    folder_path = input("Введите путь для сохранения модели: ")
    os.makedirs(folder_path, exist_ok=True)  # Создаёт папку, если её нет

    create_graph_and_save(history, folder_path)

    save_model(model, label_encoder, scaler, folder_path)


if __name__ == "__main__":
    main()
