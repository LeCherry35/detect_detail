from tensorflow.keras.models import load_model
import numpy as np
import joblib
import os
from utils.countAngle import calculate_angle_from_points


def retrieve_data_from_file(path):
    new_data = []
    with open(path, "r") as file:
        for line in file:
            # Разделяем строку на название детали и координаты
            points = line.strip().split(";")
            # print(points)
            # exit()
            coordinates = []
            for point in points:
                # Преобразуем координаты в числа
                coords = list(map(float, point.split(",")))

                # print(coords)
                coordinates.append(coords)  # Добавляем координаты в общий список
            if len(coordinates) < 4:
                coordinates.append(coordinates[2])
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

            new_data.append(angles)
    return new_data


def etrieve_model(folder_path):
    # Загружаем модель
    loaded_model = load_model(os.path.join(folder_path, "model.h5"))

    # Загружаем MinMaxScaler
    scaler = joblib.load(os.path.join(folder_path, "min_max_scaler_wse.pkl"))

    # Загружаем LabelEncoder
    label_encoder = joblib.load(os.path.join(folder_path, "label_encoder_wse.pkl"))

    return loaded_model, scaler, label_encoder


def main():
    folder = input("Введите путь к папке содержащей модель, енкодер и скейлер: ")

    loaded_model, scaler, label_encoder = etrieve_model(folder)

    data_folder = input("Введите путь к папке содержащей течтовый файл: ")
    filename = input("Введите имя тестового файла: ")

    new_data = retrieve_data_from_file(os.path.join(data_folder, filename))

    # Нормализация данных (используйте тот же scaler, что и при обучении)
    new_data_normalized = scaler.fit_transform(new_data)  # Нормализуем новые данные

    prediction = loaded_model.predict(new_data_normalized)
    predicted_class = np.argmax(prediction, axis=1)

    # Преобразуем числовой класс обратно в название детали
    predicted_part_name = label_encoder.inverse_transform(predicted_class)

    for i, (part, confidence) in enumerate(zip(predicted_part_name, prediction)):
        print(f"Line {i+1}:")
        print(f"Predicted part: {part}")
        print(f"Confidence: {np.max(confidence)*100:.2f}%")
        print()


if __name__ == "__main__":
    main()
