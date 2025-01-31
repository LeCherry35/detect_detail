from tensorflow.keras.models import load_model
import numpy as np
import joblib
import os
import math

def calculate_vector(p1, p2):
    # Вычисляет вектор между двумя точками
    return (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])

def vector_length(v):
    # Вычисляет длину вектора
    return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

def dot_product(v1, v2):
    # Вычисляет скалярное произведение векторов
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]

def angle_between_vectors(v1, v2):
    # Вычисляет угол между векторами в градусах
    dot = dot_product(v1, v2)
    lengths = vector_length(v1) * vector_length(v2)
    if lengths == 0:
        return -1
    # Защита от погрешностей вычислений
    cos_angle = max(min(dot / lengths, 1), -1)
    return math.degrees(math.acos(cos_angle))

def calculate_angle_from_points(p1, p2, p3, p4):
    # Вычисляет угол между отрезками, заданными тремя точками

        # Вычисляем векторы
        vector1 = calculate_vector(p1, p2)
        vector2 = calculate_vector(p3, p4)
        
        # Вычисляем угол
        return round(angle_between_vectors(vector1, vector2))


folder = input("Введите путь к папке содержащей модель, енкодер и скейлер: ")
# Загружаем модель
loaded_model = load_model(os.path.join(folder, "model.h5"))

# Загружаем MinMaxScaler
scaler = joblib.load(os.path.join(folder, "min_max_scaler_wse.pkl"))

# Загружаем LabelEncoder
label_encoder = joblib.load(os.path.join(folder, "label_encoder_wse.pkl"))

# Пример предсказания
# new_data = np.array([[0.62, 0.17, 0.54, 0.62, 0.15, 0.54, 0.63, 0.13, 0.54, 0, 0, 0]])  # Пример данных
data_folder = input("Введите путь к папке содержащей течтовый файл: ")
filename = input("Введите имя тестового файла: ")
new_data = []
with open(os.path.join(data_folder, filename), "r") as file:
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
        angle12 = calculate_angle_from_points(coordinates[1], coordinates[0], coordinates[1], coordinates[2])
        angle23 = calculate_angle_from_points(coordinates[2], coordinates[1], coordinates[2], coordinates[3])
        angle13 = calculate_angle_from_points(coordinates[0], coordinates[1], coordinates[2], coordinates[3])
        angles = [angle12, angle23, angle13]
        
        new_data.append(angles)


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