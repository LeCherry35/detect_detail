import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Открываем файл с координатами
with open('testCoordinates.txt', 'r') as file:
# with open('coordinates.txt', 'r') as file:
    lines = file.readlines()  # Читаем все строки из файла

# Создаем пустой список для хранения координат
points = []

# Обрабатываем каждую строку
for line in lines:
    # Разделяем строку по точке с запятой
    parts = line.strip().split(';')
    
    # Первая часть — название детали, её пропускаем
    # Остальные части — координаты точек
    for part in parts[1:]:  # Начинаем с индекса 1, чтобы пропустить название
        # Разделяем координаты по запятой и преобразуем в числа
        x, y, z = map(float, part.split(','))
        points.append([x, y, z])  # Добавляем точку в список

# Преобразуем список в массив NumPy
points = np.array(points)

# Запрашиваем у пользователя число n
n = int(input("Введите число n (визуализировать каждую n-ую точку): "))

# Выбираем каждую n-ую точку
selected_points = points[::n]

# Выводим количество выбранных точек для проверки
print(f"Выбрано {len(selected_points)} точек из {len(points)}")

# Выводим результат для проверки
print("Координаты точек:")
print(points)

# Создаем 3D-график
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Разделяем координаты
x = selected_points[:, 0]  # Все значения по оси X
y = selected_points[:, 1]  # Все значения по оси Y
z = selected_points[:, 2]  # Все значения по оси Z

# Рисуем точки
ax.scatter(x, y, z, c='blue', marker='x', s=1)  # s=50 — размер точек

# Подписываем оси
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Добавляем заголовок
ax.set_title('3D Визуализация точек')

# Показываем график
plt.show()