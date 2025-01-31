import ast
import matplotlib.pyplot as plt
import os

def read_line_from_file(file_name, line_number):
    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            if 1 <= line_number <= len(lines):
                return lines[line_number - 1].strip()
            else:
                return f"Ошибка: строка с номером {line_number} не существует в файле."
    except FileNotFoundError:
        return "Ошибка: файл не найден."
    except Exception as e:
        return f"Произошла ошибка: {e}"
    
def parse_coordinates(input_string):
    print(input_string)
    # Разделяем строку по символу ";"
    parts = input_string.split(";")
    
    # Проверяем, есть ли название детали (первая часть не является координатами)
    if not parts[0].replace(".", "").replace(",", "").strip().isdigit():
        # Если есть название, удаляем его
        parts = parts[1:]
    
    # Извлекаем координаты
    coordinates = []
    for part in parts:
        try:
            # Разделяем часть на три числа по запятым
            x, y, z = map(float, part.split(","))
            coordinates.append((x, y, z))
        except ValueError:
            # Если не удалось преобразовать в числа, пропускаем
            print(f"Ошибка: некорректные данные в части '{part}'")
    
    return coordinates

def visualize_coordinates(coordinates):
    if not coordinates:
        return

    # Создаем 3D-график
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Извлекаем координаты X, Y, Z
    x = [point[0] for point in coordinates]
    y = [point[1] for point in coordinates]
    z = [point[2] for point in coordinates]

    # Рисуем точки
    ax.scatter(x, y, z, c='r', marker='o', label='Точки')

    # Рисуем отрезки, соединяющие точки последовательно
    for i in range(len(coordinates) - 1):
        ax.plot(
            [x[i], x[i + 1]],
            [y[i], y[i + 1]],
            [z[i], z[i + 1]],
            c='b',
            label='Отрезки' if i == 0 else ""
        )

    # Настройки графика
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title("3D-визуализация точек и отрезков")
    plt.show()

def main():
    
    # file_name = input("Введите имя файла: ") 
    # folder_name = input("Введите имя папки: ")
    # file_path = os.path.join(folder_name, file_name)
    # file_name = "stabCoordinates.txt"  
    # file_name = "startData/DetailsData-KickRight.txt"  
    # file_name = "startData/DetailsData-KickStandard.txt" 
    # file_name = "startData/DetailsData-BackToBackS-Offset.txt"  
    file_name = input("Введите имя файла и путь: ")

    try:
        line_number = int(input("Введите номер строки: "))
        result = read_line_from_file(file_name, line_number)
        coordinates = parse_coordinates(result)

        visualize_coordinates(coordinates)

        print(f"Координаты точек в строке {line_number}: {coordinates}")
        
        
    except ValueError:
        print("Ошибка: введите корректный номер строки (целое число).")

if __name__ == "__main__":
    main()