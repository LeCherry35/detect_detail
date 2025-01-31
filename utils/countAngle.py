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

