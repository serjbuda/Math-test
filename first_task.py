import numpy as np

# Завдання 1
array1d = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print("Завдання 1:")
print(array1d)

# Завдання 2
array2d = np.zeros((3, 3))
print("\nЗавдання 2:")
print(array2d)

# Завдання 3
random_array = np.random.randint(1, 11, size=(5, 5))
print("\nЗавдання 3:")
print(random_array)

# Завдання 4
random_float_array = np.random.rand(4, 4)
print("\nЗавдання 4:")
print(random_float_array)

# Завдання 5
array1 = np.random.randint(1, 11, size=5)
array2 = np.random.randint(1, 11, size=5)
add_result = array1 + array2
sub_result = array1 - array2
mul_result = array1 * array2
print("\nЗавдання 5:")
print("Додавання:", add_result)
print("Віднімання:", sub_result)
print("Множення:", mul_result)

# Завдання 6
vector1 = np.random.random(7)
vector2 = np.random.random(7)
scalar_product = np.dot(vector1, vector2)
print("\nЗавдання 6:")
print("Скалярний добуток:", scalar_product)

# Завдання 7
matrix1 = np.random.randint(1, 11, size=(2, 2))
matrix2 = np.random.randint(1, 11, size=(2, 3))
product_matrix = np.dot(matrix1, matrix2)
print("\nЗавдання 7:")
print("Матричний добуток:")
print(product_matrix)

import numpy as np

# Завдання 8
matrix3x3 = np.random.randint(1, 11, size=(3, 3))
inverse_matrix = np.linalg.inv(matrix3x3)
print("\nЗавдання 8:")
print("Обернена матриця:")
print(inverse_matrix)

# Завдання 9
matrix4x4 = np.random.rand(4, 4)
transposed_matrix = np.transpose(matrix4x4)
print("\nЗавдання 9:")
print("Транспонована матриця:")
print(transposed_matrix)

# Завдання 10
matrix3x4 = np.random.randint(1, 11, size=(3, 4))
vector4 = np.random.randint(1, 11, size=4)
result_vector = np.dot(matrix3x4, vector4)
print("\nЗавдання 10:")
print("Результат множення матриці на вектор:")
print(result_vector)

# Завдання 11
matrix2x3 = np.random.random((2, 3))
vector3 = np.random.random(3)
result_vector = np.dot(matrix2x3, vector3)
print("\nЗавдання 11:")
print("Результат множення матриці на вектор:")
print(result_vector)

# Завдання 12
matrixA = np.random.randint(1, 11, size=(2, 2))
matrixB = np.random.randint(1, 11, size=(2, 2))
elementwise_product = np.multiply(matrixA, matrixB)
print("\nЗавдання 12:")
print("Поелементне множення матриць:")
print(elementwise_product)

# Завдання 13
matrixA = np.random.randint(1, 11, size=(2, 2))
matrixB = np.random.randint(1, 11, size=(2, 2))
matrix_product = np.dot(matrixA, matrixB)
print("\nЗавдання 13:")
print("Матричний добуток матриць:")
print(matrix_product)

# Завдання 14
matrix5x5 = np.random.randint(1, 101, size=(5, 5))
sum_elements = np.sum(matrix5x5)
print("\nЗавдання 14:")
print("Сума елементів матриці:")
print(sum_elements)

# Завдання 15
matrixC = np.random.randint(1, 11, size=(4, 4))
matrixD = np.random.randint(1, 11, size=(4, 4))
matrix_difference = matrixC - matrixD
print("\nЗавдання 15:")
print("Різниця між матрицями:")
print(matrix_difference)

# Завдання 16
matrix3x3 = np.random.random((3, 3))
row_sums = np.sum(matrix3x3, axis=1)
column_vector = row_sums.reshape(-1, 1)
print("\nЗавдання 16:")
print("Вектор-стовпчик сум рядків матриці:")
print(column_vector)

# Завдання 17
matrixE = np.random.randint(-10, 11, size=(3, 4))
squared_matrix = np.square(matrixE)
print("\nЗавдання 17:")
print("Матриця з квадратами чисел:")
print(squared_matrix)

# Завдання 18
vector5 = np.random.randint(1, 51, size=4)
sqrt_vector = np.sqrt(vector5)
print("\nЗавдання 18:")
print("Вектор з квадратними коренями чисел:")
print(sqrt_vector)

