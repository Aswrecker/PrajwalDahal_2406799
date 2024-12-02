import numpy as np
from time import time

# Problem 1
# Q1
arr_empty = np.empty((2, 2), dtype=int)

# Q2
arr_ones = np.ones((4, 2), dtype=int)

# Q3
fill_value = 7
arr_full = np.full((3, 3), fill_value)

# Q4
reference_array = np.array([[1, 2], [3, 4], [5, 6]])
arr_zeroes = np.zeros_like(reference_array)

# Q5
arr_ones_like = np.ones_like(reference_array)

# Q6
new_list = [1, 2, 3, 4]
array_list = np.array(new_list)

print("arr_empty: ", arr_empty)
print("arr_ones: ", arr_ones)
print("arr_full: ", arr_full)
print("arr_zeroes: ", arr_zeroes)
print("arr_ones_like: ", arr_ones_like)
print("new_list: ", new_list)
print("array_list: ", array_list)


# Problem 2
# Q1
arr_arange = np.arange(10, 50, dtype=int)
print(arr_arange)

# Q2
arr_reshape = np.reshape(np.arange(9), (3, 3))
print(arr_reshape)

# Q3
arr_identity = np.eye(3, dtype=int)
print(arr_identity)

# Q4
arr_mean = np.mean(np.random.random(30))
print(arr_mean)

# Q5
arr_max_min = np.random.random((10, 10))
arr_max = np.max(arr_max_min)
arr_min = np.min(arr_max_min)
print(arr_max)
print(arr_min)

# Q6
arr_zero = np.zeros(10, dtype=int)
print(arr_zero)
arr_zero[4] = 1
print(arr_zero)

# Q7
arr = [1, 2, 0, 0, 4, 0]
arr_reverse = arr[::-1]
print(arr_reverse)

# Q8
rows, cols = 4, 4
array = []
for i in range(rows):
    row = []
    for j in range(cols):
        if i == 0 or i == rows - 1 or j == 0 or j == cols - 1:
            row.append(1)
        else:
            row.append(0)
    array.append(row)
for row in array:
    print(row)

# Q9
matrix = []
for i in range(8):
    row = []
    for j in range(8):
        if (i + j) % 2 == 0:
            row.append(1)
        else:
            row.append(0)
    matrix.append(row)
for row in matrix:
    print(row)


# Problem 3
x = np.array([[1, 2], [3, 5]])
y = np.array([[5, 6], [7, 8]])
v = np.array([9, 10])
w = np.array([11, 12])

# Q1
add_xy = x + y
print("Addition of x and y:\n", add_xy)

# Q2
subtract_xy = x - y
print("Subtraction of x and y:\n", subtract_xy)

# Q3
multiply_x = 3 * x
print("Multiplication of x with 3:\n", multiply_x)

# Q4
square_x = np.square(x)
print("Square of each element in x:\n", square_x)

# Q5
dot_vw = np.dot(v, w)
print("Dot product of v and w:", dot_vw)

dot_xv = np.dot(x, v)
print("Dot product of x and v:", dot_xv)

dot_xy = np.dot(x, y)
print("Dot product of x and y:\n", dot_xy)

# Q6
concat_xy_rows = np.concatenate((x, y))
print("Concatenation of x and y along rows:\n", concat_xy_rows)

concat_vw_columns = np.vstack((v, w))
print("Concatenation of v and w along columns:\n", concat_vw_columns)

# Q7
try:
    concat_xv = np.concatenate((x, v))
    print("Concatenation of x and v:\n", concat_xv)
except ValueError as e:
    print("Error while concatenating x and v:", e)


# Problem 4.1
A = np.array([[3, 4], [7, 8]])
B = np.array([[5, 3], [2, 1]])

# Q1.1
A_inv = np.linalg.inv(A)
identity_matrix = np.dot(A, A_inv)
print(np.allclose(identity_matrix, np.eye(2)))

# Q1.2
AB = np.dot(A, B)
BA = np.dot(B, A)
AB_equals_BA = np.array_equal(AB, BA)
print(AB_equals_BA)

# Q1.3
AB_transpose = np.transpose(AB)
B_transpose_A_transpose = np.dot(np.transpose(B), np.transpose(A))
AB_transpose_equals_BT_AT = np.array_equal(AB_transpose, B_transpose_A_transpose)
print(AB_transpose_equals_BT_AT)

# Q2
A = np.array([[2, -3, 1], [1, -1, 2], [3, 1, -1]])
B = np.array([-1, -3, 9])
A_inv = np.linalg.inv(A)
X = np.dot(A_inv, B)
print("The solution is:\n", X)


# Experiment: How Fast is Numpy?
list_size = 1_000_000

# Q1. Element-wise Addition
# Using Python lists
start_time = time()
list1 = [i for i in range(list_size)]
list2 = [i for i in range(list_size)]
list_addition = [list1[i] + list2[i] for i in range(list_size)]
list_addition_time = time() - start_time
print(f"Start Time: {start_time}\nEnd Time: {list_addition_time}")
# Using NumPy arrays
array1 = np.arange(list_size)
array2 = np.arange(list_size)
start_time = time()
array_addition = array1 + array2
array_addition_time = time() - start_time
print(f"Start Time: {start_time}\nEnd Time: {array_addition}")

# Q2. Element-wise Multiplication
# Using Python lists
start_time = time()
list_multiplication = [list1[i] * list2[i] for i in range(list_size)]
list_multiplication_time = time() - start_time
print(f"Start Time: {start_time}\nEnd Time: {list_multiplication_time}")

# Using NumPy arrays
start_time = time()
array_multiplication = array1 * array2
array_multiplication_time = time() - start_time
print(f"Start Time: {start_time}\nEnd Time: {array_multiplication_time}")

# Q3. Dot Product
# Using Python lists
start_time = time()
list_dot_product = sum(list1[i] * list2[i] for i in range(list_size))
list_dot_product_time = time() - start_time
print(f"Start Time: {start_time}\nEnd Time: {list_dot_product_time}")

# Using NumPy arrays
start_time = time()
array_dot_product = np.dot(array1, array2)
array_dot_product_time = time() - start_time
print(f"Start Time: {start_time}\nEnd Time: {array_dot_product_time}")

# Q4. Matrix Multiplication
# Using Python lists
matrix_size = 1000
matrix1 = [[i for i in range(matrix_size)] for _ in range(matrix_size)]
matrix2 = [[i for i in range(matrix_size)] for _ in range(matrix_size)]
start_time = time()
matrix_multiplication = [
    [sum(matrix1[i][k] * matrix2[k][j] for k in range(matrix_size)) for j in range(matrix_size)]
    for i in range(matrix_size)
]
list_matrix_multiplication_time = time() - start_time
print(f"Start Time: {start_time}\nEnd Time: {list_matrix_multiplication_time}")

# Using NumPy arrays
array_matrix1 = np.arange(matrix_size**2).reshape(matrix_size, matrix_size)
array_matrix2 = np.arange(matrix_size**2).reshape(matrix_size, matrix_size)
start_time = time()
array_matrix_multiplication = np.dot(array_matrix1, array_matrix2)
array_matrix_multiplication_time = time() - start_time
print(f"Start Time: {start_time}\nEnd Time: {array_matrix_multiplication_time}")
