import numpy as np
import pandas as pd

def calculate_factorial(n):
    """
    Args:
     n(int32)
    Returns:
     int32
    """
    # Base case: 0! = 1
    if n <= 0 or n > 12:
        return 1

    # Initialize factorial result
    result = 1

    # Loop to calculate factorial
    for i in range(1, n + 1):
        result *= i

    return result

def reverse_number(n):
    """
    Args:
     n(int32)
    Returns:
     int32
    """
    # Check the constraints
    if n < -100 or n > 100:
        raise ValueError("The input number must be between -100 and 100.")

    # Reverse and preserve the sign for negative numbers
    if n < 0:
        reversed_num = int('-' + str(-n)[::-1])  # Handle negative numbers
    else:
        reversed_num = int(str(n)[::-1])  # Handle positive numbers

    return reversed_num

def count_senior_citizens(p):
    """
    Args:
     p(list_str)
    Returns:
     int32
    """
    # Write your code here.
    count = 0

    for passenger in p:
        # Extract the age (12th and 13th characters of the string)
        age = int(passenger[11:13])

        # Check if the age is strictly over 60
        if age > 60:
            count += 1

    return count

def divisible_by_k(t, k):
    """
    Args:
     t(list_list_int32)
     k(int32)
    Returns:
     list_list_int32
    """
    # Write your code here.
    result = []

    for sublist in t:
        # Check if all elements in the sublist are divisible by k
        if all(element % k == 0 for element in sublist):
            result.append(sublist)

    return result


def compound_interest(principal, rate, time, duration):
    """
    Args:
     principal(float)
     rate(float)
     time(int32)
    Returns:
     float
    """
    # Write your code here.

    # Validate constraints
    if principal <= 0 or principal > 1_000_000:
        raise ValueError("Principal must be a positive float not exceeding 1,000,000.")
    if rate < 0 or rate > 1:
        raise ValueError("Rate must be a decimal between 0 and 1.")
    if time <= 0 or not isinstance(times_compounded, int):
        raise ValueError("Number of times compounded (n) must be a positive integer.")
    if duration <= 0 or not isinstance(duration, int):
        raise ValueError("Duration (t) must be a positive integer.")

    # Calculate the total amount
    amount = principal * (1 + rate / time) ** (time * duration)

    # Compound interest
    c_interest = amount - principal

    # Return rounded result
    return round(c_interest, 2)




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #print('factorial ', calculate_factorial(5))
    #print('reversed number 54:', reverse_number(54))
    #print('reversed number -90:', reverse_number(-90))

    # passengers_list = [
    #     "7823190130M7511",
    #     "6868346633F3422",
    #     "9921362780M5644",
    #     "9954362340M6707"
    # ]
    # print("Number of passengers over 60 years old:", count_senior_citizens(passengers_list))

    # t = [[1, 2, 3], [2, 4, 6], [3, 10, 11]]
    # k = 2
    # print(divisible_by_k(t, k))  # Output: [[2, 4, 6]]

    # # Example 1
    # principal = 1000
    # rate = 0.05
    # times_compounded = 1
    # result = compound_interest(principal, rate, times_compounded, duration)
    # print(f"Compound Interest (Example 1): {result}")  # Output: 157.63
    # duration = 3
    #
    # # Example 2
    # principal = 2500
    # rate = 0.08
    # times_compounded = 2
    # duration = 5
    # result = compound_interest(principal, rate, times_compounded, duration)
    # print(f"Compound Interest (Example 2): {result}")  # Output: 1210.64


    # arr = np.random.randint(1, 101, 100).reshape(10, -1)
    # print(arr.shape )

    # assignment 2
    # matrix = np.eye(4)
    # result = np.diagonal(matrix)
    # print(result)

    # A = np.array([[1, 2], [3, 4]])
    # B = A.T
    # print(B)

    # Create a 5x5 matrix with random integers between 1 and 100
    # matrix = np.random.randint(1, 101, size=(5, 5))
    #
    # # Calculate the mean of the matrix
    # mean = np.mean(matrix)
    #
    # # Calculate the standard deviation of the matrix
    # std_dev = np.std(matrix)
    #
    # # Print the matrix, mean, and standard deviation
    # print("5x5 Random Matrix:\n", matrix)
    # print("\nMean of the matrix:", mean)
    # print("Standard Deviation of the matrix:", std_dev)

    # Step 1: Create a 5x5 identity matrix
    # identity_matrix = np.eye(5)
    #
    # # Step 2: Add a scalar value of 5 to the identity matrix
    # modified_matrix = identity_matrix + 5
    #
    # # Step 3: Generate a 5x1 column vector with random integers between 1 and 10
    # column_vector = np.random.randint(1, 11, size=(5, 1))
    #
    # # Step 4: Multiply the modified matrix by the column vector
    # resulting_vector = np.dot(modified_matrix, column_vector)
    #
    # # Print the matrices and the resulting vector
    # print("5x5 Identity Matrix:\n", identity_matrix)
    # print("\nModified Matrix (Identity Matrix + 5):\n", modified_matrix)
    # print("\n5x1 Column Vector:\n", column_vector)
    # print("\nResulting Vector:\n", resulting_vector)

    # Step 1: Create two 3x3 matrices filled with random integers between 1 and 100
    # matrix1 = np.random.randint(1, 101, size=(3, 3))
    # matrix2 = np.random.randint(1, 101, size=(3, 3))
    #
    # # Step 2: Perform element-wise multiplication
    # result_matrix = np.multiply(matrix1, matrix2)
    #
    # # Print the matrices and the resulting matrix
    # print("Matrix 1:\n", matrix1)
    # print("\nMatrix 2:\n", matrix2)
    # print("\nResulting Matrix (Element-wise Multiplication):\n", result_matrix)

    # Step 1: Create a 1,000-element array of random integers between 1 and 1,000
    # random_array = np.random.randint(1, 1001, size=1000)
    #
    # # Step 2: Calculate the cumulative sum of the array
    # cumulative_sum = np.cumsum(random_array)
    #
    # # Step 3: Print the 10th, 100th, and 500th elements of the cumulative sum array
    # print("10th element of cumulative sum:", cumulative_sum[9])  # Index 9 (10th element)
    # print("100th element of cumulative sum:", cumulative_sum[99])  # Index 99 (100th element)
    # print("500th element of cumulative sum:", cumulative_sum[499])  # Index 499 (500th element)

    # Given data
    # data = {
    #     'Title': ['Inception', 'Dunkirk', 'Interstellar', 'The Prestige', 'Memento'],
    #     'Director': ['Christopher Nolan', 'Christopher Nolan', 'Christopher Nolan', 'Christopher Nolan',
    #                  'Christopher Nolan'],
    #     'Rating': [8.8, 7.9, 8.6, 8.5, 8.4]
    # }
    #
    # # Step 1: Create a DataFrame from the data dictionary
    # df = pd.DataFrame(data)
    #
    # # Step 2: Filter the DataFrame for movies directed by Christopher Nolan
    # nolan_movies = df[df['Director'] == 'Christopher Nolan']
    #
    # # Step 3: Calculate the average rating of Nolan's movies
    # average_rating = nolan_movies['Rating'].mean()
    #
    # # Step 4: Print the average rating
    # print("Average Rating of Christopher Nolan's Movies:", round(average_rating, 2))

    # Given the following DataFrame, write a Python program using Pandas to extract
    # all rows where the price is greater than or equal to 20,000 and sort them in
    # descending order by price.
    # data = {
    #     'Product': ['Laptop', 'Desktop', 'Tablet', 'Phone', 'Smartwatch'],
    #     'Price': [25000, 12000, 8000, 22000, 5000]
    # }
    #
    # df = pd.DataFrame(data)
    #
    # print(df[df['Price'] > 20000].sort_values('Price', ascending=False) )

    # # Write a Python program using Pandas to create a DataFrame from the given data and find the
    # # total revenue and average price of items sold in each store.
    # data = {
    #     'Store': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
    #     'Item': ['Apple', 'Banana', 'Orange', 'Grape', 'Apple', 'Banana', 'Orange', 'Grape'],
    #     'Price': [50, 20, 30, 60, 55, 22, 33, 65],
    #     'Quantity': [10, 12, 15, 16, 20, 25, 30, 35]
    # }
    #
    # df = pd.DataFrame(data)
    # df['Revenue'] = df['Price'] * df['Quantity']
    #
    # store_nums = df.groupby('Store').agg({'Revenue': 'sum', 'Price': 'mean'}).reset_index()
    # store_nums.rename(columns={'Price': 'Average_Price', 'Revenue': 'Total_Revenue'}, inplace=True)
    #
    # print(store_nums)

    # Write a Python program using Pandas to create a DataFrame from the given data and calculate
    # the total amount spent by each customer.
    data = {
        'Customer': ['Alice', 'Bob', 'Alice', 'Alice', 'Bob', 'Bob', 'Alice', 'Bob'],
        'Item': ['Pen', 'Pencil', 'Notebook', 'Eraser', 'Pen', 'Pencil', 'Notebook', 'Eraser'],
        'Price': [10, 5, 50, 20, 10, 5, 50, 20],
        'Quantity': [3, 4, 2, 5, 10, 6, 1, 2]
    }

    df = pd.DataFrame(data)

    df['Total_Spent'] = df['Price'] * df['Quantity']
    print(df.groupby('Customer')['Total_Spent'].sum().reset_index())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
