import numpy as np
import matplotlib.pyplot as plt


def main():
    # TODO: Might it be possible to read it in "transposed"?
    my_data = np.genfromtxt('linear_regression_data_with_one_variable.csv', delimiter=',')
    my_data_transposed = my_data.transpose()

    y = my_data_transposed[1]
    x = my_data_transposed[0]
    x_with_ones = [np.ones(len(y)), x]

    plt.title('Plot of training set')
    plt.xlabel('Population in City of 10,000s')
    plt.ylabel('Prices in $10,000s')
    plt.scatter(x, y)
    plt.show()

    initial_tetha = [0, 0]

    print('Initial cost: ', compute_cost(x_with_ones, y, initial_tetha))
    computed_tetha = gradient_descent(x_with_ones, y, initial_tetha, 0.01, 1500)
    print('Computed Theta1: ', computed_tetha[0])
    print('Computed Theta2: ', computed_tetha[1])

    computed_function = lambda x_in: computed_tetha[0] + computed_tetha[1] * x_in

    print('For population = 35,000, we predict a profit of', computed_function(3.5) * 10000)
    print('For population = 70,000, we predict a profit of', computed_function(7) * 10000)

    plt.title('Plot with trained function')
    plt.xlabel('Population in City of 10,000s')
    plt.ylabel('Prices in $10,000s')
    plt.scatter(x, y)
    y_values = [None] * 20
    for i, x in zip(range(0, 20), range(5, 25)):
        y_values[i] = computed_function(x)
    plt.plot(range(5, 25), y_values)
    plt.show()


def compute_cost(X, y_vector, theta):
    m = len(y_vector)
    cost = 0
    for x, y in zip(X[1], y_vector):
        cost += (theta[0] + theta[1] * x - y) ** 2

    return cost * (1 / (2 * m))


def gradient_descent(X, y_vector, initial_theta, alpha, number_of_iterations):
    m = len(y_vector)

    theta = initial_theta
    for i in range(0, number_of_iterations):
        sum_part_1 = 0
        for x, y in zip(X[1], y_vector):
            sum_part_1 += (theta[0] + theta[1] * x) - y
        temp1 = theta[0] - alpha * (1 / m) * sum_part_1

        sum_part_2 = 0
        for x, y in zip(X[1], y_vector):
            sum_part_2 += ((theta[0] + theta[1] * x) - y) * x
        temp2 = theta[1] - alpha * (1 / m) * sum_part_2

        theta = [temp1, temp2]

    return theta


if __name__ == "__main__":
    main()
