import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


class ProbabilitesTheory:
    @classmethod
    def mathemactical_expectation(cls, array: np.array, probabilites_array: np.array) -> float:
        return array @ probabilites_array
    
    @classmethod
    def dispersion(cls, array: np.array, probabilites_array: np.array) -> float:
        first = cls.mathemactical_expectation(np.square(array), probabilites_array)
        second = np.square(cls.mathemactical_expectation(array, probabilites_array))

        return first - second

    @classmethod
    def standart_deviation(cls, array: np.array, probabilites_array: np.array) -> float:
        return np.sqrt(cls.dispersion(array, probabilites_array))

    @classmethod
    def covariance(cls, probabilites_matrix: np.matrix, a: np.array, b: np.array) -> float:
        values = a @ np.sum(probabilites_matrix, axis=1), b @ np.sum(probabilites_matrix, axis=0)

        return a @ probabilites_matrix @ b - np.prod(values)
    
    @classmethod
    def correlation(cls, probabilites_matrix: np.matrix, a: np.array, b: np.array) -> float:
        st_dev_a = cls.standart_deviation(a, np.sum(probabilites_matrix, axis=1))
        st_dev_b = cls.standart_deviation(b, np.sum(probabilites_matrix, axis=0))

        return cls.covariance(probabilites_matrix, a, b) / st_dev_a / st_dev_b

    @classmethod
    def pirson(cls, matrix_emp: np.matrix, matrix_th: np.matrix, count: int) -> float:
        return count * np.sum(np.square(matrix_emp - matrix_th) / matrix_th)
    
    @classmethod
    def statistical_pirson(cls, matrix: np.matrix) -> float:
        return stats.chi2.ppf(0.95, matrix.size - 1)
    
    @classmethod
    def satisfies_theoretical(cls, matrix_emp: np.matrix, matrix_th: np.matrix, count: int) -> bool:
        return cls.statistical_pirson(matrix_th) > cls.pirson(matrix_emp, matrix_th, count)

    @classmethod
    def mathematical_expectation_interval(cls, count, values) -> tuple:
        normal_quantile = stats.norm.ppf(1.95 / 2)

        values_mean = np.mean(values)
        values_var = np.var(values, ddof=1)

        return (
            values_mean - np.sqrt(values_var / count) * normal_quantile, 
            values_mean + np.sqrt(values_var / count) * normal_quantile
        )

    @classmethod
    def dispersion_interval(cls, n, values) -> tuple:
        rv_var = np.var(values, ddof=1)
        chi = stats.chi2(n - 1)
        array = chi.rvs(100000)
        q = stats.mstats.mquantiles(array, prob=[0.05 / 2, 1.95 / 2])

        xi_plus = q[0]
        xi_minus = q[1]

        return (n - 1) * rv_var / xi_minus, (n - 1) * rv_var / xi_plus


class DTDRandomVariableGenerator:
    def __init__(self, matrix):
        self.sum_by_row = np.sum(matrix, axis=1)
        self.cum_sum = np.cumsum(self.sum_by_row)
        self.cum_sum_by_row = np.cumsum(matrix, axis=1) / self.sum_by_row.reshape(-1, 1)

    def __next__(self):
        value = np.random.uniform(size=2)
        row = np.searchsorted(self.cum_sum, value[0])
        column = np.searchsorted(self.cum_sum_by_row[row], value[1])
        return row, column


def init() -> tuple:
    print("Input n", end=" ")
    n = int(input())

    print("Input m", end=" ")
    m = int(input())

    matrix = np.random.rand(n, m)
    matrix /= matrix.sum()

    print("matrix = ", matrix)

    a = np.array([i for i in range(0, n)])
    b = np.array([i for i in range(0, m)])

    print("a = ", a)
    print("b = ", b)

    return matrix, a, b


def create_histograms(matrix_th, matrix_emp):
    values_a_th = np.sum(matrix_th, axis=1)
    values_b_th = np.sum(matrix_th, axis=0)

    values_a_emp = np.sum(matrix_emp, axis=1)
    values_b_emp = np.sum(matrix_emp, axis=0)

    data = {'теоретическая': values_a_th.tolist(), 'эмпирическая': values_a_emp.tolist()}
    df = pd.DataFrame(data)
    df.plot(kind='bar')
    plt.title("Гистограммы А")
    plt.show()

    data = {'теоретическая': values_b_th.tolist(), 'эмпирическая': values_b_emp.tolist()}
    df = pd.DataFrame(data)
    df.plot(kind='bar')
    plt.title("Гистограммы B")
    plt.show()


def generate_matrix_emp_and_values(gen):
    matrix_emp = np.zeros(matrix_th.shape)
    values = []

    for _ in range(count):
        row, column = next(gen)
        matrix_emp[(row, column)] += 1
        values.append((row, column))
    
    matrix_emp /= np.sum(matrix_emp)

    return matrix_emp, values


if __name__ == "__main__":
    count = 10000
    matrix_th, a, b = init()
    gen = DTDRandomVariableGenerator(matrix_th)
    matrix_emp, values = generate_matrix_emp_and_values(gen)
    print(matrix_emp)

    create_histograms(matrix_th, matrix_emp)

    # Mathematical Expectations
    me_th_a = ProbabilitesTheory.mathemactical_expectation(a, np.sum(matrix_th, axis=1))
    me_th_b = ProbabilitesTheory.mathemactical_expectation(b, np.sum(matrix_th, axis=0))
    me_emp_a = ProbabilitesTheory.mathemactical_expectation(a, np.sum(matrix_emp, axis=1))
    me_emp_b = ProbabilitesTheory.mathemactical_expectation(b, np.sum(matrix_emp, axis=0))
    print("Mathematical expectation:")
    print(f"Theoretical: M[A] = {me_th_a}; M[B] = {me_th_b}")
    print(f"Emperical: M[A] = {me_emp_a}; M[B] = {me_emp_b}\n")

    # Dispersions
    d_th_a = ProbabilitesTheory.dispersion(a, np.sum(matrix_th, axis=1))
    d_th_b = ProbabilitesTheory.dispersion(b, np.sum(matrix_th, axis=0))
    d_emp_a = ProbabilitesTheory.dispersion(a, np.sum(matrix_emp, axis=1))
    d_emp_b = ProbabilitesTheory.dispersion(b, np.sum(matrix_emp, axis=0))
    print("Dispersion:")
    print(f"Theoretical: D[A] = {d_th_a}; D[B] = {d_th_a}")
    print(f"Emperical: D[A] = {d_emp_a}; D[B] = {d_th_b}\n")

    # Correlation and covariance coefficients:
    correlation_th = ProbabilitesTheory.correlation(matrix_th, a, b)
    correlation_emp = ProbabilitesTheory.correlation(matrix_emp, a, b)
    covariance_th = ProbabilitesTheory.covariance(matrix_th, a, b)
    covariance_emp = ProbabilitesTheory.covariance(matrix_emp, a, b)
    print("Covariance and Correlation:")
    print(f"Theoretical: cov(a,b) = {covariance_th}; r(x,y) = {correlation_th}")
    print(f"Emperical: cov(a,b) = {covariance_emp}; r(x,y) = {correlation_emp}\n")

    # Mathematical Expectation intervals
    me_intervals_a = ProbabilitesTheory.mathematical_expectation_interval(count, [value[0] for value in values])
    me_intervals_b = ProbabilitesTheory.mathematical_expectation_interval(count, [value[1] for value in values])
    print(f"Mathematical Expecatation intervals:")
    print(f"A: {me_intervals_a}")
    print(f"B: {me_intervals_b}\n")

    # Dispersion intervals
    d_intervals_a = ProbabilitesTheory.dispersion_interval(count, [value[0] for value in values])
    d_intervals_b = ProbabilitesTheory.dispersion_interval(count, [value[1] for value in values])
    print("Dispersion intervals:")
    print(f"A: {d_intervals_a}")
    print(f"B: {d_intervals_b}\n")

    # Pirson
    pirson = ProbabilitesTheory.pirson(matrix_emp, matrix_th, count)
    is_satisfying = ProbabilitesTheory.satisfies_theoretical(matrix_emp, matrix_th, count)
    print("Pirson criteria")
    print(f"Pirson coefficient: {pirson}; Is satisfying theoretical: {is_satisfying}")
