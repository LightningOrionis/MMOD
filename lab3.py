import matplotlib.pyplot as plt
import numpy as np


def factorial(n: int) -> int:
    res = 1
    for i in range(2, n + 1):
        res *= i
    return res


def generate_p(n: int, m: int, ro: float, betta: float) -> list:
    p = []
    summa = 0

    p0 = sum([ro ** i / factorial(i) for i in range(0, n + 1)])

    for i in range(1, m + 1):
        summa += ro ** i / np.prod([n + l * betta for l in range(1, i + 1)])

    p0 += summa * ro ** n / factorial(n)
    p0 = p0 ** -1
    p.append(p0)

    for k in range(1, n + 1):
        p.append(ro ** k / factorial(k) * p0)

    for i in range(1, m + 1):
        p_i = p[n] * (ro ** i / np.prod([n + l * betta for l in range(1, i + 1)]))
        p.append(p_i)

    return p


def generate_requests(time_max: float, lambda_: int) -> list:
    requests = []
    t = 0
    while t < time_max:
        t += np.random.exponential(lambda_ ** -1)
        requests.append(t)
    return requests


def generate_p_emp(n: int, m: int, mu: int, v: int, lambda_: int, max_time: int) -> tuple:
    system_states = [0 for i in range(0, max_time)]
    system_states_time = [0 for i in range(0, max_time)]
    current_time = 0
    queue = []
    smo = []
    requests = generate_requests(max_time, lambda_)
    p_emp = [0 for i in range(0, n + m + 1)]
    requests_count = len(requests)
    unprocessed_count = 0

    while current_time < max_time:
        request_min = min(requests)
        queue_min = -1 if len(queue) == 0 else min(queue)
        smo_min = -1 if len(smo) == 0 else min(smo)
        to_find = [request_min, queue_min, smo_min]
        min_value = min([value for value in to_find if value != -1])

        p_emp[len(smo) + len(queue)] += min_value
        p_emp[len(smo) + len(queue)] -= current_time

        if min_value == request_min:
            requests.remove(min_value)

            if len(smo) < n:
                smo.append(min_value + np.random.exponential(mu ** -1))
            elif len(queue) < m:
                queue.append(min_value + np.random.exponential(v ** -1))
            else:
                unprocessed_count += 1

        if min_value == smo_min:
            smo.remove(min_value)
            if len(queue) != 0:
                queue.pop(0)
                smo.append(min_value + np.random.exponential(mu ** -1))

        if min_value == queue_min:
            queue.remove(min_value)

        current_time = min_value
        system_states_time.append(current_time)
        system_states.append(len(smo) + len(queue))

    A_emp = (requests_count - unprocessed_count) / max_time
    p_emp_n = [p / max_time for p in p_emp]

    plt.plot(list(system_states_time), system_states)
    plt.xlabel("time")
    plt.ylabel("state")
    plt.show()

    return p_emp_n, A_emp


def plot(p: list, p_emp: list, n: int, m: int, lambda_: int, mu: int) -> None:
    plt.plot(list(range(n + m + 1)), p)
    plt.plot(list(range(n + m + 1)), p_emp)
    plt.legend(['theoretical', 'emperical'])
    plt.title(f"n = {n}, m = {m}, lambda_ = {lambda_}, mu = {mu}")
    plt.show()


if __name__ == '__main__':
    n = 1
    m = 10
    lambda_ = 4
    mu = 2
    ro = lambda_ / mu
    v = 4  
    betta = v / mu
    max_time = 150

    p = generate_p(n, m, ro, betta)
    p_emp, _ = generate_p_emp(n, m, mu, v, lambda_, max_time)
    print(f"Теоретическое: {p[0]}, {p[1]}, {p[2]}")
    print(f"Эмпирическое: {p_emp[0]}, {p_emp[1]} {p_emp[2]}")

    print(f"Число каналов: {n}, мест в очереди: {m}, интенсивность потока заявок: {lambda_}, интенсивность потока обслуживания: {mu}, параметр v: {v}, ограничение пребывания заявки в очереди: {max_time}")

    plot(p, p_emp, n, m, lambda_, mu)
