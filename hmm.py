import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd


def print2d(matrix, col, ind):
    df = pd.DataFrame(matrix, columns=col)
    df.index = ind
    print(df)


def plot_state(s1, s2, s1t, s2t):
    x = np.linspace(-3, 7, 200)
    plt.plot(x, s1, label='State 1')
    plt.plot(x, s2, label='State 2')
    plt.plot(x, s1t, '--', label='State 1(Training)')
    plt.plot(x, s2t, '--', label='State 2(Training)')
    plt.xlabel('obs')
    plt.ylabel('P(obs)')
    plt.title('Output probability density functions')
    plt.legend()
    plt.show()


def hmm_back(state_tran, b):
    m, n = state_tran.shape
    a = state_tran[1:m - 1, 1:n - 1]
    T = len(b[1])  # time
    n1 = len(a[1])  # branch
    n2 = len(a)  # state_j
    beta = np.zeros((T, n2))
    for i in range(n2):
        beta[T - 1][i] = state_tran[i + 1, n - 1].transpose()  # initialing beta_T(i)
    for t in range(T - 2, -1, -1):  # time t
        for si in range(n1):  # branch si
            beta[t][si] = sum(
                np.multiply(np.multiply(a[si, 0:n2].transpose(), b[:, t + 1]), beta[t + 1, 0:n2].transpose()))
            # beta_t(i) = sum[a_ij * b_j(t + 1) * beta_t + 1(j)]
    pb = sum(np.multiply(np.multiply(state_tran[0, 1:n2 + 1].transpose(), b[:, 0]), beta[0, :].transpose()))
    return beta, pb


def hmm_forw(state_tran, b):
    m, n = state_tran.shape
    a = state_tran[1:m - 1, 1:n - 1]
    T = len(b[1])  # time
    n1 = len(a[1])  # branch
    n2 = len(a)  # state_j
    alpha = np.zeros((T, n1))
    for i in range(n1):
        alpha[0][i] = state_tran[0][i + 1] * b[i][0]  # initialing beta_T(i)
    for t in range(1, T):  # time t
        for si in range(n1):
            alpha[t][si] = sum(np.multiply(a[:, si].transpose(), alpha[t - 1, 0:n2])) * b[si][t]
            # alpha_t(i)=sum[a_t-1(i)*alpha(ij)]*bj(t)
    pf = 0  # overall forward possibility
    for i in range(n2):
        pf = pf + alpha[T - 1, i] * state_tran[1 + i, n - 1]
    return alpha, pf


def reestimate(B, obs, gama, u_p):
    u = np.zeros(len(B[:, 1]))
    var = np.zeros(len(B[:, 1]))
    for i in range(len(B[:, 1])):
        u[i] = sum(np.multiply(gama[:, i], obs.transpose())) / sum(gama[:, i])
        var[i] = sum(np.multiply(gama[:, i], np.power((obs.transpose() - u_p[i]), 2))) / sum(gama[:, i])
    return u, var


def main():
    x = np.linspace(-3, 7, 200)
    u_p = [[1.0], [4.0]]
    state_tran = np.array([[0, 0.44, 0.56, 0], [0, 0.92, 0.06, 0.02], [0, 0.04, 0.93, 0.03], [0, 0, 0, 0]])
    obs = np.array([3.8, 4.2, 3.4, -0.4, 1.9, 3.0, 1.6, 1.9, 5.0])

    state_1 = norm(u_p[0][0], np.sqrt(1.44)).pdf(x)  # output probability density state 1
    state_2 = norm(u_p[1][0], np.sqrt(0.49)).pdf(x)  # output probability density state 2

    b1 = np.zeros(9)
    b2 = np.zeros(9)

    for k in range(9):
        b1[k] = (1 / (np.sqrt(1.44) * np.sqrt(2 * np.pi))) * np.exp(-np.power((obs[k] - 1.0), 2) / (2 * 1.44))
        b2[k] = (1 / (np.sqrt(0.49) * np.sqrt(2 * np.pi))) * np.exp(-np.power((obs[k] - 4.0), 2) / (2 * 0.49))
    b = np.array([b1, b2])

    col_t = ["Time " + str(i) for i in range(1, 10)]
    state = ['State 1', 'State 2']

    print('Output probability density functions b_i(o_t):')
    print2d(b, col_t, state)

    alpha, pf = hmm_forw(state_tran, b)
    beta, pb = hmm_back(state_tran, b)
    gama = np.divide(np.multiply(alpha, beta), pf)

    print("\n Overall likelihood of the observations: ", pf, "(Forward)")
    print("\n Forward likelihoods of the observation: \n")
    print2d(alpha, state, col_t)
    print("\n Overall likelihood of the observations: ", pb, "(Backward)")
    print("\n Backward likelihoods of the observation: \n")
    print2d(beta, state, col_t)
    print("\n Occupation likelihood: \n")
    print2d(gama, state, col_t)

    x, y = b.shape
    B = np.zeros((x, y))

    for j in range(x):
        for k in range(y):
            B[j][k] = gama[k][j] / sum(gama[:, j])

    u, var = reestimate(B, obs, gama, u_p)

    x = np.linspace(-3, 7, 200)
    state_11 = norm(u[0], np.sqrt(var[0])).pdf(x)
    state_22 = norm(u[1], np.sqrt(var[1])).pdf(x)

    x, y = state_tran.shape
    x = x - 2
    y = y - 2

    T = len(b[1, :])
    epsilon = np.zeros((x, y, T))
    a = state_tran[1:x + 1, 1:y + 1]
    p = pf

    for t in range(1, T):
        for i in range(x):
            for j in range(y):
                epsilon[i][j][t] = (alpha[t - 1, i] * a[i, j] * b[j, t] * beta[t, j]) / p

    A = np.zeros((x + 2, y + 2))
    for i in range(x):
        for j in range(y):
            A[i + 1, j + 1] = sum(epsilon[i, j, 1:T]) / sum(gama[:, i])

    A[0, 1: 2 + len(gama[1, :]) - 1] = gama[1, :]
    A[2, len(A) - 1] = 1 - sum(A[2, :])
    A[3, len(A) - 1] = 1 - sum(A[3, :])

    print('\n Output probability density functions (after training) b_i(o_t):')
    print2d(B, col_t, state)
    print("\n Re-estimation transition likelihood: ")
    for t in range(T):
        print('\nT=', t)
        print2d(epsilon[:, :, t], state, state)
    print("\n Re-estimated A matrix: \n")
    print(A)
    mv = ['Mean', 'Variance']
    print("\n Re-estimated mean and variance matrix: \n")
    print2d(np.array([u, var]), state, mv)
    plot_state(state_1, state_2, state_11, state_22)


if __name__ == "__main__":
    main()
