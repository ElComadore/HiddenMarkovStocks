import datetime
import numpy as np
import yfinance as yf


def forward_algorithm(obs, a, b, p):
    """
    The forward algorithm for calculating the probability of the obs, given the lambda
    :param obs: the observations made
    :param a: the underlying transition matrix
    :param b: the observation matrix
    :param p: the initial distro
    :return: the probability of the obs given lambda
    """
    L = len(obs)
    N = len(p)
    T = len(obs[0])

    prob = 1

    for l in range(L):
        prob_obs = 0
        alpha = np.zeros((L, T, N))

        for i in range(N):
            b_obs = b[i][0]
            if obs[l][0] < 0:
                b_obs = b[i][1]

            alpha[l][0][i] = p[i] * b_obs

        for t in range(1, T):
            for j in range(N):
                b_obs = b[j][0]
                if obs[l][t] < 0:
                    b_obs = b[j][1]

                for i in range(N):
                    alpha[l][t][j] += alpha[l][t - 1][i] * a[i][j]

                alpha[l][t][j] *= b_obs

        for i in range(N):
            prob_obs = prob_obs + alpha[l][T - 1][i]

        prob = prob * prob_obs

    return prob


def baum_welch(obs, a, b, p, tol):
    """
    This is the parameter optimiser!
    :param obs: the observed data
    :param a: the underlying transition matrix
    :param b: the observation matrix
    :param p: the initial distro
    :param tol: the tolerance for the probability
    :return: nothing; mutates the params as it goes
    """
    L = len(obs)
    N = len(p)
    T = len(obs[0])

    def beta_generator():
        """Making the backwards probability variable beta"""
        beta = np.zeros((L, T, N))
        for l in range(L):
            for i in range(N):
                beta[l][T - 1][i] = 1
            for t in range(T - 2, -1, -1):
                for i in range(N):
                    for j in range(N):
                        b_obs = b[j][0]
                        if obs[l][t + 1] < 0:
                            b_obs = b[j][1]
                        beta[l][t][i] += a[i][j] * b_obs * beta[l][t + 1][j]
        return beta

    def zeta_generator(beta):
        """Making the filler variable zeta"""
        alpha = np.zeros((L, T, N))
        for l in range(L):
            for i in range(N):
                b_obs = b[i][0]
                if obs[l][0] < 0:
                    b_obs = b[i][1]
                alpha[l][0][i] = p[i] * b_obs

            for t in range(1, T):
                for j in range(N):
                    b_obs = b[j][0]
                    if obs[l][t] < 0:
                        b_obs = b[j][1]
                    for i in range(N):
                        alpha[l][t][j] += alpha[l][t - 1][i] * a[i][j]
                    alpha[l][t][j] *= b_obs

        zeta = np.zeros((L, T - 1, N, N))
        for l in range(L):
            for t in range(T - 1):
                for i in range(N):
                    for j in range(N):
                        denom = 0
                        for k in range(N):
                            for w in range(N):
                                b_den_obs = b[w][0]
                                if obs[l][t + 1] < 0:
                                    b_den_obs = b[w][1]
                                denom += alpha[l][t][k] * beta[l][t + 1][w] * a[k][w] * b_den_obs
                        b_num_obs = b[j][0]
                        if obs[l][t + 1] < 0:
                            b_num_obs = b[j][1]
                        zeta[l][t][i][j] = (alpha[l][t][i] * a[i][j] * b_num_obs * beta[l][t + 1][j]) / denom
            return zeta

    def gamma_generator(beta):
        """Making the filler variable gamma"""
        alpha = np.zeros((L, T, N))
        for l in range(L):
            for i in range(N):
                b_obs = b[i][0]
                if obs[l][0] < 0:
                    b_obs = b[i][1]
                alpha[l][0][i] = p[i] * b_obs

            for t in range(1, T):
                for j in range(N):
                    b_obs = b[j][0]
                    if obs[l][t] < 0:
                        b_obs = b[j][1]
                    for i in range(N):
                        alpha[l][t][j] += alpha[l][t - 1][i] * a[i][j]
                    alpha[l][t][j] *= b_obs

        gamma = np.zeros((L, T, N))
        for l in range(L):
            for t in range(T):
                for i in range(N):
                    denom = 0
                    for j in range(N):
                        denom += alpha[l][t][j] * beta[l][t][j]
                    gamma[l][t][i] = (alpha[l][t][i] * beta[l][t][i]) / denom

        return gamma
    # Here begins the optimisation loop
    delta = 1
    old_prob = forward_algorithm(obs, a, b, p)
    while delta > tol:

        beta_1 = beta_generator()
        zeta_1 = zeta_generator(beta_1)
        gamma_1 = gamma_generator(beta_1)

        for i in range(N):
            p[i] = 0

            for l in range(L):
                p[i] += gamma_1[l][0][i]

            p[i] = p[i] / L

            for j in range(N):
                numerator = 0
                denominator = 0

                for l in range(L):
                    for t in range(T - 1):
                        numerator += zeta_1[l][t][i][j]
                        denominator += gamma_1[l][t][i]

                a[i][j] = numerator / denominator

            for j in range(N):
                numerator = 0
                denominator = 0

                for l in range(L):
                    for t in range(T - 1):

                        if (obs[l][t] > 0) and (j == 0):
                            numerator += gamma_1[l][t][i]

                        elif (obs[l][t] < 0) and (j == 1):
                            numerator += gamma_1[l][t][i]

                        denominator += gamma_1[l][t][i]

                b[i][j] = numerator / denominator

        new_prob = forward_algorithm(obs, a, b, p)
        delta = abs(new_prob - old_prob)
        old_prob = new_prob


# Here is the where you choose the stock and time interval of interest
start_date = datetime.date(2010, 1, 1)
time_delta = datetime.timedelta(weeks=52 * 11)

stock = 'googl'
stock_data = yf.download(stock, start=start_date, end=datetime.datetime.today(), interval='1d')

start_pos = -51  # Where to start modeling
start_pos_copy = start_pos
look_back = 100  # How far to look back; m

correct_guess = 0
incorrect_guess = 0
no_guess = 0

predicted_price = list()
wealth = 1  # Initial wealth

while start_pos < -1:
    obs_stock = np.array([stock_data['Close'].values[start_pos - look_back:start_pos]])

    for j in range(look_back):
        obs_stock[0][j] = (stock_data['Close'].values[start_pos - look_back + j] - stock_data['Close'].values[
            start_pos - look_back + j - 1]) / stock_data['Close'].values[start_pos - look_back + j - 1]

    states = 2  # How many states in a
    a = np.array([[0.7, 0.3], [0.3, 0.7]])  # A init
    p = np.ones((states, 1)) / states  # pi init
    b = np.array([[0.8, 0.2], [0.2, 0.8]])  # B init
    tol = 0.00000000000000000000000000000001  # tolerance for BW; needs to be adjusted for look_back

    baum_welch(obs_stock, a, b, p, tol)  # Optimisation

    obs_prob = forward_algorithm(obs_stock, a, b, p)  # Never tell me the odds
    comp_prob = 1

    i = 0
    found_one = True

    while abs(obs_prob - comp_prob) > tol:  # Finding historical patch with similar prob
        i += 1
        comp_obs = np.array([stock_data['Close'].values[start_pos - look_back - i:start_pos - i]])

        for j in range(look_back):
            try:
                comp_obs[0][j - 1] = (stock_data['Close'].values[start_pos - look_back - i + j] -
                                      stock_data['Close'].values[start_pos - look_back - i + j - 1]) / \
                                     stock_data['Close'].values[start_pos - look_back - i + j - 1]
            except IndexError:
                found_one = False
                break
        if not found_one:
            break
        comp_prob = forward_algorithm(comp_obs, a, b, p)

    if (stock_data['Close'].values[start_pos - i + 1] - stock_data['Close'].values[start_pos - i]) >= 0 and found_one:
        r = 1 + (stock_data['Close'].values[start_pos - i + 1] - stock_data['Close'].values[start_pos - i]) / \
            stock_data['Close'].values[start_pos - i]
        predicted_price.append(stock_data['Close'].values[start_pos] * r)

        if (stock_data['Close'].values[start_pos + 1] - stock_data['Close'].values[start_pos]) >= 0:
            correct_guess += 1
            print('Correct, g')

        else:
            incorrect_guess += 1
            print('Incorrect, g')

        wealth *= 1 + (stock_data['Close'].values[start_pos + 1] - stock_data['Open'].values[start_pos + 1]) \
                  / stock_data['Open'].values[start_pos + 1]

    elif found_one:
        r = 1 + (stock_data['Close'].values[start_pos - i + 1] - stock_data['Close'].values[start_pos - i]) / \
            stock_data['Close'].values[start_pos - i]
        predicted_price.append(stock_data['Close'].values[start_pos] * r)

        if (stock_data['Close'].values[start_pos + 1] - stock_data['Close'].values[start_pos]) < 0:
            correct_guess += 1
            print('Correct, l')

        else:
            incorrect_guess += 1
            print('Incorrect, l')
        # wealth *= 1 - (stock_data['Close'].values[start_pos + 1] - stock_data['Close'].values[start_pos]) \
        #          / stock_data['Close'].values[start_pos]

    else:
        predicted_price.append(stock_data['Close'].values[start_pos])
        no_guess += 1
        print('No dice')

    start_pos += 1

# How did we do?
print("\nCorrect Guesses: " + str(correct_guess))
print("Incorrect Guesses: " + str(incorrect_guess))
print("Failed: " + str(no_guess) + "\n")

print(predicted_price)
print('\nWealth: ' + str(wealth))

m_r = 1 + (stock_data['Close'].values[-1] - stock_data['Close'].values[start_pos_copy]) \
      / stock_data['Close'].values[start_pos_copy]
print("\nMarket return of holding: " + str(m_r))

# A little bit of IO stuff
with open(stock + '.txt', "w") as f:
    f.write(str(predicted_price[0]))
    for p in predicted_price[1:]:
        f.write('\n' + str(p))

with open(stock + '_r.txt', 'w') as f:
    f.write('Correct Guesses: ' + str(correct_guess))
    f.write('\nIncorrect Guesses: ' + str(incorrect_guess))
    f.write("\nFailed: " + str(no_guess))
    f.write('\nWealth: ' + str(wealth))
    f.write('\nMarket Return: ' + str(m_r))
