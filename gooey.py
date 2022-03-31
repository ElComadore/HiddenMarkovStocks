import datetime
import tkinter as tk

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


def get_stock_data():
    start_date = datetime.date(2010, 1, 1)

    stock = ent_ticker.get()
    try:
        stock_data = yf.download(stock, start=start_date, end=datetime.datetime.today(), interval='1d')
    except Exception as e:
        raise e
    return stock_data


def parse_stock_data(stock_data, start_pos: int, look_back: int):
    parsed_stock = np.array([stock_data['Close'].values[start_pos - look_back:start_pos]])

    for j in range(look_back):
        parsed_stock[0][j] = (stock_data['Close'].values[start_pos - look_back + j] - stock_data['Close'].values[
            start_pos - look_back + j - 1]) / stock_data['Close'].values[start_pos - look_back + j - 1]

    return parsed_stock


def predictor():
    lbl_message['text'] = "Nothing to report"
    try:
        stock_data = get_stock_data()
    except Exception as e:
        lbl_message['text'] = e
        return
    start_pos = -101
    start_pos_copy = start_pos
    look_back = 100

    wealth = 1
    corr_inc = 0
    corr_dec = 0
    miss_inc = 0
    miss_dec = 0
    no_dice = 0

    while start_pos < -1:
        parsed_stock = parse_stock_data(stock_data, start_pos=start_pos, look_back=look_back)

        states = 2
        a = np.array([[0.7, 0.3], [0.3, 0.7]])  # A init
        p = np.ones((states, 1)) / states  # pi init
        b = np.array([[0.8, 0.2], [0.2, 0.8]])  # B init

        tol = 0.00000000000000000000000000000001
        baum_welch(parsed_stock, a, b, p, tol)

        obs_prob = forward_algorithm(parsed_stock, a, b, p)
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
            if (stock_data['Close'].values[start_pos + 1] - stock_data['Close'].values[start_pos]) >= 0:
                corr_inc += 1
                lbl_corr_inc['text'] = f"#Correct guessed increase: {corr_inc}"
            else:
                miss_inc += 1
                lbl_miss_inc['text'] = f"#Guessed inc when dec: {miss_inc}"
            wealth *= 1 + (stock_data['Close'].values[start_pos + 1] - stock_data['Close'].values[start_pos])/stock_data['Close'].values[start_pos]

        elif found_one:
            if (stock_data['Close'].values[start_pos + 1] - stock_data['Close'].values[start_pos]) < 0:
                corr_dec += 1
                lbl_corr_dec['text'] = f"#Correct guessed decrease: {corr_dec}"
            else:
                miss_dec += 1
                lbl_miss_dec['text'] = f"#Guessed dec when inc: {miss_dec}"
        else:
            no_dice += 1
            lbl_no_dice['text'] = f"#Failed to predict: {no_dice}"
        print(start_pos)
        start_pos += 1


window = tk.Tk()
window.title("Stock Price Predictor")
window.resizable(width=True, height=True)

for i in range(3):
    window.rowconfigure(i, weight=1, minsize=10)
for j in range(2):
    window.columnconfigure(j, weight=1, minsize=50)

# Building the entry frame
frm_entry = tk.Frame(master=window)
lbl_ticker = tk.Label(master=frm_entry, text="Ticker:")
ent_ticker = tk.Entry(master=frm_entry, width=8)
lbl_message = tk.Label(master=frm_entry, text="Nothing to report")

# Positioning in the entry frame
lbl_ticker.grid(row=0, column=0, sticky="e")
ent_ticker.grid(row=0, column=1, sticky="w")
lbl_message.grid(row=1, sticky="w")

# Building the central data table
# First the prediction stats
frm_stats = tk.Frame(master=window)
lbl_stats = tk.Label(master=frm_stats, text="Prediction Stats")
lbl_corr_inc = tk.Label(master=frm_stats, text="#Correctly guessed increase: -")
lbl_corr_dec = tk.Label(master=frm_stats, text="#Correctly guessed decrease: -")
lbl_miss_inc = tk.Label(master=frm_stats, text="#Guessed inc when dec: -")
lbl_miss_dec = tk.Label(master=frm_stats, text="#Guessed dec when inc: -")
lbl_no_dice = tk.Label(master=frm_stats, text="#Failed to predict: -")

# Positioning the above
lbl_stats.grid(row=0)
lbl_corr_inc.grid(row=1, sticky="w")
lbl_corr_dec.grid(row=2, sticky="w")
lbl_miss_inc.grid(row=3, sticky="w")
lbl_miss_dec.grid(row=4, sticky="w")
lbl_no_dice.grid(row=5, sticky="w")

# Next the wealth evolution
frm_wealth = tk.Frame(master=window)
lbl_wealth_evo = tk.Label(master=frm_wealth, text="Wealth Evolution")
lbl_final_wealth = tk.Label(master=frm_wealth, text="Final wealth: -")
lbl_market_return = tk.Label(master=frm_wealth, text="Market return: -")

# Positioning the above
lbl_wealth_evo.grid(row=0)
lbl_final_wealth.grid(row=1, sticky="w")
lbl_market_return.grid(row=2, sticky="w")

# Other elements in the window
btn_predict = tk.Button(master=window, text="Predict!", command=predictor)
lbl_buy_sell = tk.Label(master=window, text="Waiting...")

# Positioning in the window
lbl_buy_sell.grid(row=0, column=1)
frm_stats.grid(row=1, column=0, padx=10)
frm_wealth.grid(row=1, column=1, padx=10)
btn_predict.grid(row=2, column=1, padx=10)
frm_entry.grid(row=2, column=0, sticky="e", padx=10)

window.mainloop()
