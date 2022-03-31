import matplotlib.pyplot as plt
from matplotlib import rc
import yfinance as yf
import datetime


def readInFloat(name):
    with open(name) as f:
        array = []
        for line in f:
            array.append(float(line))
    return array


def linePlot(n, r1, r2):
    start_date = datetime.date(2010, 1, 1)
    google = yf.download('googl', start=start_date, end=datetime.datetime.today(), interval='1d')
    tesla = yf.download('tsla', start=start_date, end=datetime.datetime.today(), interval='1d')
    start_pos = -n

    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)

    fig, ax = plt.subplots(2, 1)

    ax[0].plot(range(start_pos, 0, 1), r1, label="Predicted", color='royalBlue')
    ax[0].plot(range(start_pos, 0, 1), google['Close'].values[start_pos-1:-1], label="Actual", color="orange")
    ax[0].set_title("Google's Stock", fontsize=28)
    ax[0].set_xlabel("Number of days from the present", fontsize=22)
    ax[0].set_ylabel("Value in USD, \$", fontsize=22)
    ax[0].legend(fontsize=14)
    ax[0].tick_params(axis='both', which='major', labelsize=16)

    ax[1].plot(range(start_pos, 0, 1), r2, label="Predicted", color='royalBlue')
    ax[1].plot(range(start_pos, 0, 1), tesla['Close'].values[start_pos-1:-1], label="Actual", color="orange")
    ax[1].set_title("Tesla's Stock", fontsize=28)
    ax[1].set_xlabel("Number of days from the present", fontsize=22)
    ax[1].set_ylabel("Value in USD, \$", fontsize=22)
    ax[1].legend(fontsize=14)
    ax[1].tick_params(axis='both', which='major', labelsize=16)
    plt.show()


google_predict = readInFloat("googl.txt")
tesla_predict = readInFloat("tsla.txt")

linePlot(len(google_predict), google_predict, tesla_predict)