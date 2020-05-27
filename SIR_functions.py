from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from hyperopt import hp, fmin, tpe
from typing import Tuple


def SIR(t, y, N, kappa, tau, nu):
    """
    Expresses SIR model in initial value ODE format, including time,
    state variables, and parameters required to compute the derivatives

    Parameters
    ---------------
    t : array
        Independent variable
    y : array-like
        Initial state of system, [S, I, R]
    parameters : array-like
        Parameters of model [N, kappa, tau, nu]

    Returns
    ----------------
    f : array-like
        Array of derivatives for the system to be numerically integrated
    """
    S, I, R = y

    dSdt = - kappa * tau / N * S * I
    dIdt = (kappa * tau / N * S - nu) * I
    dRdt = nu * I
    f = [dSdt, dIdt, dRdt]
    return f


def create_SIR_curves(N: int, kappa: float, tau: float,
                      nu: float, start_date: object,
                      total_points: int) -> object:
    """
    Use parameters to solve system of ODEs as an initial value problem

    Parameters
    ----------
    N : int
        Size of population
    kappa : float
        Contacts per unit time individual has with rest of population
    tau : float [0, 1]
        Probability that contact leads to transmission
    nu : float [0, 1]
        Probability per unit time that infected host be removed from population
    total_points : int
        Total points from start for simulated solutions

    Returns
    -------
    df : object
    """

    S, I, R = N - 1, 1, 0
    sys0 = (S, I, R)
    start = 0
    t = np.linspace(start, stop, total_points)
    sol = solve_ivp(fun=SIR, t_span=(start, stop), y0=sys0, args=parameters, t_eval=t)
    df = pd.DataFrame(sol.y.T, columns=['S', 'I', 'R'])
    df['Date'] = pd.date_range(start_date, freq='D', periods=124)
    df['Predicted Confirmed'] = df['I'] + df['R']
    return df


def plot_SIR(parameters, df):
    N, kappa, tau, nu = parameters
    Re = kappa * tau / (nu)
    #df.plot(x='Time', y=df.columns.drop('Time'), figsize=(15, 10))
    plt.title('SIR')
    return fig

def model_loss(parameters):
    # initial state variables
    df = create_SIR_curves(parameters)
    loss = mean_squared_error(df['Cases'].values, df[df.columns[-1]].values)
    return loss

def prepare_data(data: object, location: str) -> Tuple(object, int):
    """
    Extract start date and length from dataframe

    Parameters
    ----------
    data : object
        Aggregated dataframe object with Confirmed, Death, Recovered etc. of
    location : str
        Name of location to filter

    Returns
    -------
    start_date : object
        Start datetime of first nonzero data point of infected
    num_days : int
        Number of days from start date to current reported data
    """
    start_date = data[data['Confirmed']!=0].head(1)['Date']
    num_days = len(data)
    data.fillna(0)




