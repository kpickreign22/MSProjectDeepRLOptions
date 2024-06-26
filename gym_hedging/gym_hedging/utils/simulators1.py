import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import scipy.stats as ss



def longstaff_schwartz(paths, strike, r, option_type):
    cash_flows = np.zeros_like(paths)
    if option_type == "Call":
        for i in range(0,cash_flows.shape[0]):
            cash_flows[i] = [max(round(x - strike,2),0) for x in paths[i]]
    else:
        for i in range(0,cash_flows.shape[0]):
            cash_flows[i] = [max(-round(x - strike,2),0) for x in paths[i]]
    discounted_cash_flows = np.zeros_like(cash_flows)


    T = cash_flows.shape[0]-1

    for t in range(1,T):
        
        # Look at time t+1
        # Create index to only look at in the money paths at time t
        in_the_money =paths[t,:] < strike

        # Run Regression
        X = (paths[t,in_the_money])
        X2 = X*X
        Xs = np.column_stack([X,X2])
        Y = cash_flows[t-1,in_the_money]  * np.exp(-r)
        model_sklearn = LinearRegression()
        model = model_sklearn.fit(Xs, Y)
        conditional_exp = model.predict(Xs)
        continuations = np.zeros_like(paths[t,:])
        continuations[in_the_money] = conditional_exp

        # # First rule: If continuation is greater in t =0, then cash flow in t=1 is zero
        cash_flows[t,:] = np.where(continuations> cash_flows[t,:], 0, cash_flows[t,:])

        # 2nd rule: If stopped ahead of time, subsequent cashflows = 0
        exercised_early = continuations < cash_flows[t, :]
        cash_flows[0:t, :][:, exercised_early] = 0
        discounted_cash_flows[t-1,:] = cash_flows[t-1,:]* np.exp(-r * 3)

    discounted_cash_flows[T-1,:] = cash_flows[T-1,:]* np.exp(-r * 1)


    # Return final option price
    final_cfs = np.zeros((discounted_cash_flows.shape[1], 1), dtype=float)
    for i,row in enumerate(final_cfs):
        final_cfs[i] = sum(discounted_cash_flows[:,i])
    option_price = np.mean(final_cfs)
    return option_price


def simulate_gbm(mu, sigma, S0, T, dt, num_paths):
    num_steps = int(T / dt) + 1
    times = np.linspace(0, T, num_steps)
    paths = np.zeros(( num_steps,num_paths))

    for i in range(num_paths):
        # Generate random normal increments
        dW = np.random.normal(0, np.sqrt(dt), num_steps - 1)
        # Calculate the cumulative sum of increments
        cumulative_dW = np.cumsum(dW)
        # Calculate the stock price path using the GBM formula
        paths[ 1:,i] = S0 * np.exp((mu - 0.5 * sigma**2) * times[1:] + sigma * cumulative_dW)
    return paths



class LSM():
    def __init__(self):
        self.r = 0.1  # interest rate
        self.sig = 0.2  # diffusion coefficient
        self.S0 = 100  # current price
        self.K =  100 # strike
        self.T =  1 # maturity in years
        self.payoff = "put"
        pass


    def LSM(self, N=10000, paths=10000, order=2):
        """
        Longstaff-Schwartz Method for pricing American options

        N = number of time steps
        paths = number of generated paths
        order = order of the polynomial for the regression
        """

        if self.payoff != "put":
            raise ValueError("invalid type. Set 'call' or 'put'")

        dt = self.T / (N - 1)  # time interval
        df = np.exp(-self.r * dt)  # discount factor per time time interval

        X0 = np.zeros((paths, 1))
        increments = ss.norm.rvs(
            loc=(self.r - self.sig**2 / 2) * dt,
            scale=np.sqrt(dt) * self.sig,
            size=(paths, N - 1),
        )
        X = np.concatenate((X0, increments), axis=1).cumsum(1)
        S = self.S0 * np.exp(X)

        H = np.maximum(self.K - S, 0)  # intrinsic values for put option
        V = np.zeros_like(H)  # value matrix
        V[:, -1] = H[:, -1]

        # Valuation by LS Method
        for t in range(N - 2, 0, -1):
            good_paths = H[:, t] > 0
            rg = np.polyfit(S[good_paths, t], V[good_paths, t + 1] * df, 2)  # polynomial regression
            C = np.polyval(rg, S[good_paths, t])  # evaluation of regression

            exercise = np.zeros(len(good_paths), dtype=bool)
            exercise[good_paths] = H[good_paths, t] > C

            V[exercise, t] = H[exercise, t]
            V[exercise, t + 1 :] = 0
            discount_path = V[:, t] == 0
            V[discount_path, t] = V[discount_path, t + 1] * df

        V0 = np.mean(V[:, 1]) * df  #
        return V0

np.random.seed(99)
# Parameters
mu = 0.00  # Drift (average return per unit time)
sigma = 0.2  # Volatility (standard deviation of the returns)
S0 = 100  # Initial stock price
T = 1  # Total time period (in years)
dt = 1/25000  # Time increment (daily simulation)
num_paths = 10000  # Number of simulation paths


K = 100.0  # strike
r = 0.1  # risk free rate
N = 25000  # number of periods or number of time steps
payoff = "put"  # payoff

dT = float(T) / N  # Delta t
u = np.exp(sigma * np.sqrt(dT))  # up factor
d = 1.0 / u  # down factor


V = np.zeros(N + 1)  # initialize the price vector
S_T = np.array([(S0 * u**j * d ** (N - j)) for j in range(N + 1)])  # price S_T at time T

a = np.exp(r * dT)  # risk free compound return
p = (a - d) / (u - d)  # risk neutral up probability
q = 1.0 - p  # risk neutral down probability

if payoff == "call":
    V[:] = np.maximum(S_T - K, 0.0)
elif payoff == "put":
    V[:] = np.maximum(K - S_T, 0.0)

for i in range(N - 1, -1, -1):
    V[:-1] = np.exp(-r * dT) * (p * V[1:] + q * V[:-1])  # the price vector is overwritten at each step
    S_T = S_T * u  # it is a tricky way to obtain the price at the previous time step
    if payoff == "call":
        V = np.maximum(V, S_T - K)
    elif payoff == "put":
        V = np.maximum(V, K - S_T)


BS = LSM()
print(BS.LSM(N=10000, paths=10000, order=2))

print("American BS Tree Price: ", V[0])





# Simulate stock price paths
paths = simulate_gbm(mu, sigma, S0, T, dt, num_paths)
paths[0,:]=S0
paths = paths[::-1]

print(longstaff_schwartz(paths = paths, strike =100, r = 0.1, option_type="Put"))