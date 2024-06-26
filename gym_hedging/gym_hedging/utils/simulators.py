import numpy as np
# from tqdm.notebook import tqdm
# import tensorflow as tf
import matplotlib.pyplot as plt
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.optimizers import Adam
# from keras.layers import Activation
# from keras import backend as keras_backend
# from keras.utils.generic_utils import get_custom_objects
# from tensorflow.keras.utils import get_custom_objects
# from tensorflow.keras import regularizers

from scipy.stats import norm



def BSM_delta_call(asset_price, strike_price, volatility, duration, return_rate=0):
    d1 = (
        np.log(asset_price / strike_price)
        + (return_rate + (volatility ** 2) / 2.) * duration
        ) / (volatility * np.sqrt(duration))
    # print(d1)
    # print("after d1")
    return norm.cdf(d1)

def simple_delta(asset_price1, asset_price2, option_price1, option_price2, volatility):
  delta_numerical = (option_price2-option_price1)/(asset_price2-asset_price1)
  # delta_naive = delta_numerical * ((volatility * asset_price1) / (volatility * asset_price2))
  return delta_numerical


class GBMSimulator():
    def __init__(self, dt, drift, volatility, s_0):
        """
        Args:
            dt (float): Difference in time between samples. Unit is years.
            drift (float): Mu parameter of differential equation, constant.
            volatility (float): Sigma parameter of differential equation, constant.
            s_0 (float): Initial price value.
        """
        self._price = s_0
        self.dt = dt
        self.drift = drift
        self.volatility = volatility
        self.s_0 = s_0
        self.duration = 0.
        self.history = []

    def __call__(self, n_samples=1):
        return self.generate(n_samples=n_samples)

    def generate(self, n_samples=1):
        """
        Generates an array of price samples following geometric brownian motion.
        https://en.wikipedia.org/wiki/Geometric_Brownian_motion

        Args:
            n_samples (int): Number of price samples.

        Returns:
            array[float]: Array of sampled price values.
        """
        s_t = np.exp(
            (self.drift - (self.volatility ** 2) / 2.) * self.dt
            + self.volatility * np.random.normal(0., np.sqrt(self.dt), n_samples))
        s_t = self._price * np.cumprod(s_t)
        self._price = s_t[-1]
        self.duration += self.dt * n_samples
        for sample in s_t:
            # print(f"Printing sameple {sample}")
            self.history.append(sample)
        return s_t

    def get_current_price(self):
        return self._price

    def reset(self):
        self._price = self.s_0
        self.duration = 0.
        self.history = []

    def plot_history(self):
        """
        Plots array of prices over a duration of time.
        """
        if len(self.history) > 0:
            domain = np.linspace(0., self.duration, len(self.history))
            plt.plot(domain, self.history)
            plt.xlabel("Time")
            plt.ylabel("Asset Price")
            plt.title("Asset Price over Time - GBM")


class BinomialTreeOptionSimulator():
    def __init__(
            self,
            initial_asset_price,
            strike_price,
            risk_free_interest_rate,
            volatility,
            T,
            dt,
            n_samples=100,
            option_type='call'
    ):
        self.initial_asset_price = initial_asset_price
        self.strike_price = strike_price
        self.risk_free_interest_rate = risk_free_interest_rate
        self.volatility = volatility
        self.initial_T = T
        self.T = T
        self.dt = dt
        self.n_samples = n_samples
        self.option_type = option_type
        self.price = None
        self.duration = 0
        self.history = []

        if self.option_type not in ['call', 'put']:
                raise ValueError("Option type must be either 'call' or 'put'")

        self.compute_price([initial_asset_price])
        self.AmericanDelta = 0


    def __call__(self, asset_prices):
        return self.compute_price(asset_prices)

    def compute_delta(self, asset_price, h=0.01):
      asset_price1 = asset_price + h
      # print(asset_price1)
      option_price1 = self.compute_price([asset_price1], stay_at_current_step=True)
      # print(option_price1)
      asset_price2 = asset_price - h
      option_price2 = self.compute_price([asset_price2], stay_at_current_step=True)
      delta_numerical = (option_price2-option_price1)/(asset_price2-asset_price1)
      print(delta_numerical)
      return delta_numerical




    def compute_price(self, asset_prices, stay_at_current_step=False):

        """
        This function uses the binomial tree approximation approach to calculate the price of an American option.

        Parameters:
        asset_prices (list[float]): The price of the underlying asset across timesteps.

        Returns:
        float: The final price of the American option.
        """
        if stay_at_current_step:
          self.duration -= 1  # Stay at the current time step
          self.T += self.dt

        for asset_price in asset_prices:
            # Calculate delta t and u
            T = self.T - (self.dt * self.duration)
            # print(T)
            delta_t = T / self.n_samples
            # print(f"This is delta_t {delta_t}")
            u = np.exp(self.volatility * np.sqrt(delta_t))

            # Calculate d
            d = 1 / u

            # Calculate p
            p = (np.exp(self.risk_free_interest_rate * delta_t) - d) / (u - d)

            # Initialize arrays for stock prices and option values
            stock_prices = np.zeros((self.n_samples+1, self.n_samples+1))
            option_values = np.zeros((self.n_samples+1, self.n_samples+1))

            # Calculate stock prices at each node
            for i in range(self.n_samples+1):
                for j in range(i+1):
                    stock_prices[j, i] = asset_price * (u ** (i-j)) * (d ** j)

            # Calculate option values at final node
            if self.option_type == 'call':
                option_values[:, self.n_samples] = np.maximum(stock_prices[:, self.n_samples] - self.strike_price, 0)
            else:
                option_values[:, self.n_samples] = np.maximum(self.strike_price - stock_prices[:, self.n_samples], 0)

            # Calculate option values at earlier nodes
            for i in range(self.n_samples-1, -1, -1):
                for j in range(i+1):
                    if self.option_type == 'call':
                        option_values[j, i] = np.maximum(stock_prices[j, i] - self.strike_price,
                                                        np.exp(-self.risk_free_interest_rate * delta_t) * (p * option_values[j, i+1] + (1-p) * option_values[j+1, i+1]))
                    else:
                        option_values[j, i] = np.maximum(self.strike_price - stock_prices[j, i],
                                                        np.exp(-self.risk_free_interest_rate * delta_t) * (p * option_values[j, i+1] + (1-p) * option_values[j+1, i+1]))

            # Return option price at time 0
            self.duration += 1
            option_value = option_values[0, 0]
            self.history.append(option_value)
            self.price = option_value
        # print(f"This is the option price {self.price}")
        return self.price

    def get_current_price(self):
        return self.price

    def payoff(self, option_price, K):
      return(np.maximum(K-option_price, 0))

    def reset(self):
        self.duration = 0
        self.history.clear()
        self.compute_price([self.initial_asset_price])

    def plot_history(self):
        """
        Plots array of prices over a duration of time.
        """
        if len(self.history) > 0:
            domain = np.linspace(0., self.duration, len(self.history))
            plt.plot(domain, self.history)
            plt.xlabel("Time")
            plt.ylabel("Option Price")
            plt.title("American Option Price over Time")


class BSMSimulator():
    def __init__(
            self,
            initial_asset_price,
            strike_price,
            risk_free_interest_rate,
            volatility,
            T,
            dt
        ):
        """
        Args:
            strike_price (float): Strike price of the option.
            risk_free_interest_rate (float): Interest rate for risk-free
                investment. Essentially, how fast your money would grow if
                uninvested.
            volatility (float): Sigma parameter of delta equation, constant.
            T (float): Initial time remaining until option maturity.
            dt (float): Difference in time between samples. Unit is years.
        """
        self.initial_asset_price = initial_asset_price
        self.strike_price = strike_price
        self.risk_free_interest_rate = risk_free_interest_rate
        self.volatility = volatility
        self.T = T
        self.dt = dt
        self.duration = 0
        self.price = None
        self.history = []

        self.compute_price([initial_asset_price])

    def __call__(self, asset_prices):
        return self.compute_price(asset_prices=asset_prices)

    def compute_price(self, asset_prices):
        """
        Takes an array of asset prices and calculates the option price for
        each, stepping by dt along the remaining duration each time.

        Args:
            asset_prices (array[float]): Array of underlying asset prices. All
                prices refer to the same asset, they are just taken at
                sequential steps in time.

        Returns:
            array[float]: Option prices corresponding to input asset prices.
        """
        for asset_price in asset_prices:
            self.duration += self.dt
            time_to_maturity = self.T - self.duration

            if time_to_maturity < 1e-7:
                option_price = max(0, asset_price - self.strike_price)
                self.history.append(option_price)
                continue
            d1 = BSM_delta_call(
                asset_price=asset_price,
                strike_price=self.strike_price,
                volatility=self.volatility,
                duration=time_to_maturity
            )
            d2 = d1 - self.volatility * np.sqrt(time_to_maturity)
            pvk = self.strike_price * np.exp(
                -self.risk_free_interest_rate * time_to_maturity
            )
            option_price = norm.cdf(d1) * asset_price - norm.cdf(d2) * pvk
            self.history.append(option_price)
            self.price = option_price
        return np.maximum(0, option_price)

    def reset(self):
        self.duration = 0
        self.history.clear()
        self.compute_price([self.initial_asset_price])

    def plot_history(self):
        """
        Plots array of prices over a duration of time.
        """
        if len(self.history) > 0:
            domain = np.linspace(0., self.duration, len(self.history))
            plt.plot(domain, self.history)
            plt.xlabel("Time")
            plt.ylabel("Option Price")
            plt.title("European Option Price over Time")




class BSM_call_option:
    @staticmethod
    def payoff(x, K):
        return np.maximum(x - K, np.zeros(x.shape))

    @staticmethod
    def price(x, ttm, sigma, K, r):
        if np.isclose(ttm, 0):
            price = BSM_call_option.payoff(x, K)
        else:
            d_1 = (np.log(x / K) + (r + 0.5 * (sigma ** 2)) * ttm) / (sigma * np.sqrt(ttm))
            d_2 = d_1 - sigma * np.sqrt(ttm)
            first_term = x * norm.cdf(d_1)
            second_term = K * np.exp(-r * ttm) * norm.cdf(d_2)
            price = first_term - second_term
        return price

    @staticmethod
    def delta(x, ttm, sigma, K, r):
        # changed this
        d_1 = (np.log(x / K) + (r + 0.5*(sigma ** 2)) * (ttm)) / (sigma * np.sqrt(ttm))
        delta = norm.cdf(d_1)
        # return [np.squeeze(delta)]
        return [delta]


    @staticmethod
    def exp_payoff(x, ttm, sigma, mu, K, n_samples):
        mean = (mu - 0.5 * (sigma ** 2)) * ttm
        std = sigma * np.sqrt(ttm)
        Y = np.random.normal(mean, std, size=n_samples)
        S_T = x * np.exp(Y)
        payoff = BSM_call_option.payoff(S_T, K)
        return np.mean(payoff)