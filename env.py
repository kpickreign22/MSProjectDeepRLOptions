import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.stats import norm

from gym_hedging.utils import GBMSimulator, BinomialTreeOptionSimulator, SimpleGBMSim, SimpleBTSim

class DeltaHedgingEnv(gym.Env):

    def __init__(
        self,
        T=1.,
        num_steps=100,
        s_0=100.,
        drift=0.,
        volatility=0.15,
        risk_free_interest_rate=0.,
        trading_cost_para=0.01,
        risk_aversion=1.0,
        strike_price=None,
        initial_holding=100,
        L=1 # Number of options contracts held, each for 100 shares
    ):
        super().__init__()

        self.T = T
        self.num_steps = num_steps
        self.dt = T / num_steps
        self.asset_price = s_0
        self.drift = drift
        self.volatility = volatility
        self.risk_free_interest_rate = risk_free_interest_rate
        self.trading_cost_para = trading_cost_para
        self.risk_aversion = risk_aversion
        self.strike_price = strike_price if strike_price is not None else s_0
        self.initial_holding = initial_holding
        self.holdings = initial_holding
        self.cash = 0.
        self.L = L
        self.target = 0

        # Initialize asset model
        self.asset_price_model = GeometricBrownianMotionSimulator(
            dt=self.dt,
            drift=self.drift,
            volatility=self.volatility,
            s_0=self.asset_price
        )
        # Initialize option model
        self.option_price_model = BinomialTreeOptionSimulator(
            initial_asset_price=s_0,
            strike_price=self.strike_price,
            risk_free_interest_rate=self.risk_free_interest_rate,
            volatility=self.volatility,
            T=self.T,
            dt=self.dt
        )
        self.initial_option_price = self.option_price_model.get_current_price()
        self.option_price = self.initial_option_price

        self.PnL = 0.
        self.portfolio_value = 0.
        
        # Action space: integer amount of the underlyer to hold
        # 100L + 1 actions for range [0, 100L] inclusive
        self.action_space = spaces.Discrete(100 * L + 1)

        # State space: current stock price, time to maturity, current holdings, strike price
        # self.observation_space = spaces.Tuple((
        #     spaces.Box(low=0., high=np.inf, shape=(1,), dtype=np.float32),
        #     spaces.Box(low=0., high=T, shape=(1,), dtype=np.float32),
        #     spaces.Box(low=0., high=1., shape=(1,), dtype=np.float32)
        # ))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)


    def step(self, action):
        # Convert action to amount of underlyer to trade
        prev_port_value = self.portfolio_value
        prev_asset_price = self.asset_price

        # delta = -100 * self.L + action
        delta = -action
        amt_change = delta - self.holdings
        self.holdings += amt_change
        transaction_cost = self.trading_cost_para * np.abs((amt_change)) * self.asset_price
        self.cash -= (amt_change * self.asset_price) + transaction_cost

        # Update market state and compute new wealth
        self.asset_price = self.asset_price_model()[-1]
        self.portfolio_value = self.cash + self.holdings * self.asset_price

        change_port_value = self.portfolio_value - prev_port_value
        self.target = self.target_delta(prev_asset_price)
        scaled_target_delta = -int((self.target * self.L * 100) + 0.5)
        delta_error = abs(delta - scaled_target_delta)

        self.option_price = self.option_price_model([self.asset_price])
        self.PnL = self.initial_option_price - self.option_price

        # Compute reward
        # reward = change_port_value - self.risk_aversion * delta_error
        reward = -self.risk_aversion * delta_error

        self.current_step += 1
        # End episode when option expires
        done = self.current_step >= self.num_steps - 1
        self.state = self.get_state()

        # Gymnasium env should return (state, reward, terminated, truncated, info)
        # truncated is always False because it is irrelevant for our situation
        # info is currently an empty dict but can contain keys and values if desired
        return self.state, reward, done, False, {}


    def target_delta(self, asset_price, h=0.1):
        asset_price1 = asset_price + h
        option_price1 = self.option_price_model.compute_price([asset_price1], stay_at_current_step=True)
        asset_price2 = asset_price - h
        option_price2 = self.option_price_model.compute_price([asset_price2], stay_at_current_step=True)
        delta_numerical = (option_price2 - option_price1) / (asset_price2 - asset_price1)
        return delta_numerical


    def compute_reward(self, change_wealth, change_option):
        # Compute the change in wealth
        PnL = change_wealth - change_option

        # Calculate the reward based on quadratic utility
        reward = PnL - (self.risk_aversion / 2.) * (PnL ** 2)
        return reward


    def get_state(self):
        # Construct state
        time_to_maturity = max(0., self.T - self.current_step * self.dt)
        # return np.array([
        #     self.asset_price / self.strike_price,
        #     time_to_maturity,
        #     self.holdings / (100 * self.L)
        # ], dtype=np.float32)
        return np.array([
            self.asset_price,
            time_to_maturity,
            self.holdings,
            self.strike_price
        ], dtype=np.float32)


    def render(self):
        print(f"Step: {self.current_step}")
        print(f"Current Price: {self.asset_price}")
        print(f"Time to Maturity: {self.T - self.current_step * self.dt}")
        print(f"Holdings: {self.holdings}")
        print(f"Cash: {self.cash}")
        print(f"Wealth: {self.portfolio_value}")
        print("=" * 30)


    def reset(self, seed=None, options=None):
        self.asset_price_model.reset()
        self.asset_price = self.asset_price_model.get_current_price()
        self.option_price_model.reset()
        self.option_price = self.option_price_model.get_current_price()
        self.holdings = self.initial_holding
        self.cash = 0.
        self.current_step = 0
        self.state = self.get_state()
        return self.state, {}
    
    def close(self):
        pass


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
        d_1 = (np.log(x / K) + (r + 0.5 * (sigma ** 2)) * (ttm)) / (sigma * np.sqrt(ttm))
        delta = norm.cdf(d_1)
        return [np.squeeze(delta)]

    @staticmethod
    def exp_payoff(x, ttm, sigma, mu, K, n_samples):
        mean = (mu - 0.5 * (sigma ** 2)) * ttm
        std = sigma * np.sqrt(ttm)
        Y = np.random.normal(mean, std, size=n_samples)
        S_T = x * np.exp(Y)
        payoff = BSM_call_option.payoff(S_T, K)
        return np.mean(payoff)


class ThesisHedgingEnv(gym.Env):
    def __init__(self, r, K, sigma, mu, S_0, Y_0, sigma_1, mu_1, corr, dt, T, kappa, transaction_cost):
        self.r = r
        self.K = K
        self.sigma = sigma
        self.mu = mu
        self.S_0 = S_0
        self.Y_0 = Y_0
        self.sigma_1 = sigma_1
        self.mu_1 = mu_1
        self.corr = corr
        self.dt = dt
        self.T = T
        self.kappa = kappa
        self.transaction_cost = transaction_cost

        self.action_space = spaces.Discrete(201)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        self.state_features = np.zeros((4), dtype=np.float32)
        self.state = np.zeros((6))
        self.set_stock_price(self.S_0)
        self.set_ttm(self.T)
        self.set_port_val(BSM_call_option.price(self.get_stock_price(self.state),
                                                self.get_ttm(self.state),
                                                self.sigma,
                                                self.K, self.r))
        self.set_moneyness(np.log(self.get_stock_price(self.state) / self.K))
        self.set_previous_delta(0)

    def generate_next_prices(self, S_t, Y_t):
        delta_W1 = np.sqrt(self.dt) * np.random.normal(0, 1)
        delta_W2 = np.sqrt(self.dt) * np.random.normal(0, 1)
        delta_W3 = self.corr * delta_W1 + np.sqrt(1 - (self.corr ** 2)) * delta_W2

        X_1 = np.exp(self.sigma * delta_W3 + ((self.mu - 0.5 * (self.sigma ** 2)) * self.dt))
        X_2 = np.exp(self.sigma_1 * delta_W1 + ((self.mu_1 - 0.5 * (self.sigma_1 ** 2)) * self.dt))

        next_stock_price = X_1 * S_t
        next_traded_price = X_2 * Y_t

        return next_stock_price, next_traded_price

    def reset(self, seed=None, options=None):
        self.state = np.zeros((6))
        self.set_stock_price(self.S_0)
        self.set_traded_price(self.Y_0)
        self.set_ttm(self.T)
        self.set_port_val(BSM_call_option.price(self.get_stock_price(self.state),
                                                self.get_ttm(self.state),
                                                self.sigma,
                                                self.K, self.r))
        self.set_moneyness(np.log(self.get_stock_price(self.state) / self.K))
        self.set_previous_delta(0)
        self.update_state_features(reset=True)

        return self.state_features, {}

    def step(self, action):
        # delta = 1. / (1. + np.exp(-action)) + max(0., 0.05 * action)
        delta = action / 100.
        prev_state = self.state.copy()
        prev_stock_price = self.get_stock_price(prev_state)
        prev_traded_price = self.get_traded_price(prev_state)
        next_stock_price, next_traded_price = self.generate_next_prices(prev_stock_price, prev_traded_price)
        prev_portfolio_value = self.get_port_val(prev_state)
        prev_delta = self.get_previous_delta(prev_state)

        self.set_stock_price(next_stock_price)
        self.set_traded_price(next_traded_price)
        self.set_ttm(self.get_ttm(prev_state) - self.dt)

        transaction_costs = self.transaction_cost * np.abs((delta - prev_delta)) * prev_traded_price
        next_portfolio_val = delta * next_traded_price + (1 + self.r * self.dt) * (
                prev_portfolio_value - delta * prev_traded_price - transaction_costs)
        self.set_port_val(next_portfolio_val)
        self.set_moneyness(np.log(next_stock_price / self.K))
        self.set_previous_delta(delta)

        reward = self.reward(prev_state)

        done = False
        if np.isclose(self.get_ttm(self.state), 0):
            done = True

        self.update_state_features()

        return self.state_features, reward, done, False, {}

    def reward(self, prev_state):
        change_portfolio_value = self.get_port_val(self.state) - self.get_port_val(prev_state)
        if np.isclose(self.get_ttm(self.state), 0):
            next_option_value = BSM_call_option.payoff(self.get_stock_price(self.state), self.K)
        else:
            next_option_value = BSM_call_option.exp_payoff(self.get_stock_price(self.state),
                                                           self.get_ttm(self.state), self.sigma, self.mu,
                                                           self.K, 1000)
        prev_option_value = BSM_call_option.exp_payoff(self.get_stock_price(prev_state),
                                                           self.get_ttm(prev_state), self.sigma, self.mu,
                                                           self.K, 1000)
        change_option_value = next_option_value - prev_option_value
        PnL = change_portfolio_value - change_option_value
        reward = PnL - (self.kappa / 2) * (PnL ** 2)
        return reward

    # Updates the state features
    def update_state_features(self, reset=False):
        if reset:
            self.state_features = np.zeros((4), dtype=np.float32)
        self.state_features[0] = self.get_stock_price(self.state) / self.get_traded_price(self.state)
        self.state_features[1] = self.get_ttm(self.state) / self.T
        self.state_features[2] = self.get_moneyness(self.state)
        self.state_features[3] = self.get_traded_price(self.state) / self.Y_0

    def render(self):
        pass

    def close(self):
        pass

    # The following functions allow us to more conveniently set and get state variables
    def get_stock_price(self, state_vec):
        return state_vec[0]

    def get_traded_price(self, state_vec):
        return state_vec[1]

    def get_ttm(self, state_vec):
        return state_vec[2]

    def get_port_val(self, state_vec):
        return state_vec[3]

    def get_moneyness(self, state_vec):
        return state_vec[4]

    def get_previous_delta(self, state_vec):
        return state_vec[5]

    def set_stock_price(self, S_t):
        self.state[0] = S_t

    def set_traded_price(self, Y_t):
        self.state[1] = Y_t

    def set_ttm(self, ttm):
        self.state[2] = ttm

    def set_port_val(self, port_val):
        self.state[3] = port_val

    def set_moneyness(self, moneyness):
        self.state[4] = moneyness

    def set_previous_delta(self, p):
        self.state[5] = p


class AmericanThesisEnv(gym.Env):
    def __init__(self, r, K, sigma, mu, S_0, sigma_1, mu_1, corr, dt, T, kappa, transaction_cost):
        self.r = r
        self.K = K
        self.sigma = sigma
        self.mu = mu
        self.S_0 = S_0
        self.sigma_1 = sigma_1
        self.mu_1 = mu_1
        self.corr = corr
        self.dt = dt
        self.T = T
        self.kappa = kappa
        self.transaction_cost = transaction_cost

        # self.action_space = spaces.Discrete(201)
        self.action_space = spaces.Box(low=0., high=2., dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        self.state_features = np.zeros((3), dtype=np.float32)
        self.state = np.zeros((5), dtype=np.float32)
        self.set_stock_price(self.S_0)
        self.set_ttm(self.T)

        # Initialize asset model
        self.asset_price_model = SimpleGBMSim(
            dt=self.dt,
            drift=self.mu,
            volatility=self.sigma,
            s_0=self.S_0
        )
        # Initialize option model
        self.option_price_model = SimpleBTSim(
            initial_asset_price=self.S_0,
            strike_price=self.K,
            risk_free_interest_rate=self.r,
            volatility=self.sigma,
            T=self.T
        )

        self.set_port_val(self.option_price_model.reset())

        self.set_moneyness(np.log(self.get_stock_price(self.state) / self.K))
        self.set_previous_delta(0)

    def reset(self, seed=None, options=None):
        self.state = np.zeros((5), dtype=np.float32)
        self.set_stock_price(self.S_0)
        self.set_ttm(self.T)
        self.asset_price = self.asset_price_model.reset()
        self.set_port_val(self.option_price_model.reset())

        self.set_moneyness(np.log(self.get_stock_price(self.state) / self.K))
        self.set_previous_delta(0)
        self.update_state_features(reset=True)

        return self.state_features, {}

    def step(self, action):
        delta = action[0]
        prev_state = self.state.copy()
        prev_stock_price = self.get_stock_price(prev_state)

        next_stock_price = self.asset_price_model()


        prev_portfolio_value = self.get_port_val(prev_state)
        prev_delta = self.get_previous_delta(prev_state)

        self.set_stock_price(next_stock_price)
        self.set_ttm(self.get_ttm(prev_state) - self.dt)

        transaction_costs = self.transaction_cost * np.abs((delta - prev_delta)) * prev_stock_price
        next_portfolio_val = delta * next_stock_price + (1 + self.r * self.dt) * (
                prev_portfolio_value - delta * prev_stock_price - transaction_costs)

        self.set_port_val(next_portfolio_val)
        self.set_moneyness(np.log(next_stock_price / self.K))
        self.set_previous_delta(delta)

        reward = self.reward(prev_state)

        done = False
        if self.get_ttm(self.state) <= self.dt:
            done = True

        self.update_state_features()

        return self.state_features, reward, done, False, {}

    def reward(self, prev_state):
        change_portfolio_value = self.get_port_val(self.state) - self.get_port_val(prev_state)
        if np.isclose(self.get_ttm(self.state), 0):
            next_option_value = self.option_price_model.payoff(self.get_stock_price(self.state))
        else:
            next_option_value = self.option_price_model.exp_payoff(
                self.get_stock_price(self.state),
                self.get_ttm(self.state),
                self.sigma,
                self.mu,
                self.K
            )
        prev_option_value = self.option_price_model.exp_payoff(
            x=self.get_stock_price(prev_state),
            ttm=self.get_ttm(prev_state),
            sigma=self.sigma,
            mu=self.mu,
            K=self.K
        )

        change_option_value = next_option_value - prev_option_value
        PnL = change_portfolio_value - change_option_value
        reward = PnL - (self.kappa / 2) * (PnL ** 2)
        return reward

    # Updates the state features
    def update_state_features(self, reset=False):
        if reset:
            self.state_features = np.zeros((3), dtype=np.float32)
        self.state_features[0] = self.get_ttm(self.state) / self.T
        self.state_features[1] = self.get_moneyness(self.state)
        self.state_features[2] = self.get_stock_price(self.state) / self.S_0

    def render(self):
        pass

    def close(self):
        pass

    # The following functions allow us to more conveniently set and get state variables
    def get_stock_price(self, state_vec):
        return state_vec[0]

    def get_ttm(self, state_vec):
        return state_vec[1]

    def get_port_val(self, state_vec):
        return state_vec[2]

    def get_moneyness(self, state_vec):
        return state_vec[3]

    def get_previous_delta(self, state_vec):
        if len(state_vec) >= 5:
          return state_vec[4]
        else:
          return 0

    def set_stock_price(self, S_t):
        self.state[0] = S_t

    def set_ttm(self, ttm):
        self.state[1] = ttm

    def set_port_val(self, port_val):
        self.state[2] = port_val

    def set_moneyness(self, moneyness):
        self.state[3] = moneyness

    def set_previous_delta(self, p):
        self.state[4] = p


class American_call_option:
      @staticmethod
      def payoff(x, K):
        return np.maximum(x - K, np.zeros(x.shape))

      @staticmethod
      def exp_payoff(x, ttm, sigma, mu, K, n_samples):
        # Generate random samples
        Y = np.random.normal((mu - 0.5 * (sigma ** 2)) * ttm, sigma * np.sqrt(ttm), size=n_samples)
        final_value = 0
        # Initialize option value at expiration, can break out into separate payoff function
        intrinsic_value = np.maximum(x * np.exp(Y) - K, np.zeros(x.shape))

        # Work backward through time to check for optimal exercise, might need to have dt in here (?)
        for t in range(int(ttm) - 1, 0, -1):
            Y = np.random.normal((mu - 0.5 * (sigma ** 2)) * t, sigma * np.sqrt(t), size=n_samples)

            # Calculate discounted expected option value
            discounted_option_value = intrinsic_value * np.exp(-mu * (ttm - t))

            S_T = x * np.exp(Y)

            # Update intrinsic value for early exercise
            intrinsic_value = np.maximum(x * np.exp(Y) - K, discounted_option_value)
            final_value = np.maximum(intrinsic_value, np.zeros(intrinsic_value.shape))

        # Calculate the expected payoff
        expected_payoff = np.mean(final_value)

        return expected_payoff


class KellyThesisEnv(gym.Env):
    def __init__(self, asset_price_model, option_price_model,r, K, sigma, mu, S_0, Y_0, sigma_1, mu_1, corr, dt, T, kappa, transaction_cost):
        self.r = r
        self.K = K
        self.sigma = sigma
        self.mu = mu
        self.S_0 = S_0
        self.Y_0 = Y_0
        self.sigma_1 = sigma_1
        self.mu_1 = mu_1
        self.corr = corr
        self.dt = dt
        self.T = T
        self.kappa = kappa
        self.transaction_cost = transaction_cost

        self.action_space = spaces.Discrete(101)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)

        self.state_features = np.zeros((4), dtype=np.float64)
        self.state = np.zeros((6), dtype=np.float64)
        self.set_stock_price(self.S_0)
        self.set_ttm(self.T)

        # Adding in the simulators
        # Initialize asset model
        self.asset_price_model = asset_price_model(
            dt=self.dt,
            drift=self.mu,
            volatility=self.sigma,
            s_0=self.S_0
        )
        # Initialize option model
        self.option_price_model = option_price_model(
            initial_asset_price=self.S_0,
            strike_price=self.K,
            risk_free_interest_rate=self.r,
            volatility=self.sigma,
            T=self.T,
            dt=self.dt
        )





        #this needs to get changed, this is the initial option price
        # self.set_port_val(BSM_call_option.price(self.get_stock_price(self.state),
        #                                         self.get_ttm(self.state),
        #                                         self.sigma,
        #                                         self.K, self.r))
        self.set_port_val(self.option_price_model.compute_price([self.S_0]))

        self.set_moneyness(np.log(self.get_stock_price(self.state) / self.K))
        self.set_previous_delta(0)

        #this needs to get changed, this is the BSM delta ** Leave for now, change later if need
        self.BSMDelta = BSM_call_option.delta(self.get_stock_price(self.state),
                                                self.get_ttm(self.state),
                                                self.sigma,
                                                self.K, self.r)

# I don't like how they are generating prices, this will not be used any more
    def generate_next_prices(self, S_t, Y_t):
        delta_W1 = np.sqrt(self.dt) * np.random.normal(0, 1)
        delta_W2 = np.sqrt(self.dt) * np.random.normal(0, 1)
        delta_W3 = self.corr * delta_W1 + np.sqrt(1 - (self.corr ** 2)) * delta_W2

        X_1 = np.exp(self.sigma * delta_W3 + ((self.mu - 0.5 * (self.sigma ** 2)) * self.dt))
        X_2 = np.exp(self.sigma_1 * delta_W1 + ((self.mu_1 - 0.5 * (self.sigma_1 ** 2)) * self.dt))

        next_stock_price = X_1 * S_t
        next_traded_price = X_2 * Y_t

        return next_stock_price, next_traded_price

    def reset(self, seed=None, options=None):
        self.state = np.zeros((6), dtype=np.float64)
        self.set_stock_price(self.S_0)
        self.set_traded_price(self.Y_0)
        self.set_ttm(self.T)
        self.asset_price_model.reset()
        self.asset_price = self.asset_price_model.get_current_price()
        self.option_price_model.reset()
        self.option_price = self.option_price_model.get_current_price()

        #this needs to get changed, this will be the option price
        # self.set_port_val(BSM_call_option.price(self.get_stock_price(self.state),
        #                                         self.get_ttm(self.state),
        #                                         self.sigma,
        #                                         self.K, self.r))
        self.set_port_val(self.option_price_model.compute_price([self.get_stock_price(self.state)]))

        self.set_moneyness(np.log(self.get_stock_price(self.state) / self.K))
        self.set_previous_delta(0)
        self.update_state_features(reset=True)

        return self.state_features, {}

    def step(self, action):
        # delta = action[0]
        delta = action / 100.
        prev_state = self.state.copy()
        prev_stock_price = self.get_stock_price(prev_state)

        #this will go
        # prev_traded_price = self.get_traded_price(prev_state)
        #this needs to get changed, this will be the price from our asset model,
        # next_stock_price, next_traded_price = self.generate_next_prices(prev_stock_price, prev_traded_price)
        next_stock_price = self.asset_price_model()[-1]


        prev_portfolio_value = self.get_port_val(prev_state)
        prev_delta = self.get_previous_delta(prev_state)

        #this needs to get changed, this will be the BSM delta, can leave here for now, change later if need
        self.BSMDelta = BSM_call_option.delta(self.get_stock_price(self.state),
                                                self.get_ttm(self.state),
                                                self.sigma,
                                                self.K, self.r)

        self.set_stock_price(next_stock_price)

        #this will go
        # self.set_traded_price(next_traded_price)

        self.set_ttm(self.get_ttm(prev_state) - self.dt)

        #this will change to prev_stock_price (both) - getting rid of trade price
        transaction_costs = self.transaction_cost * np.abs((delta - prev_delta)) * prev_stock_price
        next_portfolio_val = delta * next_stock_price + (1 + self.r * self.dt) * (
                prev_portfolio_value - delta * prev_stock_price - transaction_costs)

        self.set_port_val(next_portfolio_val)
        self.set_moneyness(np.log(next_stock_price / self.K))
        self.set_previous_delta(delta)

        reward = self.reward(prev_state)

        done = False
        if self.get_ttm(self.state) <= self.dt:
            done = True

        self.update_state_features()

        return self.state_features, reward, done, False, {}

    def reward(self, prev_state):
        change_portfolio_value = self.get_port_val(self.state) - self.get_port_val(prev_state)
        if np.isclose(self.get_ttm(self.state), 0):

          #this needs to get changed, this will be the option price or value?
            next_option_value = American_call_option.payoff(self.get_stock_price(self.state), self.K)
        else:
          #this needs to get changed, this will be the option price or value?
            next_option_value = American_call_option.exp_payoff(self.get_stock_price(self.state),
                                                           self.get_ttm(self.state), self.sigma, self.mu,
                                                           self.K, 1000)
            #this needs to get changed
        prev_option_value = American_call_option.exp_payoff(self.get_stock_price(prev_state),
                                                           self.get_ttm(prev_state), self.sigma, self.mu,
                                                           self.K, 1000)
        change_option_value = next_option_value - prev_option_value
        PnL = change_portfolio_value - change_option_value
        reward = PnL - (self.kappa / 2) * (PnL ** 2)
        return reward

    # Updates the state features
    def update_state_features(self, reset=False):
        if reset:
            self.state_features = np.zeros((4), dtype=np.float64)
        self.state_features[0] = self.get_stock_price(self.state) / self.get_traded_price(self.state)
        self.state_features[1] = self.get_ttm(self.state) / self.T
        self.state_features[2] = self.get_moneyness(self.state)
        self.state_features[3] = self.get_traded_price(self.state) / self.Y_0

    def render(self):
        pass

    def close(self):
        pass

    # The following functions allow us to more conveniently set and get state variables
    def get_stock_price(self, state_vec):
        return state_vec[0]

    def get_traded_price(self, state_vec):
        return state_vec[1]

    def get_ttm(self, state_vec):
        return state_vec[2]

    def get_port_val(self, state_vec):
        return state_vec[3]

    def get_moneyness(self, state_vec):
        return state_vec[4]

    def get_previous_delta(self, state_vec):
        if len(state_vec) >= 6:
          return state_vec[5]
        else:
          return 0

    def set_stock_price(self, S_t):
        self.state[0] = S_t

    def set_traded_price(self, Y_t):
        self.state[1] = Y_t

    def set_ttm(self, ttm):
        self.state[2] = ttm

    def set_port_val(self, port_val):
        self.state[3] = port_val

    def set_moneyness(self, moneyness):
        self.state[4] = moneyness

    def set_previous_delta(self, p):
        self.state[5] = p


class TestEnv(gym.Env):
    def __init__(
        self,
        T,
        asset_prices,
        drift=0.,
        volatility=0.15,
        risk_free_interest_rate=0.,
        trading_cost_para=0.01,
        risk_aversion=1.0,
        strike_price=None
    ):
        super().__init__()

        self.T = T
        self.num_steps = len(asset_prices)
        self.dt = T / self.num_steps
        self.asset_prices = asset_prices
        self.asset_price = asset_prices[0]
        self.drift = drift
        self.volatility = volatility
        self.risk_free_interest_rate = risk_free_interest_rate
        self.trading_cost_para = trading_cost_para
        self.risk_aversion = risk_aversion
        self.strike_price = strike_price if strike_price is not None else asset_prices[0]
        self.holdings = 0.
        self.current_step = 1

        # Initialize option model
        self.option_price_model = SimpleBTSim(
            initial_asset_price=asset_prices[0],
            strike_price=self.strike_price,
            risk_free_interest_rate=self.risk_free_interest_rate,
            volatility=self.volatility,
            T=self.T
        )
        self.initial_option_price = self.option_price_model.reset()
        self.option_price = self.initial_option_price

        self.portfolio_value = self.initial_option_price
        
        # Action space: integer amount of the underlyer to hold
        # 100L + 1 actions for range [0, 100L] inclusive
        self.action_space = spaces.Discrete(101)

        # State space: current stock price, time to maturity, current holdings, strike price
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)


    def step(self, action):
        # Convert action to amount of underlyer to trade
        prev_port_value = self.portfolio_value
        prev_stock_price = self.asset_price
        next_stock_price = self.asset_prices[self.current_step]
        prev_delta = self.holdings
        prev_option_price = self.option_price

        delta = action
        transaction_costs = self.trading_cost_para * np.abs((delta - prev_delta)) * prev_stock_price
        next_portfolio_val = delta * next_stock_price + (1 + self.r * self.dt) * (
                prev_port_value - delta * prev_stock_price - transaction_costs)

        self.holdings = delta
        self.asset_price = next_stock_price
        self.portfolio_value = next_portfolio_val

        change_port_value = self.portfolio_value - prev_port_value

        ttm = max(1e-7, self.T - self.current_step * self.dt)
        self.option_price = self.option_price_model(self.asset_price, ttm)
        change_option_value = self.option_price - prev_option_price

        reward = change_port_value - change_option_value

        self.current_step += 1
        # End episode when option expires
        done = self.current_step >= self.num_steps
        self.state = self.get_state()

        return self.state, reward, done, False, {}

    def get_state(self):
        # Construct state
        ttm = max(0., self.T - self.current_step * self.dt)
        return np.array([
            self.asset_price,
            ttm,
            self.holdings,
            self.strike_price
        ], dtype=np.float32)

    def render(self):
        pass

    def reset(self, seed=None, options=None):
        self.current_step = 1
        self.asset_price = self.asset_prices[0]
        self.holdings = 0
        self.option_price = self.initial_option_price
        self.portfolio_value = self.initial_option_price
        self.state = self.get_state()
        return self.state, {}
    
    def close(self):
        pass