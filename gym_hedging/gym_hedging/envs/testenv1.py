import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.stats import norm
from gymnasium.envs.registration import EnvSpec

from gym_hedging.utils.simulators import GBMSimulator, BinomialTreeOptionSimulator, BSM_call_option

class DeltaHedging(gym.Env):
    def __init__(self, env_context=None):
        self.r = 0
        self.K = 100
        self.sigma = 0.04
        self.mu = 0.0013
        self.S_0 = 100
        self.Y_0 = 100
        self.dt = 0.2
        self.T = 50
        self.kappa = 2.5
        self.transaction_cost = 0.015
        self.is_call_option = True


        # self.state_features = np.zeros((4))
        self.state = np.zeros((5))

        self.set_stock_price(self.S_0)
        # print(f"This is the initial stock price {self.state[0]}")
        self.set_ttm(self.T)
        #Initial number of holdings is zero for RL agent 
        self.setN(0)
        self.state[3] = self.K


        # Choosing the amount of underlyer to trade, assuming one option contract of 100 shares
        self.action_space = spaces.Box(low=-100, high=100, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf]), high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf]), shape=(5,), dtype=np.float32)


        # Adding in the simulators
        # Initialize asset model
        self.asset_price_model = GBMSimulator(
            dt=self.dt,
            drift=self.mu,
            volatility=self.sigma,
            s_0=self.S_0
        )
        # Initialize option model
        self.option_price_model = BinomialTreeOptionSimulator(
            initial_asset_price=self.S_0,
            strike_price=self.K,
            risk_free_interest_rate=self.r,
            volatility=self.sigma,
            T=self.T/self.dt,
            dt=self.dt
        )

        self.BSM_call_option = BSM_call_option()

        self.initial_option = self.option_price_model.compute_price([self.S_0])
        self.option = self.initial_option

        self.set_port_val(self.option_price_model.compute_price([self.S_0]))
        # self.set_moneyness(np.log(self.get_stock_price(self.state) / self.K))
        # self.set_previous_delta(0)
        self.spec = EnvSpec("DeltaHedging-v0", max_episode_steps=self.T)

        self.BSMDelta = BSM_call_option.delta(self.get_stock_price(self.state),
                                                self.get_ttm(self.state),
                                                self.sigma,
                                                self.K, self.r)
        
        # print(f"From start BSM delta: {self.BSMDelta}")




    def step(self, action):
        # The action that the RL agent choses is the number of shares to be traded, so the delta value is 
        # previous number of shares + action /100
        
        # num_traded = action[0]
        num_traded = action

        prev_state = self.state.copy()
        prev_stock_price = self.get_stock_price(prev_state)
        prev_option_value = self.option
        next_stock_price = self.asset_price_model()[-1]
        # print(f"This is the next asset price: {next_stock_price}")
        next_option_value = self.option_price_model.compute_price([self.get_stock_price(self.state)])

        truncated = False
        info = {}

        prev_portfolio_value = self.get_port_val(prev_state)

        #good
        prev_delta = self.getN(prev_state)/100
        delta = (self.getN(prev_state) + num_traded)/100

        self.BSMDelta = BSM_call_option.delta(self.get_stock_price(self.state),
                                                self.get_ttm(self.state),
                                                self.sigma,
                                                self.K, self.r)
        # print(f"From step BSM delta: {self.BSMDelta}")

        self.set_stock_price(next_stock_price)
        self.set_ttm(self.get_ttm(prev_state) - self.dt)
        self.setN(self.getN(prev_state) + num_traded)
        self.state[3] = self.K

        transaction_costs = self.transaction_cost * 100 * np.abs((delta - prev_delta)) * prev_stock_price
        # transaction_costs = self.transaction_cost * num_traded * prev_stock_price

        # next_portfolio_val = delta * next_stock_price + (1 + self.r * self.dt) * (
        #         prev_portfolio_value - delta * prev_stock_price - transaction_costs)

        next_portfolio_val = num_traded * next_stock_price + (1 + self.r * self.dt) * (
                prev_portfolio_value - num_traded * prev_stock_price - transaction_costs)
        
        # next_portfolio_val = num_traded * next_stock_price + (1 + self.r * self.dt) * (
        #         prev_portfolio_value - transaction_costs)

        self.set_port_val(next_portfolio_val)
        # self.set_moneyness(np.log(next_stock_price / self.K))
        # self.set_previous_delta(delta)

        reward = self.reward(prev_state, prev_option_value, next_option_value)

        done = False
        if (self.get_ttm(self.state) <= self.dt):
            done = True

        # print("ttm" + str(self.get_ttm(self.state)))

        # self.update_state_features()
        info = {"truncated": False}

        # print(self.state)

        return self.state, reward, done, truncated, info

    

    def reward(self, prev_state, prev_option_value, next_option_value):
        current_stock_price = self.get_stock_price(self.state)
        strike_price = self.K
        intrinsic_value = max(0, current_stock_price - strike_price) if self.is_call_option else max(0, strike_price - current_stock_price)
        time_value = next_option_value - intrinsic_value

        change_portfolio_value = self.get_port_val(self.state) - self.get_port_val(prev_state)
        change_option_value = next_option_value - prev_option_value
        PnL = change_portfolio_value - change_option_value

        reward = PnL - (self.kappa / 2) * (PnL ** 2)
        if np.isclose(self.get_ttm(self.state), 0) or intrinsic_value > time_value:
            reward += intrinsic_value  # Encourage exercising if intrinsic > time value

        return reward



    
        # Updates the state features
    # def update_state_features(self, reset=False):
    #     if reset:
    #         self.state_features = np.zeros((4))
    #     self.state_features[0] = self.get_stock_price(self.state) / self.get_traded_price(self.state)
    #     self.state_features[1] = self.get_ttm(self.state) / self.T
    #     self.state_features[2] = self.get_moneyness(self.state)
    #     self.state_features[3] = self.get_traded_price(self.state) / self.Y_0

    # The following functions allow us to more conveniently set and get state variables
    def get_stock_price(self, state_vec):
        return state_vec[0]
    
    # def get_traded_price(self, state_vec):
    #     return state_vec[1]

    def get_ttm(self, state_vec):
        return state_vec[1]
    
    def getN(self, state_vec):
        return state_vec[2]

    def get_port_val(self, state_vec):
        return state_vec[4]

    # def get_moneyness(self, state_vec):
    #     return state_vec[4]

    # def get_previous_delta(self, state_vec):
    #     if len(state_vec) >= 6:
    #       return state_vec[5]
    #     else:
    #       return 0

    def set_stock_price(self, S_t):
        self.state[0] = S_t

    # def set_traded_price(self, Y_t):
    #     self.state[1] = Y_t

    def set_ttm(self, ttm):
        self.state[1] = ttm

    def setN(self, N):
        # print(f"setting Number of holdings to: {N}")
        self.state[2] = N

    def set_port_val(self, port_val):
        self.state[4] = port_val

    # def set_moneyness(self, moneyness):
    #     self.state[4] = moneyness

    # def set_previous_delta(self, p):
    #     self.state[5] = p

    def reset(self, seed=None, options=None):
        self.state = np.zeros((5))
        self.set_stock_price(self.S_0)
        self.set_ttm(self.T)
        self.setN(0)
        # self.set_traded_price(self.Y_0)
        self.asset_price_model.reset()
        self.asset_price = self.asset_price_model.get_current_price()
        self.option_price_model.reset()
        self.option_price = self.option_price_model.get_current_price()

        self.set_port_val(self.option_price_model.compute_price([self.get_stock_price(self.state)]))

        # self.set_moneyness(np.log(self.get_stock_price(self.state) / self.K))
        # self.set_previous_delta(0)
        # self.update_state_features(reset=True)

        return self.state, {}