from ray.rllib.algorithms.ppo import PPO
from gym_hedging.envs.testenv1 import DeltaHedging
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import torch
import numpy as np
from tabulate import tabulate


from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms import Algorithm
from gym_hedging.utils.simulators import GBMSimulator, BinomialTreeOptionSimulator

class NaiveDeltaHedgingAgent:
    def __init__(self, option_price_model):
        self.option_price_model = option_price_model
        self.delta = 0

    def get_delta(self, asset_price, h=0.1):
        asset_price1 = asset_price + h
        option_price1 = self.option_price_model.compute_price([asset_price1], stay_at_current_step=True)
        asset_price2 = asset_price - h
        option_price2 = self.option_price_model.compute_price([asset_price2], stay_at_current_step=True)
        delta_numerical = (option_price2 - option_price1) / (asset_price2 - asset_price1)
        return delta_numerical

    def takeAction(self, state):
        asset_price, _, _, _, _ = state
        self.delta = self.get_delta(asset_price*100)
        return self.delta


config = (  # 1. Configure the algorithm,
    PPOConfig()
    .environment(env = DeltaHedging)
    .rollouts(num_rollout_workers=2)
    .framework("torch")
    .training(model={"fcnet_hiddens": [64, 64]}, vf_loss_coeff=0.5,  # This is c1, the coefficient for the value function loss
        entropy_coeff=0.2 )
    .evaluation(evaluation_num_workers=1, evaluation_interval=5, evaluation_duration=10)
)

algo = config.build()
env = DeltaHedging()

naiveAgent = NaiveDeltaHedgingAgent(option_price_model=env.option_price_model)


checkpoint_path = "/Users/kellypickreign/Desktop/MSDRL/MSProjectDeepRLOptions/gym_hedging/gym_hedging/checkpoints3"
algo.restore(checkpoint_path)

n_episodes = 50

portVal = []

for episode in range(n_episodes):
    action_list = []
    BSM_delta_list = []
    RL_agent_pos = []
    asset_pnl = []
    option_pnl = []
    test_list = []

    state, _ = env.reset()
    done = False
    episode_reward = 0
    initial_option = env.option_price_model.compute_price([env.S_0])
    start = 0
    while not done:
        # print(state)

        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
            if len(state.shape) == 1:
                # Add batch dimension if there's none
                state = state.unsqueeze(0)
        # Compute action using the trained model
        action = algo.compute_single_action(state)
        # print(state)
        # action = naiveAgent.takeAction(state)

        # This is the amount to trade, the current stock position is N state[2]
        action_list.append(action)
        BSM_delta_list.append(env.BSMDelta[0]*100)
        RL_agent_pos.append(env.getN(env.state)/100+50)
        update =((env.getN(env.state)/100+50) - start) * env.get_stock_price(state)[0]
        # update = (action - start) * env.get_stock_price(state)

        start = env.getN(env.state)/100+50
        # portVal.append(env.get_port_val(env.state))
        portVal.append(update)

        new = action
        # test_list.append(new - old)
        # old = new
        # BSM_delta_policy.append(env.BSMDelta[0]*100 - #previousdelta*100)
        asset_pnl.append(env.get_stock_price(state)[0])
        # print("This")
        # print(env.get_stock_price(state))
        current_option = env.option_price_model.compute_price([env.get_stock_price(env.state)])
        option_pnl.append(initial_option-current_option)
        
        # print(f"Action taken: {action}")
        
        # Step through the environment using the action
        next_state, reward, done, _, _ = env.step(action)
        episode_reward += reward
        state = next_state
    
    # print(f"Total reward for episode {episode + 1}: {episode_reward}")
    print( f"Episode num: {episode}")




    # plt.figure(figsize=(10, 5))
    # plt.plot(RL_agent_pos, label='Agent Actions')
    # plt.plot(BSM_delta_list, label='BSM Delta Values', linestyle='--')
    # plt.plot(asset_pnl, label='Asset Price')
    # plt.plot(option_pnl, label='Option PnL')
    # plt.xlabel('Time step')
    # plt.ylabel('Value')
    # plt.title(f'Actions and BSM Delta for PPO')
    # plt.legend()
    # plt.show()

    # plt.figure(figsize=(10, 5))
    # plt.plot(action_list, label='Agent Actions')
    # plt.xlabel('Time step')
    # plt.ylabel('Value')
    # plt.title(f'Actions for PPO')
    # plt.legend()
    # plt.show()


print(f"Port Val list: {portVal}")
stats_array1 = [np.mean(portVal), np.std(portVal), np.min(portVal), np.max(portVal)]

table_data = [
    ['Statistic', 'Agent', 'Agent'],
    ['Mean', stats_array1[0]],
    ['Std Dev', stats_array1[1]],
    ['Min', stats_array1[2], ],
    ['Max', stats_array1[3], ],
]

print(tabulate(table_data, headers='firstrow', tablefmt='fancy_grid'))

    # plt.figure(figsize=(10, 5))
    # plt.plot(test_list, label='Agent Actions')
    # plt.xlabel('Time step')
    # plt.ylabel('Value')
    # plt.title(f'new and old')
    # plt.legend()
    # plt.show()






# # Assume you have set up your configuration and environment setup code as before
# config = PPOConfig().environment(env=DeltaHedging).rollouts(num_rollout_workers=2).framework("torch").training(model={"fcnet_hiddens": [64, 64]})
# algo = config.build()

# # Path to the checkpoint directory
# checkpoint_path = "gym_hedging/gym_hedging/checkpoints"

# # Restore the model
# algo.restore(checkpoint_path)



    

#     checkpoint_path = "/Users/kellypickreign/Desktop/MSDRL/MSProjectDeepRLOptions/gym_hedging/gym_hedging/checkpoints"

#     agent = ...

# terminal_hedging_error_Agent = []
# n_episodes = 50
# env = DeltaHedging()

# for episode in tqdm(range(n_episodes)):

#   option_pnl = []
#   asset_pnl = []
#   agent_positions = []
#   delta_positions = []
#   BSM_delta_positions = []
#   hedge_error_Agent = []
#   # hedge_error_Naive = []

#   state = env.reset()
#   # state_Naive = env.reset()
#   done = False
#   initial_option = env.option_price_model.compute_price([env.S_0])
#   # print(initial_option)

#   while True:
#     tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
#     action = agent.select_action(tf_state, episode)
#     # actionNaive = np.array([naive_agent.takeAction(state)])
#     # print(actionNaive)
#     # print(state_Naive[0])
#     # print(action)
#     new_state, reward, done = env.step(action)
#     # new_state, reward_naive, done_naive = env.step(actionNaive)
#     agent_positions.append(action[0]*100)
#     asset_pnl.append(env.get_stock_price(state)*100)
#     current_option = env.option_price_model.compute_price([env.get_stock_price(env.state)])
#     option_pnl.append(initial_option-current_option)
#     delta_positions.append(env.BSMDelta[0]*100)
#     hedge_error_Agent.append(env.get_port_val(env.state))
#     # hedge_error_Naive.append(env.get_port_val(env.state))
#     # print(env.get_port_val(env.state))
#     # delta_positions.append(BSM_call_option.delta(env.get_stock_price(env.state), env.get_ttm(env.state), env.sigma, env.K, env.r))
#             # self.BSMDelta = BSM_call_option.delta(self.get_stock_price(self.state),
#           #                                         self.get_ttm(self.state),
#           #                                         self.sigma,
#           #                                         self.K, self.r)
#     # BSM_delta_positions.append(env.BSMDelta[0]*100)
#     # option_pnl.append(env.option_PnL*100)
#     if done:
#       terminal_hedging_error_Agent.append(hedge_error_Agent[-1])
#       # terminal_hedging_error_Naive.append(hedge_error_Naive[-1])
#       break
#     # state = new_state
#     state = new_state

