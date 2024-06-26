from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms import Algorithm
# import ray.rllib as po
from tqdm import tqdm  

from gym_hedging.envs.testenv1 import DeltaHedging
import matplotlib.pyplot as plt


# import numpy as np
import gymnasium
from ray import tune


# def create_environment(env_config):
#     return gymnasium.make('DeltaHedging-v0', r=0, K=100, sigma=0.04, mu=0.0013, S_0=100, 
#                         dt=0.2, T=50, kappa=2.5, transaction_cost=0.015)
# for env_id in gymnasium.envs.registry:
#     print(env_id)

# tune.register_env('DeltaHedging-v0', create_environment)
# print("after tune")

config = (  # 1. Configure the algorithm,
    PPOConfig()
    .environment(env = DeltaHedging)
    .rollouts(num_rollout_workers=2)
    .framework("torch")
    .training(model={"fcnet_hiddens": [64, 64]})
    .evaluation(evaluation_num_workers=1, evaluation_interval=5, evaluation_duration=10)
)

print("done config")

algo = config.build()  # 2. build the algorithm,

print("done build")

# Number of training iterations.
num_iterations = 10
# Interval at which to perform evaluations.
eval_interval = 5

checkpoint_path = "gym_hedging/checkpoints2"

training_rewards = []
evaluation_rewards = []
evaluation_lengths = []

print("Starting training...")
for i in tqdm(range(num_iterations), desc="Training"):
    result = algo.train()
    training_rewards.append(result['episode_reward_mean'] / 249)  # Normalized mean training reward.

    # Evaluate and save checkpoints at specified intervals.
    if (i + 1) % eval_interval == 0 or i == num_iterations - 1:
        checkpoint = algo.save(checkpoint_path)
        print(f"Checkpoint saved at {checkpoint} after iteration {i+1}")
        
        # Perform evaluation and store results.
        eval_result = algo.evaluate()
        evaluation_rewards.append(eval_result['evaluation']['episode_reward_mean'] / 249)
        evaluation_lengths.append(eval_result['evaluation']['episode_len_mean'])
    
    # Extract the loss components from the training results.
    # learner_stats = result['info']['learner']['default_policy']['learner_stats']
    # print("Keys")
    # print(learner_stats.keys())
    # print("done keys")
    # policy_loss = learner_stats.get('policy_loss', 'N/A')
    # vf_loss = learner_stats.get('vf_loss', 'N/A')  # Value function loss.
    # total_loss = learner_stats.get('total_loss', 'N/A')

    # print(f"Iteration {i+1}: mean training reward = {result['episode_reward_mean']/249 if 'episode_reward_mean' in result else 'N/A'}")
    # # print(f"Iteration {i+1}: mean training reward = {learner_stats.get('policy_reward_mean', 'N/A')}")
    # print(f"Policy Loss: {policy_loss}, VF Loss: {vf_loss}, Total Loss: {total_loss}")

    # if (i + 1) % eval_interval == 0:
    #     print("Evaluating...")
    #     eval_result = algo.evaluate()
    #     print(f"Evaluation results after {i+1} iterations:")
    #     print(f"- Evaluation mean reward: {eval_result['evaluation']['episode_reward_mean']/249}")
    #     print(f"- Evaluation mean length: {eval_result['evaluation']['episode_len_mean']}")

print("Training and evaluation complete.")

# for env_id in gymnasium.envs.registry:
#     print(env_id)


# Plotting the training rewards.
plt.figure(figsize=(12, 6))
plt.plot(training_rewards, label='Mean Training Reward')
plt.xlabel('Training Iteration')
plt.ylabel('Normalized Reward')
plt.title('Training Rewards Over Time')
plt.legend()
plt.grid()
plt.show()