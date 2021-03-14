LSTM-TD3
==================================
This repository implementes the LSTM-TD3 proposed in [**Memory-based Deep Reinforcement Learning for POMDP**](https://arxiv.org/pdf/2102.12344.pdf). The baselines are based on the implementations provided in [Spinning Up](https://spinningup.openai.com/) with two key changes:

- `env_wrapper` is added to implement POMDP-version of the tasks in MuJoCo and PyBullet
- `lstm_td3` is the implementation of the proposed method
- `td3_ow` is a simple variant of TD3 to incorporate memory and play the role of baseline algorithm.

How To Use
------------------
Clone the `lstm_td3` repository anywhere you'd like. (For example, in `~/lstm_td3`).
Then run the following to install the lstm_td3 code into your conda environment: 

    `cd ~/lstm_td3`
    `pip install -e .`
