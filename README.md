# Playing-Pong-with-REINFORCE

# Usage:


```bash
usage: python pong.py [-h] [--mode MODE] [--file_name FILE_NAME] [--reset_weights RESET_WEIGHTS]
               [--learning_rate LEARNING_RATE] [--gamma GAMMA] [--n_dense_nodes N_DENSE_NODES]
               [--score_goal SCORE_GOAL] [--n_games_goal N_GAMES_GOAL] [--render RENDER] [--log_file LOG_FILE]
               [--n_games N_GAMES] [--render_speed RENDER_SPEED]

```
# Arguments

|short|long|default|help|
| :--- | :--- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
||`--mode`|`train`|Train or play. If play mode, model weights file must exist. Default: train|
||`--file_name`|`pong_model.h5`|File to save model weights to. Make sure to change this parameter if training/comparing multiple models. Default: pong_model.h5|
||`--reset_weights`||Whether or not to overwrite model weights file. Default: True|
||`--learning_rate`|`0.2`|Learning rate of policy gradient updates. Default: 0.2|
||`--gamma`|`0.99`|Discount rate for episode rewards. Default: 0.99|
||`--n_dense_nodes`|`512`|Number of nodes in Dense layer. Default: 512|
||`--score_goal`|`21`|The model will be trained until it reaches this score on average for n_games_goal. Default: 21|
||`--n_games_goal`|`25`|Number of episodes used to track average episode scores. Default: 25|
||`--render`||Whether or not to display games as model trains. Default: False|
||`--log_file`|`training_log.txt`|File to log episode results to. Default: training_log.txt|
||`--n_games`|`25`|If mode is play, the number of games to play. Default: 25|
||`--render_speed`|`1`|If mode is play, speed of rendering. Options: 1 (fast), 2 (medium), 3 (slow). Default: 1|
