import experiment_buddy

# Hyperparameters
ENV_NAME = 'BipedalWalker-v3'

MAX_ITER = 500000

BATCH_SIZE = 64
PPO_EPOCHS = 7
CLIP_GRADIENT = 0.2
CLIP_EPS = 0.2

TRAJECTORY_SIZE = 2049
GAE_LAMBDA = 0.95
GAMMA = 0.99

## Test Hyperparameters
test_episodes = 50
best_test_result = -1e5
save_video_test = True
N_ITER_TEST = 100

POLICY_LR = 0.0004
VALUE_LR = 0.001

experiment_buddy.register(locals())
tensorboard = experiment_buddy.deploy(
    "",
    sweep_yaml="",
    proc_num=1,
    wandb_kwargs={"entity": "ionelia"}
)
