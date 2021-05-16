import torch
from model import A2C_policy
from utils import transforms
from utils import create_env


def main():
    env, action_size, state_size = create_env()
    checkpoint = torch.load(f"./ckpts/best-checkpoint.pth")
    best_reward_so_far = float(checkpoint["test_reward"])
    print(f"best reward: {best_reward_so_far:.2f}")
    agent_policy = A2C_policy(state_size, action_size)
    agent_policy.load_state_dict(checkpoint['agent_policy'])
    dummy_input = torch.randn((1, state_size[0]))
    dummy_input_t = transforms(dummy_input, "cpu")
    model = agent_policy.to(torch.device('cpu'))
    torch.onnx.export(model, dummy_input_t.to(torch.device('cpu')), f"submission_actor_{best_reward_so_far:.2f}.onnx",
                      verbose=False,
                      opset_version=10,
                      export_params=True, do_constant_folding=True)


if __name__ == '__main__':
    main()
