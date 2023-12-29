from pettingzoo.atari import boxing_v2

env = boxing_v2.env(render_mode="human", auto_rom_install_path=None)
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        action = env.action_space(agent).sample()

    env.step(action)
env.close()
