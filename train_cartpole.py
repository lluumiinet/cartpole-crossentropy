import gymnasium as gym
import numpy as np
import json
from sklearn.neural_network import MLPClassifier
from template_crossentropy import select_elites


def generate_session_cartpole(env, agent, t_max=500):
    """Запуск одной игры CartPole с текущей политикой."""
    states, actions = [], []
    total_reward = 0
    s, info = env.reset()

    for _ in range(t_max):
        probs = agent.predict_proba([s])[0]
        a = np.random.choice(len(probs), p=probs)
        new_s, r, done, truncated, info = env.step(a)

        states.append(s.tolist())
        actions.append(int(a))
        total_reward += r
        s = new_s

        if done or truncated:
            break

    return states, actions, total_reward


def main():
    env = gym.make("CartPole-v1")
    n_actions = env.action_space.n

    # Простая нейросеть с одним скрытым слоем
    agent = MLPClassifier(hidden_layer_sizes=(50,), max_iter=20)

    # Первичная инициализация модели
    obs0 = env.reset()[0]
    agent.fit([obs0] * n_actions, list(range(n_actions)))

    n_sessions = 100      # игр за итерацию
    percentile = 50       # процент лучших
    n_iters = 50          # итерации обучения

    for it in range(n_iters):
        sessions = [generate_session_cartpole(env, agent) for _ in range(n_sessions)]
        states_batch, actions_batch, rewards_batch = zip(*sessions)

        mean_reward = np.mean(rewards_batch)
        print(f"Iter {it}: mean_reward={mean_reward:.1f}")

        elite_states, elite_actions = select_elites(
            states_batch, actions_batch, rewards_batch, percentile
        )
        if not elite_states:
            continue

        agent.fit(elite_states, elite_actions)

        if mean_reward > 190:
            print("Solved!")
            break

    # Сохраняем сессии для проверки
    sessions = [generate_session_cartpole(env, agent) for _ in range(20)]
    sessions_to_send = [(s, a) for s, a, r in sessions]

    with open("sessions_to_send.json", "w") as f:
        json.dump(sessions_to_send, f, ensure_ascii=True, indent=4)

    print("Файл sessions_to_send.json сохранён.")


if __name__ == "__main__":
    main()
