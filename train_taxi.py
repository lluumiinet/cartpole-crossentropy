# train_taxi.py
import gymnasium as gym
import numpy as np
import json
from template_crossentropy import select_elites, update_policy


def generate_session(env, policy, t_max=10**4):
    """
    Проигрывает один эпизод в среде Taxi-v3.
    Возвращает: список состояний, список действий, суммарную награду.
    """
    states, actions = [], []
    total_reward = 0.0

    s, info = env.reset()

    for _ in range(t_max):
        # выбираем действие случайно на основе вероятностей policy
        a = np.random.choice(env.action_space.n, p=policy[s])
        new_s, r, done, truncated, info = env.step(a)

        states.append(s)
        actions.append(a)
        total_reward += r

        s = new_s
        if done or truncated:
            break
    return states, actions, total_reward


def main():
    # создаём окружение
    env = gym.make("Taxi-v3")

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # инициализируем политику равномерно
    policy = np.ones((n_states, n_actions)) / n_actions

    # параметры обучения
    n_sessions = 200    # сколько эпизодов в одной итерации
    percentile = 70     # порог отбора лучших эпизодов
    n_iters = 30        # количество итераций обучения

    for it in range(n_iters):
        # собираем данные
        sessions = [generate_session(env, policy) for _ in range(n_sessions)]
        states_batch, actions_batch, rewards_batch = zip(*sessions)

        mean_reward = np.mean(rewards_batch)
        print(f"Iter {it}: mean_reward={mean_reward:.1f}")

        # выбираем элитные состояния и действия
        elite_states, elite_actions = select_elites(states_batch, actions_batch, rewards_batch, percentile)

        if not elite_states:
            continue

        # обновляем стратегию
        new_policy = update_policy(elite_states, elite_actions, n_states, n_actions)
        policy = 0.5 * policy + 0.5 * new_policy  # сглаживание

        # условие успешного обучения
        if mean_reward > 100:
            print("Решено!")
            break

    # сохраняем несколько эпизодов в json
    sessions = [generate_session(env, policy) for _ in range(20)]
    sessions_to_send = []
    for s, a, r in sessions:
        observations = [int(x) for x in s]   # состояния как int
        actions = [int(x) for x in a]        # действия как int
        sessions_to_send.append((observations, actions))

    with open("sessions_to_send.json", "w") as f:
        json.dump(sessions_to_send, f, ensure_ascii=True, indent=4)

    print("Файл sessions_to_send.json сохранён.")


if __name__ == "__main__":
    main()
