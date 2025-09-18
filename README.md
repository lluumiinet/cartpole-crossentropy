# CartPole Cross-Entropy Method 🎯

Реализация обучения агента для среды **CartPole-v1** с помощью метода **Cross-Entropy**.  
Агент обучается выбирать действия, используя простую нейросеть (`sklearn.MLPClassifier`).

---

## 📌 Описание
- Среда: `gymnasium` (CartPole-v1)  
- Алгоритм: Cross-Entropy Method (CEM)  
- Модель: `MLPClassifier` из scikit-learn  
- Сессии обучения сохраняются в `sessions_to_send.json` для последующей проверки  

---

## 🎮 Демонстрация
![CartPole-v1](https://miro.medium.com/v2/resize:fit:720/format:webp/1*FJ5bFDFywfSZdcQj5NCz8g.gif)

---

## 📂 Файлы
- **train_cartpole.py** — основной скрипт для обучения агента  
- **template_crossentropy.py** — вспомогательные функции (`select_elites` и др.)  
- **sessions_to_send.json** — сохранённые траектории игр (генерируется после обучения)  

---

## 🚀 Запуск
Установи зависимости:
```bash
pip install gymnasium numpy scikit-learn
