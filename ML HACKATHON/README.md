# Hangman ML Agent

This project is an intelligent agent designed to play Hangman. It uses a hybrid approach, combining a linguistic **Hidden Markov Model (HMM)** for "intuition" with a **Q-Learning (Reinforcement Learning)** agent for "strategy."

The agent is trained on a 50,000-word corpus (`corpus.txt`) and evaluated on a separate 2,000-word test set (`test.txt`).

## 🧠 Core Concept: A Hybrid Approach

This agent uses two components working together:

1.  **The "Intuition" (Hybrid HMM):** A linguistic model (`HybridHangmanHMM`) provides educated guesses. It's an **ensemble** that combines four probability sources:
    * **Positional Probs (40%):** Frequency of a letter at a specific index.
    * **Bigram Probs (30%):** Context from the preceding letter (e.g., 'U' after 'Q').
    * **Trigram Probs (20%):** Richer context from the two preceding letters.
    * **Global Freq. (10%):** A simple fallback based on overall letter frequency.

2.  **The "Strategist" (RL Agent):** A Q-Learning agent (`HangmanRLAgent`) learns the optimal *policy* (strategy) for which letter to guess next.
    * **State:** The agent's state is a hash of the full, observable game: `masked_word|sorted(guessed_letters)|lives_remaining`.
    * **Reward:** A custom reward function encourages winning fast (more lives left) and heavily penalizes wrong or, most importantly, repeated guesses.
    * **Exploration:** The Ɛ-greedy strategy is enhanced. "Exploratory" moves are not fully random; they are a **weighted random choice based on the HMM's suggestions**, making exploration much more efficient.

## 📊 Performance

The agent was trained for 5,000 episodes on the training corpus and then evaluated on the 2,000-word test set (which had 0% overlap with the training data).

| Metric | Training Result (5k episodes) | Test Result (2k games) |
| :--- | :--- | :--- |
| **Win Rate** | 24.1% | **30.05%** |
| **Avg. Wrong Guesses** | N/A | 5.34 (out of 6) |
| **Total Repeated Guesses** | N/A | **0** |

The agent successfully learned to **never repeat a guess**, proving the reward shaping was effective for that rule.

## 🔧 Setup & Dependencies

This project is a Jupyter Notebook (`.ipynb`). The main dependencies required are:

* `numpy`
* `matplotlib`
* `gradio`
* `jupyter`

## 🚀 How to Run

The notebook is designed to be run cell-by-cell.

1.  **Cell 1-2:** Install dependencies and load the `corpus.txt` and `test.txt` datasets.
2.  **Cell 3-5:** Define the classes for the HMM, the Hangman Environment, and the RL Agent.
3.  **Cell 6:** This is the main training cell. It trains the HMM on the corpus and then trains the RL agent for 5,000 episodes. A `best_agent_qtable.npy` file is saved with the best-performing Q-table.
4.  **Cell 7:** Visualizes the agent's training progress (reward and win rate).
5.  **Cell 8-9:** Runs the final evaluation on the 2,000-word test set and plots the results.
6.  **Cell 10:** Runs a simple, text-based interactive demo to watch the agent play 3 games from the test set.
7.  **Cell 11:** Launches a **Gradio web UI** to test the *HMM's* predictive performance on any custom word.