import gym
import numpy as np
import tensorflow as tf
from keras import layers
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer

# Initialize Qiskit Quantum Simulator
simulator = Aer.get_backend('aer_simulator')

# ---------------------------
# Quantum Multi-Head Attention
# ---------------------------
def quantum_attention(inputs):
    # Convert the tensor to a numpy array and flatten
    inputs = np.array(inputs).flatten()
    num_qubits = inputs.shape[0]
    qc = QuantumCircuit(num_qubits)

    # Encode classical input into quantum states
    for i in range(num_qubits):
        qc.ry(inputs[i] * np.pi, i)

    # Entangling layer (CNOT gates)
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)

    # Measurement
    qc.measure_all()
    
    # Transpile and run the circuit directly (without assembling a Qobj)
    transpiled_circuit = transpile(qc, simulator)
    result = simulator.run(transpiled_circuit, shots=1024).result()
    counts = result.get_counts()

    # Convert counts to probabilities for each basis state
    probabilities = np.array([counts.get(bin(i)[2:].zfill(num_qubits), 0)
                               for i in range(2**num_qubits)])
    total = probabilities.sum()
    probabilities = probabilities / total if total > 0 else np.zeros_like(probabilities)
    
    return probabilities[:num_qubits].astype(np.float32)

def quantum_attention_per_sample(x):
    y = tf.py_function(func=quantum_attention, inp=[x], Tout=tf.float32)
    y.set_shape((6,))
    return y

def quantum_attention_wrapper(x):
    # Apply per sample so that the batch dimension is preserved.
    return tf.map_fn(quantum_attention_per_sample, x, fn_output_signature=tf.float32)

# ---------------------------
# Qbot Model Definition
# ---------------------------
def create_qbot_model():
    inputs = layers.Input(shape=(6,))
    
    # Apply quantum attention and freeze its output so that gradients are not propagated through it.
    quantum_out = layers.Lambda(quantum_attention_wrapper, output_shape=(6,))(inputs)
    quantum_out = layers.Lambda(lambda x: tf.stop_gradient(x))(quantum_out)
    
    x = layers.Dense(32, activation="relu")(quantum_out)
    x = layers.Dense(16, activation="relu")(x)
    output = layers.Dense(2, activation="softmax")(x)  # Two actions: Up or Down
    
    return tf.keras.Model(inputs=inputs, outputs=output)

# ---------------------------
# WhaleBubbleCatch Environment
# ---------------------------
class WhaleBubbleCatchEnv(gym.Env):
    def __init__(self):
        super(WhaleBubbleCatchEnv, self).__init__()
        # Two actions: 0 -> Move Down, 1 -> Move Up
        self.action_space = gym.spaces.Discrete(2)
        # Observation: [whale_position, bubble1, bubble2, bubble3, obstacle1, obstacle2]
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.whale_position = 0.5  # Start in the middle
        self.bubbles_collected = 0
        self.lives = 3
        self.bubbles = np.random.uniform(0, 1, size=3)  # Three bubble positions
        self.obstacles = np.random.uniform(0, 1, size=2)  # Two obstacle positions
        return self._get_state(), {}

    def _get_state(self):
        return np.array([self.whale_position, *self.bubbles, *self.obstacles], dtype=np.float32)

    def step(self, action):
        # Update whale's position
        if action == 0:  # Move down
            self.whale_position = max(0, self.whale_position - 0.1)
        elif action == 1:  # Move up
            self.whale_position = min(1, self.whale_position + 0.1)

        # Bubble collection check
        for i, bubble in enumerate(self.bubbles):
            if abs(self.whale_position - bubble) < 0.05:
                self.bubbles_collected += 1
                self.bubbles[i] = np.random.uniform(0, 1)

        # Obstacle collision check
        for obstacle in self.obstacles:
            if abs(self.whale_position - obstacle) < 0.05:
                self.lives -= 1

        done = self.lives <= 0
        # Reward: +1 per bubble collected, -0.5 per life lost
        reward = self.bubbles_collected - (3 - self.lives) * 0.5
        return self._get_state(), reward, done, {}

# ---------------------------
# Helper Functions for Policy Gradient
# ---------------------------
def run_episode(model, env):
    """
    Run an episode using the current policy.
    Records states, actions, and rewards.
    """
    state, _ = env.reset()
    states = []
    actions = []
    rewards = []
    done = False

    while not done:
        state = np.reshape(state, [1, 6])
        states.append(state)
        # Use the model to compute action probabilities (without gradient recording)
        action_probs = model.predict(state, verbose=0)[0]
        action = np.random.choice(len(action_probs), p=action_probs)
        actions.append(action)
        state, reward, done, _ = env.step(action)
        rewards.append(reward)
    return np.vstack(states), np.array(actions), np.array(rewards)

def discount_rewards(rewards, gamma=0.99):
    discounted = np.zeros_like(rewards, dtype=np.float32)
    cumulative = 0.0
    for t in reversed(range(len(rewards))):
        cumulative = rewards[t] + gamma * cumulative
        discounted[t] = cumulative
    return discounted

# ---------------------------
# Training Loop (REINFORCE)
# ---------------------------
def train_qbot_model(model, env, episodes=300, learning_rate=0.01, gamma=0.99):
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    for episode in range(episodes):
        states, actions, rewards = run_episode(model, env)
        total_reward = np.sum(rewards)
        discounted_rewards = discount_rewards(rewards, gamma)
        # Normalize discounted rewards
        discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-8)
        
        with tf.GradientTape() as tape:
            states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
            # Forward pass: get probabilities for all recorded states
            probs = model(states_tensor, training=True)  # shape: (num_steps, 2)
            # Gather the probabilities of the actions that were actually taken
            indices = tf.range(tf.shape(probs)[0])
            indices = tf.stack([indices, actions], axis=1)
            chosen_probs = tf.gather_nd(probs, indices)
            log_probs = tf.math.log(chosen_probs + 1e-8)
            loss = -tf.reduce_sum(log_probs * discounted_rewards)
        
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        if episode % 50 == 0:
            print(f"Episode {episode}: Total Reward: {total_reward:.2f}")
    print("Training complete.")

# ---------------------------
# Play Episode Function (Exploitation)
# ---------------------------
def play_game(model, env):
    state, _ = env.reset()
    state = np.reshape(state, [1, 6])
    total_reward = 0
    done = False
    print("Starting game play...\n")
    while not done:
        action_probs = model.predict(state, verbose=0)
        action = np.argmax(action_probs[0])
        next_state, reward, done, _ = env.step(action)
        state = np.reshape(next_state, [1, 6])
        total_reward += reward
        print(f"Action: {'Up' if action == 1 else 'Down'}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")
    print(f"\nGame Over! Final Total Reward: {total_reward:.2f}")

# ---------------------------
# Main Execution
# ---------------------------
env = WhaleBubbleCatchEnv()
qbot_model = create_qbot_model()

print("Starting training...")
train_qbot_model(qbot_model, env, episodes=300, learning_rate=0.01, gamma=0.99)
print("Training finished, playing game...")
play_game(qbot_model, env)
