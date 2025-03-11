import gym
import numpy as np
import tensorflow as tf
from keras import layers
from qiskit import QuantumCircuit, transpile, assemble
from qiskit_aer import Aer
# Initialize Qiskit Quantum Simulator
simulator = Aer.get_backend('aer_simulator')

# Define Quantum Multi-Head Attention Circuit
def quantum_attention(inputs):
    """Quantum Multi-Head Attention using Qiskit."""
    num_qubits = len(inputs)
    qc = QuantumCircuit(num_qubits)

    # Encode classical input into quantum states
    for i in range(num_qubits):
        qc.ry(inputs[i] * np.pi, i)

    # Entangling layer (CNOT gates)
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)

    # Measurement
    qc.measure_all()
    
    # Execute on simulator
    transpiled_circuit = transpile(qc, simulator)
    qobj = assemble(transpiled_circuit)
    result = simulator.run(qobj).result()
    counts = result.get_counts()

    # Convert quantum output to classical probabilities
    probabilities = np.array([counts.get(bin(i)[2:].zfill(num_qubits), 0) for i in range(2**num_qubits)])
    probabilities = probabilities / probabilities.sum() if probabilities.sum() > 0 else np.zeros_like(probabilities)

    return probabilities[:num_qubits]  # Use first N outputs

# Define the Qbot AI Model
def create_qbot_model():
    """Hybrid Classical-Quantum Model for Qbot"""
    inputs = layers.Input(shape=(5,))
    
    # Quantum-enhanced attention layer
    quantum_layer = layers.Lambda(lambda x: tf.convert_to_tensor(quantum_attention(x), dtype=tf.float32))(inputs)
    
    # Dense layers for decision-making
    x = layers.Dense(32, activation="relu")(quantum_layer)
    x = layers.Dense(16, activation="relu")(x)
    output = layers.Dense(2, activation="softmax")(x)  # Output probabilities for Up/Down
    
    return tf.keras.Model(inputs=inputs, outputs=output)

# Create Qbot model
qbot_model = create_qbot_model()
qbot_model.summary()

# Define Whale Bubble Catch Game Environment
class WhaleBubbleCatchEnv(gym.Env):
    def _init_(self):
        super(WhaleBubbleCatchEnv, self)._init_()

        # Action space: Move up (1) or move down (0)
        self.action_space = gym.spaces.Discrete(2)

        # Observation space: [whale_position, bubble_positions, obstacle_positions]
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)

        self.reset()

    def reset(self):
        """ Reset the environment at the start of a new game. """
        self.whale_position = 0.5  # Middle of the screen
        self.bubbles_collected = 0
        self.lives = 3
        self.bubbles = np.random.uniform(0, 1, size=3)  # Random bubble positions
        self.obstacles = np.random.uniform(0, 1, size=2)  # Random obstacle positions

        return self._get_state(), {}

    def _get_state(self):
        """ Get the current game state as an observation vector. """
        return np.array([self.whale_position, *self.bubbles, *self.obstacles], dtype=np.float32)

    def step(self, action):
        """ Apply the chosen action and update the game state. """
        # Move the whale up or down
        if action == 0:  # Move down
            self.whale_position = max(0, self.whale_position - 0.1)
        elif action == 1:  # Move up
            self.whale_position = min(1, self.whale_position + 0.1)

        # Check for bubble collection
        for i, bubble in enumerate(self.bubbles):
            if abs(self.whale_position - bubble) < 0.05:  # Close to bubble
                self.bubbles_collected += 1
                self.bubbles[i] = np.random.uniform(0, 1)  # Respawn bubble

        # Check for obstacle collision
        for obstacle in self.obstacles:
            if abs(self.whale_position - obstacle) < 0.05:
                self.lives -= 1  # Lose a life

        # Check if the game is over
        done = self.lives <= 0

        # Reward: +1 per bubble, -0.5 per life lost
        reward = self.bubbles_collected - (3 - self.lives) * 0.5

        return self._get_state(), reward, done, {}

# Create environment instance
env = WhaleBubbleCatchEnv()

# Reinforcement Learning Parameters
learning_rate = 0.001
gamma = 0.95  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 32
memory = []

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_fn = tf.keras.losses.MeanSquaredError()

# Train Qbot using Deep Q-Learning
num_episodes = 500

for episode in range(num_episodes):
    state, _ = env.reset()
    state = np.reshape(state, [1, 5])
    total_reward = 0

    for time_step in range(100):  # Maximum game length
        # Choose action (explore vs exploit)
        if np.random.rand() <= epsilon:
            action = env.action_space.sample()  # Random action (exploration)
        else:
            action_probs = qbot_model.predict(state)
            action = np.argmax(action_probs[0])  # Exploit best known move

        # Take action in the environment
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, 5])

        # Store experience in replay memory
        memory.append((state, action, reward, next_state, done))

        state = next_state
        total_reward += reward

        if done:
            break

    # Train the model using experience replay
    if len(memory) > batch_size:
        minibatch = np.random.choice(memory, batch_size, replace=False)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += gamma * np.amax(qbot_model.predict(next_state)[0])

            target_f = qbot_model.predict(state)
            target_f[0][action] = target

            qbot_model.fit(state, target_f, epochs=1, verbose=0)

    # Decay exploration rate
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    print(f"Episode {episode+1}: Total Reward = {total_reward}, Epsilon = {epsilon:.4f}")

# Save trained model
qbot_model.save("qbot_qiskit_model.h5")
print("Training Complete. Qbot Model Saved.")