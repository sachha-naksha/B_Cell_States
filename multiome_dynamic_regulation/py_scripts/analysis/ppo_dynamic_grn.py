class SelectionPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Actor: Predict edge inclusion probabilities (150x68)
        self.actor = nn.Sequential(
            nn.Linear(218, 256),  # Input: state (TF + gene expression)
            nn.ReLU(),
            nn.Linear(256, 150*68),
            nn.Sigmoid()  # Probability of selecting each edge
        )
        # Critic: Value function
        self.critic = nn.Sequential(
            nn.Linear(218, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs, ...):
        probs = self.actor(obs)  # (batch, 150x68)
        return probs, self.critic(obs)

class BetaSelectionEnv:
    def __init__(self, tf_data, gene_data, beta_matrix):
        self.tf_data = tf_data  # (150 TFs × 100 time)
        self.gene_data = gene_data  # (68 genes × 100 time)
        self.beta_matrix = beta_matrix  # (150 TFs × 68 genes × 100 time)
        self.current_step = 0

    def step(self, action_mask):
        # Reshape action_mask to (150, 68)
        mask = action_mask.reshape(150, 68)
        # Apply sparsity constraint (top 8% edges)
        mask = self._enforce_sparsity(mask)
        # Retrieve precomputed β(t) for current step
        beta_t = self.beta_matrix[:, :, self.current_step]
        # Compute predicted gene expression: sum(TF * β * mask)
        predicted = np.sum(self.tf_data[:, self.current_step] * beta_t * mask, axis=0)
        # Calculate reward
        mse = np.mean((predicted - self.gene_data[:, self.current_step]) ** 2)
        reward = -mse - 0.01 * np.sum(mask)
        # Update state
        self.current_step += 1
        done = self.current_step >= 99
        return self._get_state(), reward, done, {}

    def _enforce_sparsity(self, mask):
        # Keep top 8% of edges based on agent’s probabilities
        threshold = np.percentile(mask.flatten(), 92)  # Top 8%
        mask[mask < threshold] = 0
        mask[mask >= threshold] = 1
        return mask

from stable_baselines3 import PPO

# Initialize environment with your data
env = DummyVecEnv([lambda: BetaSelectionEnv(tf_expr_data, gene_expr_data, beta_matrix)])

# Train the agent to select edges from precomputed β
model = PPO(SelectionPolicy, env, verbose=1)
model.learn(total_timesteps=100000)

selected_edges = []
obs = env.reset()
for t in range(100):
    action, _ = model.predict(obs)
    mask = (action.reshape(150, 68) > 0.5).astype(int)  # Threshold probabilities
    selected_edges.append(mask * beta_matrix[:, :, t])  # Masked β values

# For lineage PB, at pseudotime t=10:
print("Top selected edges at t=10:")
for tf, gene in np.argwhere(selected_edges[10] != 0):
    print(f"TF {tf} → Gene {gene}: β = {selected_edges[10][tf, gene]}")