import numpy as np

class TFSelectionEnv:
    def __init__(self, pb_expressions, gc_expressions, pb_curvatures, gc_curvatures, 
                 pb_forces, gc_forces):
        # Store data
        self.pb_expressions = pb_expressions  # Shape: (68 genes, time points)
        self.gc_expressions = gc_expressions  # Shape: (68 genes, time points)
        self.pb_curvatures = pb_curvatures    # Shape: (68 genes, time points)
        self.gc_curvatures = gc_curvatures    # Shape: (68 genes, time points)
        self.pb_forces = pb_forces            # Shape: (time points, 220 TFs, 68 genes)
        self.gc_forces = gc_forces            # Shape: (time points, 220 TFs, 68 genes)
        
        # Environment state
        self.max_steps = pb_expressions.shape[1]  # Number of time points
        self.current_step = 0
        self.current_lineage = 'PB'  # 'PB' or 'GC'
        
        # State and action dimensions
        self.state_dim = 68 * 2 + 1  # Expression + curvature + lineage indicator
        self.action_dim = 220        # Binary selection of TFs
    
    def reset(self):
        # Randomly select lineage and timepoint
        self.current_lineage = np.random.choice(['PB', 'GC'])
        self.current_step = np.random.randint(0, self.max_steps - 1)
        return self._get_state()
    
    def _get_state(self):
        # Build state vector based on current lineage and timepoint
        if self.current_lineage == 'PB':
            expressions = self.pb_expressions[:, self.current_step]
            curvatures = self.pb_curvatures[:, self.current_step]
            lineage_indicator = 1.0
        else:
            expressions = self.gc_expressions[:, self.current_step]
            curvatures = self.gc_curvatures[:, self.current_step]
            lineage_indicator = 0.0
        
        # Combine into state vector
        state = np.concatenate([expressions, curvatures, [lineage_indicator]])
        return state
    
    def step(self, action):
        # action is binary vector (220,) indicating which TFs are selected
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Move to next timepoint
        self.current_step = (self.current_step + 1) % self.max_steps
        
        # Randomly switch lineage occasionally (25% chance)
        if np.random.random() < 0.25:
            self.current_lineage = 'PB' if self.current_lineage == 'GC' else 'GC'
        
        # Get next state
        next_state = self._get_state()
        
        # Always return done=False to keep episodes going
        done = False
        
        return next_state, reward, done, {}
    
    def _calculate_reward(self, action):
        # Get forces for current lineage and timepoint
        if self.current_lineage == 'PB':
            forces = self.pb_forces[self.current_step]
            curvature = self.pb_curvatures[:, self.current_step]
            other_forces = self.gc_forces[self.current_step]
        else:
            forces = self.gc_forces[self.current_step]
            curvature = self.gc_curvatures[:, self.current_step]
            other_forces = self.pb_forces[self.current_step]
        
        # Get selected TFs
        selected_indices = np.where(action == 1)[0]
        
        # If no TFs selected, return negative reward
        if len(selected_indices) == 0:
            return -10.0
        
        # Calculate predicted curvature from selected TFs
        selected_forces = forces[selected_indices]
        predicted_curvature = np.sum(selected_forces, axis=0)
        
        # Calculate MSE between predicted and actual curvature
        mse = np.mean((predicted_curvature - curvature) ** 2)
        
        # Calculate how differently selected TFs act between lineages
        selected_other_forces = other_forces[selected_indices]
        force_differential = np.mean(np.abs(selected_forces - selected_other_forces))
        
        # Penalty for selecting too many TFs
        sparsity_penalty = -0.1 * len(selected_indices)
        
        # Final reward: minimize error, maximize differential, be sparse
        reward = -mse + 2.0 * force_differential + sparsity_penalty
        
        return reward