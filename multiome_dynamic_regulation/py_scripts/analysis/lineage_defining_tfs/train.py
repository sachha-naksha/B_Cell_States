import numpy as np
import torch
from tf_environment import TFSelectionEnv
from ppo_agent import PPOAgent
import matplotlib.pyplot as plt

# Function to load data - replace with your actual loading functions
def load_data():
    # Load gene expression data (68 genes x time points)
    pb_expressions = np.load('pb_expressions.npy')
    gc_expressions = np.load('gc_expressions.npy')
    
    # Load or calculate curvatures (second derivative of expression)
    pb_curvatures = np.load('pb_curvatures.npy')
    gc_curvatures = np.load('gc_curvatures.npy')
    
    # Load TF forces (time points x 220 TFs x 68 genes)
    pb_forces = np.load('pb_forces.npy')
    gc_forces = np.load('gc_forces.npy')
    
    return pb_expressions, gc_expressions, pb_curvatures, gc_curvatures, pb_forces, gc_forces

def compute_returns_advantages(rewards, values, gamma=0.99, lam=0.95):
    # Calculate returns and advantages using GAE
    returns = []
    advantages = []
    
    # Initialize with zeros
    gae = 0
    next_value = 0
    
    # Process in reverse order
    for i in reversed(range(len(rewards))):
        # For the last step, next_value is 0
        # For other steps, it's the value of the next state
        if i < len(rewards) - 1:
            next_value = values[i + 1]
        
        # Calculate TD error: reward + discount * next_value - current_value
        delta = rewards[i] + gamma * next_value - values[i]
        
        # Calculate GAE
        gae = delta + gamma * lam * gae
        
        # Insert at the beginning to maintain correct order
        returns.insert(0, gae + values[i])
        advantages.insert(0, gae)
    
    return np.array(returns), np.array(advantages)

def train_ppo(env, agent, num_episodes=1000, steps_per_episode=200, update_interval=2000):
    # Storage for training metrics
    episode_rewards = []
    all_rewards = []
    update_count = 0
    
    # Storage for experiences
    states = []
    actions = []
    log_probs = []
    rewards = []
    values = []
    
    state = env.reset()
    episode_reward = 0
    episode_count = 0
    steps_taken = 0
    
    print("Starting training...")
    
    while episode_count < num_episodes:
        # Get action from agent
        action, log_prob = agent.get_action(state)
        
        # Get value estimate
        value = agent.value(torch.FloatTensor(state)).item()
        
        # Take action in environment
        next_state, reward, done, _ = env.step(action)
        
        # Store experience
        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        values.append(value)
        
        # Update state and counters
        state = next_state
        episode_reward += reward
        steps_taken += 1
        
        # End of episode or update interval reached
        if steps_taken % steps_per_episode == 0 or steps_taken % update_interval == 0:
            # Store episode reward
            episode_rewards.append(episode_reward)
            all_rewards.extend(rewards)
            
            # Print progress
            if len(episode_rewards) % 10 == 0:
                print(f"Episode {episode_count}, Average Reward: {np.mean(episode_rewards[-10:]):.2f}")
            
            # Reset episode counters
            episode_reward = 0
            episode_count += 1
            
            # Reset environment
            state = env.reset()
        
        # Update policy if enough steps collected
        if steps_taken % update_interval == 0:
            # Calculate returns and advantages
            returns, advantages = compute_returns_advantages(rewards, values)
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Update agent
            update_info = agent.update(states, actions, log_probs, returns, advantages)
            
            # Print update info
            print(f"Update {update_count}, Policy Loss: {update_info['policy_loss']:.4f}, Value Loss: {update_info['value_loss']:.4f}")
            
            # Clear experience buffer
            states = []
            actions = []
            log_probs = []
            rewards = []
            values = []
            
            update_count += 1
            
            # Save checkpoint
            if update_count % 10 == 0:
                agent.save(f"ppo_agent_checkpoint_{update_count}.pt")
    
    return episode_rewards, all_rewards

def evaluate_tfs(env, agent, num_samples=1000):
    """Evaluate which TFs are most important for explaining lineage differences"""
    # Storage for TF selection counts
    tf_counts = np.zeros(220)
    tf_rewards = np.zeros(220)
    
    # Sample many states
    for _ in range(num_samples):
        # Reset to random state
        state = env.reset()
        
        # Get deterministic action
        action, _ = agent.get_action(state, deterministic=True)
        
        # Record which TFs were selected
        selected_indices = np.where(action == 1)[0]
        tf_counts[selected_indices] += 1
        
        # Calculate reward contribution for each selected TF
        reward = env._calculate_reward(action)
        reward_per_tf = reward / len(selected_indices) if len(selected_indices) > 0 else 0
        tf_rewards[selected_indices] += reward_per_tf
    
    # Calculate final scores
    tf_scores = tf_rewards / (tf_counts + 1e-10)  # Avoid division by zero
    
    # Get top TFs
    top_indices = np.argsort(tf_scores)[::-1]
    
    return top_indices, tf_scores, tf_counts

def main():
    # Load data
    print("Loading data...")
    pb_expressions, gc_expressions, pb_curvatures, gc_curvatures, pb_forces, gc_forces = load_data()
    
    # Create environment
    env = TFSelectionEnv(pb_expressions, gc_expressions, pb_curvatures, gc_curvatures, pb_forces, gc_forces)
    
    # Create agent
    agent = PPOAgent(env.state_dim, env.action_dim)
    
    # Train agent
    episode_rewards, all_rewards = train_ppo(env, agent, num_episodes=500, steps_per_episode=200)
    
    # Evaluate TF importance
    print("Evaluating TF importance...")
    top_tfs, tf_scores, tf_counts = evaluate_tfs(env, agent)
    
    # Print top 20 TFs
    print("\nTop 20 lineage-defining TFs:")
    for i, tf_idx in enumerate(top_tfs[:20]):
        print(f"{i+1}. TF-{tf_idx}: Score = {tf_scores[tf_idx]:.4f}, Selected {tf_counts[tf_idx]} times")
    
    # Plot training progress
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(episode_rewards)), episode_rewards)
    plt.title('PPO Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('training_progress.png')
    
    # Plot top TF scores
    plt.figure(figsize=(12, 6))
    plt.bar(range(20), [tf_scores[idx] for idx in top_tfs[:20]])
    plt.xticks(range(20), [f"TF-{idx}" for idx in top_tfs[:20]], rotation=45)
    plt.title('Top 20 Lineage-Defining TF Scores')
    plt.tight_layout()
    plt.savefig('top_tf_scores.png')
    
    # Save final results
    np.save('top_tfs.npy', top_tfs)
    np.save('tf_scores.npy', tf_scores)
    agent.save('final_ppo_model.pt')
    
    print("Training complete. Results saved.")

if __name__ == "__main__":
    main()