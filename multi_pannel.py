import matplotlib.pyplot as plt
import numpy as np

# Data from Table 4.1: Action Distribution (Local, Edge, CPU+, CPU-)
ue_counts = [100, 150, 200, 300, 400, 500]
action_distributions = [
    [0.30, 0.25, 0.05, 0.40],  # 100 UEs
    [0.30, 0.25, 0.05, 0.40],  # 150 UEs
    [0.25, 0.20, 0.05, 0.50],  # 200 UEs
    [0.20, 0.15, 0.05, 0.60],  # 300 UEs
    [0.15, 0.10, 0.05, 0.70],  # 400 UEs
    [0.10, 0.05, 0.05, 0.80],  # 500 UEs
]
actions = ['Local', 'Edge', 'CPU+', 'CPU-']
colors = ["#1d7ce1", "#25eb5a", "#eff310", "#f84129"]  # Distinctive colors for actions

# Create a 2x3 subplot grid
fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharey=True)
axes = axes.flatten()  # Flatten for easier indexing

# Plot each action distribution
for i, (ue, dist) in enumerate(zip(ue_counts, action_distributions)):
    axes[i].bar(actions, dist, color=colors)
    axes[i].set_title(f'({chr(97 + i)}) {ue} UEs', fontsize=10)
    axes[i].set_ylim(0, 1)
    axes[i].set_ylabel('Probability', fontsize=8)
    for tick in axes[i].get_xticklabels():
        tick.set_rotation(45)
        tick.set_fontsize(8)

# Adjust layout to prevent overlap
plt.tight_layout()
fig.suptitle('Action Distribution Across Different UE Counts', fontsize=12, y=1.05)

# Add a shared legend
fig.legend(actions, loc='center right', bbox_to_anchor=(1.1, 0.5), fontsize=8)

# Save the figure
plt.savefig('action_distribution_multi_panel.png', dpi=300, bbox_inches='tight')
plt.show()