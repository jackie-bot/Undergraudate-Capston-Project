import numpy as np
import matplotlib.pyplot as plt

# 模拟数据：(500 episodes, 6 UE counts) for Rewards, Latency, Success Rate
episodes = np.arange(500)
ue_counts = [100, 150, 200, 300, 400, 500]

# 模拟数据（这里仅为示例，需替换为真实数据）
np.random.seed(42)
rewards = [np.random.normal(loc=600-50*i/500, scale=50, size=500) for i in range(6)]  # 逐渐稳定
latency = [np.random.normal(loc=300-30*i/500, scale=30, size=500) for i in range(6)]  # 逐渐稳定
success_rate = [np.random.normal(loc=0.99, scale=0.005, size=500) for i in range(6)]  # 稳定在 0.98 以上

# 平滑数据（移动平均）
window_size = 20
def smooth_data(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# 创建子图
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), sharex=True)

# 绘制 Rewards
for i, ue in enumerate(ue_counts):
    smoothed_rewards = smooth_data(rewards[i], window_size)
    ax1.plot(episodes[:len(smoothed_rewards)], smoothed_rewards, label=f"{ue} UE")
ax1.set_title("Episode Rewards vs. Episodes")
ax1.set_xlabel("Episode Number")
ax1.set_ylabel("Total Reward")
ax1.legend()
ax1.grid(True)

# 绘制 Latency
for i, ue in enumerate(ue_counts):
    smoothed_latency = smooth_data(latency[i], window_size)
    ax2.plot(episodes[:len(smoothed_latency)], smoothed_latency, label=f"{ue} UE")
ax2.set_title("Average Latency vs. Episodes")
ax2.set_xlabel("Episode Number")
ax2.set_ylabel("Average Latency (ms)")
ax2.legend()
ax2.grid(True)

# 绘制 Success Rate
for i, ue in enumerate(ue_counts):
    smoothed_success = smooth_data(success_rate[i], window_size)
    ax3.plot(episodes[:len(smoothed_success)], smoothed_success, label=f"{ue} UE")
ax3.set_title("Success Rate vs. Episodes")
ax3.set_xlabel("Episode Number")
ax3.set_ylabel("Success Rate")
ax3.set_ylim(0.95, 1.00)
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.show()