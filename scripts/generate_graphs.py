import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure output directory exists
os.makedirs('data/training', exist_ok=True)

# Data from previous computation
timesteps_s0 = [4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768, 36864, 40960, 45056, 49152, 53248, 57344, 61440, 65536, 69632, 73728, 77824, 81920, 86016, 90112, 94208, 98304, 102400, 106496, 110592, 114688, 118784, 122880, 126976, 131072, 135168, 139264, 143360, 147456, 151552, 155648, 159744, 163840, 167936, 172032, 176128, 180224, 184320, 188416, 192512, 196608, 200704, 204800, 208896, 212992, 217088, 221184, 225280, 229376, 233472, 237568, 241664, 245760, 249856, 253952, 258048, 262144, 266240, 270336, 274432, 278528, 282624, 286720, 290816, 294912, 299008, 303104, 307200, 311296, 315392, 319488, 323584, 327680, 331776, 335872, 339968, 344064, 348160, 352256, 356352, 360448, 364544, 368640, 372736, 376832, 380928, 385024, 389120, 393216, 397312, 401408, 405504, 409600, 413696, 417792, 421888, 425984, 430080, 434176, 438272, 442368, 446464, 450560, 454656, 458752, 462848, 466944, 471040, 475136, 479232, 483328, 487424, 491520, 495616, 499712, 503808]

rewards_s0 = [1.9816, 2.2662, 2.5218, 2.6697, 2.7843, 3.0362, 3.1829, 3.2064, 3.3218, 3.4124, 3.4617, 3.6297, 3.5464, 3.5433, 3.5548, 3.4885, 3.4833, 3.4753, 3.4237, 3.4381, 3.3959, 3.3860, 3.3664, 3.3107, 3.3162, 3.3123, 3.2932, 3.2596, 3.2722, 3.2546, 3.2384, 3.2205, 3.2064, 3.2069, 3.1885, 3.1777, 3.1471, 3.1614, 3.1550, 3.1435, 3.1415, 3.1264, 3.1269, 3.0696, 3.1023, 3.1087, 3.0068, 3.0440, 3.0928, 3.0819, 3.0814, 3.0758, 3.0705, 3.0639, 2.9821, 3.0448, 3.0546, 2.1825, 2.3753, 2.6079, 2.9250, 2.9832, 3.0065, 3.0387, 3.0470, 3.0420, 3.0451, 3.0039, 3.0361, 2.5086, 2.5466, 2.7737, 2.9760, 2.9300, 2.9821, 3.0047, 3.0295, 3.0281, 3.0294, 3.0268, 3.0317, 3.0257, 3.0272, 3.0188, 3.0192, 3.0174, 3.0149, 3.0163, 3.0099, 3.0142, 3.0124, 3.0114, 3.0099, 3.0085, 3.0072, 2.9983, 2.9918, 2.9983, 2.9998, 2.9992, 2.9977, 2.9906, 2.9958, 1.9820, 2.1754, 2.7706, 2.9892, 3.0051, 3.0035, 2.9616, 3.0013, 2.9834, 2.9989, 2.9982, 2.9975, 2.9961, 2.9954, 1.9745, 2.3300, 2.5273, 2.7618, 2.7363, 2.6090]

timesteps_s05 = [4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768, 36864, 40960, 45056, 49152, 53248, 57344, 61440, 65536, 69632, 73728, 77824, 81920, 86016, 90112, 94208, 98304, 102400, 106496, 110592, 114688, 118784, 122880, 126976, 131072, 135168, 139264, 143360, 147456, 151552, 155648, 159744, 163840, 167936, 172032, 176128, 180224, 184320, 188416, 192512, 196608, 200704, 204800, 208896, 212992, 217088, 221184, 225280, 229376, 233472, 237568, 241664, 245760, 249856, 253952, 258048, 262144, 266240, 270336, 274432, 278528, 282624, 286720, 290816, 294912, 299008, 303104, 307200, 311296, 315392, 319488, 323584, 327680, 331776, 335872, 339968, 344064, 348160, 352256, 356352, 360448, 364544, 368640, 372736, 376832, 380928, 385024, 389120, 393216, 397312, 401408, 405504, 409600, 413696, 417792, 421888, 425984, 430080, 434176, 438272, 442368, 446464, 450560, 454656, 458752, 462848, 466944, 471040, 475136, 479232, 483328, 487424, 491520, 495616, 499712, 503808]

rewards_s05 = [-3.1092, -3.0847, -2.9994, -3.0771, -2.9708, -2.9057, -2.9134, -2.9795, -2.9199, -2.7964, -2.7536, -2.8199, -2.6772, -2.6345, -2.6238, -2.5907, -2.5905, -2.5036, -2.5981, -2.4213, -2.4339, -2.3474, -2.2378, -2.2895, -2.2011, -2.2785, -2.2583, -2.0950, -2.0810, -2.1701, -2.0423, -1.9633, -2.0581, -2.0569, -1.9858, -1.9648, -1.8083, -1.8054, -1.9131, -1.8261, -1.8462, -1.8665, -1.7623, -1.7357, -1.8349, -1.6166, -1.6586, -1.6613, -1.4943, -1.6336, -1.7057, -1.5250, -1.5216, -1.4792, -1.5229, -1.5229, -1.4322, -1.4235, -1.6154, -1.4230, -1.4767, -1.4176, -1.3878, -1.4109, -1.2979, -1.2672, -1.4565, -1.3608, -1.3345, -1.2622, -1.3529, -1.2792, -1.2877, -1.3644, -1.1908, -1.1998, -1.3598, -1.1662, -1.2313, -1.1787, -1.0817, -1.1708, -1.1342, -1.1494, -1.0776, -1.1942, -1.2203, -1.1512, -1.2069, -1.1100, -1.1135, -1.1218, -1.0200, -1.0350, -1.1083, -1.1350, -1.0222, -1.0972, -0.9815, -1.2049, -0.9737, -1.0409, -1.0814, -1.1764, -0.9601, -1.0510, -1.0871, -0.9425, -1.0827, -1.1319, -1.1019, -1.0462, -0.9038, -1.1169, -1.0131, -0.9498, -0.9728, -1.0007, -0.7627, -0.9665, -0.8197, -0.8760, -0.8163]

# Convert to numpy arrays
ts0 = np.array(timesteps_s0)
r0 = np.array(rewards_s0)
ts05 = np.array(timesteps_s05)
r05 = np.array(rewards_s05)

# ===== Figure 1: Training Curves Comparison (A-1 & B-2) =====
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10))
fig.suptitle('Stage 0 vs Stage 0.5: PPO Training Curves Comparison', fontsize=16, fontweight='bold', y=0.995)

# A-1: Stage 0
ax1.plot(ts0 / 1000, r0, 'o-', color='#1f77b4', linewidth=2.5, markersize=5, label='ep_rew_mean', alpha=0.8)
ax1.axhline(y=np.mean(r0), color='green', linestyle='--', linewidth=2, label=f'Full mean = {np.mean(r0):.4f}', alpha=0.7)
ax1.axhline(y=np.mean(r0[-50:]), color='orange', linestyle='--', linewidth=2, label=f'Last-50 mean = {np.mean(r0[-50:]):.4f}', alpha=0.7)
ax1.scatter([49.152], [3.6297], color='red', s=150, marker='*', zorder=5, label='Peak = 3.6297')
ax1.grid(True, alpha=0.3, linestyle=':')
ax1.set_ylabel('Episode Reward Mean', fontsize=11, fontweight='bold')
ax1.set_title('A-1. Stage 0: PPO Training Curve (R8 single scenario, 500K steps)', fontsize=12, fontweight='bold', pad=10)
ax1.legend(loc='lower right', fontsize=10, framealpha=0.95)
ax1.set_ylim([1.5, 3.8])
ax1.set_xlim([0, 510])
ax1.text(300, 1.8, 'Periodic crashes observed\nat timesteps 237K, 287K, 426K, 483K',
        fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
        ha='center', fontweight='normal')

# B-2: Stage 0.5
ax2.plot(ts05 / 1000, r05, 'o-', color='#2ca02c', linewidth=2.5, markersize=5, label='ep_rew_mean', alpha=0.8)
ax2.axhline(y=np.mean(r05), color='#ff7f0e', linestyle='--', linewidth=2, label=f'Full mean = {np.mean(r05):.4f}', alpha=0.7)
ax2.axhline(y=np.mean(r05[-50:]), color='#d62728', linestyle='--', linewidth=2, label=f'Last-50 mean = {np.mean(r05[-50:]):.4f}', alpha=0.7)
ax2.scatter([487.424], [-0.7627], color='darkred', s=150, marker='*', zorder=5, label='Best = -0.7627')
ax2.grid(True, alpha=0.3, linestyle=':')
ax2.set_xlabel('Timestep (K)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Episode Reward Mean', fontsize=11, fontweight='bold')
ax2.set_title('B-2. Stage 0.5: PPO Training Curve (50-scenario manifest, 500K steps)', fontsize=12, fontweight='bold', pad=10)
ax2.legend(loc='lower right', fontsize=10, framealpha=0.95)
ax2.set_ylim([-3.3, -0.5])
ax2.set_xlim([0, 510])
ax2.text(250, -1.6, 'Monotonic improvement: 56.6% of steps\nNo periodic crashes observed',
        fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
        ha='center', fontweight='normal')

plt.tight_layout()
plt.savefig('data/training/A1_B2_training_curves.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: data/training/A1_B2_training_curves.png")
plt.close()

# ===== Figure 2: PPO vs Greedy Comparison (A-2) =====
fig, ax = plt.subplots(figsize=(14, 7))

categories = ['serve_rate', 'mean_wait\n(ticks)', 'mean_ivt\n(ticks)', 'mean_detour\n(ticks)', 'total_reward']
greedy_r8 = [0.6250, 6.6000, 7.6000, 3.2000, 5.9633]
ppo_r8 = [1.0000, 5.3750, 5.6250, 1.1250, 15.4017]
greedy_r80 = [0.1875, 7.0667, 11.6000, 6.2000, -28.8767]
ppo_r80 = [0.1375, 6.6364, 10.1818, 4.7273, -41.7233]

x = np.arange(len(categories))
width = 0.2

bars1 = ax.bar(x - 1.5*width, greedy_r8, width, label='Greedy R8', color='#ff7f0e', alpha=0.85, edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x - 0.5*width, ppo_r8, width, label='PPO R8', color='#2ca02c', alpha=0.85, edgecolor='black', linewidth=1.2)
bars3 = ax.bar(x + 0.5*width, greedy_r80, width, label='Greedy R80', color='#d62728', alpha=0.85, edgecolor='black', linewidth=1.2)
bars4 = ax.bar(x + 1.5*width, ppo_r80, width, label='PPO R80', color='#1f77b4', alpha=0.85, edgecolor='black', linewidth=1.2)

ax.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
ax.set_title('A-2. Stage 0: PPO vs Greedy Baseline Comparison (R8 & R80)', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
ax.legend(loc='upper left', fontsize=11, framealpha=0.95, ncol=2)
ax.grid(True, axis='y', alpha=0.3, linestyle=':')
ax.axhline(y=0, color='black', linewidth=0.8)

# Add value labels on bars
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        if abs(height) > 0.1:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=8.5, fontweight='bold')

plt.tight_layout()
plt.savefig('data/training/A2_ppo_vs_greedy.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: data/training/A2_ppo_vs_greedy.png")
plt.close()

# ===== Figure 3: Train vs Val Generalization (B-3) =====
fig, ax = plt.subplots(figsize=(12, 7))

metrics_b3 = ['serve_rate', 'mean_wait', 'total_reward']
train_vals = [0.2850, 7.3283, -5.7312]
val_vals = [0.2778, 7.3246, -7.2023]

x_b3 = np.arange(len(metrics_b3))
width_b3 = 0.35

bars_train = ax.bar(x_b3 - width_b3/2, train_vals, width_b3, label='Train (50 scenarios)',
                    color='#1f77b4', alpha=0.85, edgecolor='black', linewidth=1.2)
bars_val = ax.bar(x_b3 + width_b3/2, val_vals, width_b3, label='Val (10 scenarios)',
                  color='#ff7f0e', alpha=0.85, edgecolor='black', linewidth=1.2)

ax.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
ax.set_title('B-3. Stage 0.5: Train vs Val Generalization Check (PPO 500K)', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x_b3)
ax.set_xticklabels(metrics_b3, fontsize=11, fontweight='bold')
ax.legend(fontsize=11, framealpha=0.95)
ax.grid(True, axis='y', alpha=0.3, linestyle=':')
ax.axhline(y=0, color='black', linewidth=0.8)

# Add value labels
for bars in [bars_train, bars_val]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.4f}', ha='center', va='bottom' if height > 0 else 'top',
               fontsize=9, fontweight='bold')

# Add gap annotations
ax.text(0, max(train_vals[0], val_vals[0]) + 0.02, 'gap: +0.0072\n(2.6%)',
        ha='center', fontsize=9, color='red', fontweight='bold')
ax.text(2, max(train_vals[2], val_vals[2]) + 2, 'gap: +1.47\n(20.4%)',
        ha='center', fontsize=9, color='red', fontweight='bold')

plt.tight_layout()
plt.savefig('data/training/B3_train_vs_val.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: data/training/B3_train_vs_val.png")
plt.close()

# ===== Figure 4: Greedy Test Split Summary (B-4) =====
fig, ax = plt.subplots(figsize=(13, 7))

test_scenarios = ['s60', 's61', 's62', 's63', 's64', 's65', 's66', 's67', 's68', 's69', 'R8_orig']
serve_rates_b4 = [0.2222, 0.1944, 0.2361, 0.1806, 0.2361, 0.2361, 0.1944, 0.2083, 0.2361, 0.2083, 0.6250]
colors_b4 = ['#d62728'] * 10 + ['#2ca02c']

bars_b4 = ax.bar(test_scenarios, serve_rates_b4, color=colors_b4, alpha=0.85, edgecolor='black', linewidth=1.2)

ax.set_ylabel('Serve Rate', fontsize=12, fontweight='bold')
ax.set_title('B-4. Stage 0.5: Greedy Baseline on Test Split (11 scenarios)', fontsize=14, fontweight='bold', pad=15)
ax.set_ylim([0, 0.7])
ax.grid(True, axis='y', alpha=0.3, linestyle=':')
ax.axhline(y=0.2153, color='red', linestyle='--', linewidth=2, label='R80 mean = 0.2153', alpha=0.7)
ax.axhline(y=0.6250, color='green', linestyle='--', linewidth=2, label='R8 original = 0.6250', alpha=0.7)
ax.legend(fontsize=10, framealpha=0.95, loc='upper right')

# Add value labels
for bar in bars_b4:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
           f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add annotation
ax.text(5, 0.5, 'R80 scenarios: 0.1806 ~ 0.2361\n(narrow range, ~1.8% mean CV)',
        fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
        ha='center', fontweight='normal')

plt.tight_layout()
plt.savefig('data/training/B4_greedy_test_split.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: data/training/B4_greedy_test_split.png")
plt.close()

print("\n" + "="*60)
print("[OK] All 4 graphs generated successfully!")
print("="*60)
print("\nGenerated files:")
print("  1. A1_B2_training_curves.png -- Stage 0 vs Stage 0.5 comparison")
print("  2. A2_ppo_vs_greedy.png -- PPO vs Greedy baseline comparison")
print("  3. B3_train_vs_val.png -- Generalization check (train/val split)")
print("  4. B4_greedy_test_split.png -- Test split greedy baseline")
print("\nAll saved to: data/training/")
