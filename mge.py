import matplotlib.pyplot as plt
import numpy as np

# Data
categories = ["Predictions", "Confidence/Noise", "Documents"]
values_raw = [1209, 161, 4]   # Raw / Noise Removed / Documents
values_clean = [43, 35, 4]    # Cleaned / High Confidence / Documents (same for doc count)

x = np.arange(len(categories))  # the label locations
width = 0.35  # width of the bars

fig, ax = plt.subplots(figsize=(10,6))
bars1 = ax.bar(x - width/2, values_raw, width, label='Raw / Noise Removed', color='salmon')
bars2 = ax.bar(x + width/2, values_clean, width, label='Cleaned / High Confidence', color='skyblue')

# Labels and title
ax.set_ylabel('Count')
ax.set_title('Quality Improvement Summary Comparison')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

# Add value labels on top
for bar in bars1 + bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(bar.get_height()), ha='center', va='bottom')

plt.tight_layout()
plt.show()
