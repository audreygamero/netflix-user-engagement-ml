# engagement_plot.py

"""
Simulate and visualize how personalized recommendations affect user engagement
"""

import numpy as np
import matplotlib.pyplot as plt

# Simulate watch time (hrs/week) before personalization
before_avg = np.random.normal(12, 3, 500)

# Simulate watch time after personalized recommendations (+3 hours boost)
after_avg = before_avg + np.random.normal(3, 1, 500)

# Plot
plt.figure(figsize=(8, 5))
plt.hist(before_avg, bins=30, alpha=0.6, label="Before", color='pink')
plt.hist(after_avg, bins=30, alpha=0.6, label="After", color='purple')
plt.xlabel("Watch Time (hrs/week)")
plt.ylabel("Number of Users")
plt.title("User Engagement: Before vs. After Personalized Recommendations")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Make sure visuals folder exists or change path
plt.savefig("visuals/engagement_comparison.png")
plt.show()