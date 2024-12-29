import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create the plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.axis('off')

# Define the steps and positions
steps = [
    "1) d = load_data",
    "2) x = feature_extraction(d)",
    "3) X = scaling_transform(x)",
    "4) y = model.predict(X)",
    "5) prediction = inverse_scaling_transform(y)"
]
y_positions = [5, 4, 3, 2, 1]

# Draw rectangles and add step text inside
rect_height = 0.8
space_between_arrows = 0.4  # Increase space between rectangles and arrows
for step, y_pos in zip(steps, y_positions):
    rect = patches.Rectangle((0.2, y_pos - rect_height / 2), 0.6, rect_height, edgecolor="black", facecolor="white")
    ax.add_patch(rect)
    ax.text(0.5, y_pos, step, fontsize=12, va='center', ha='center')

# Draw arrows with proper spacing
arrow_start_x = 0.5  # Center x for arrows
for i in range(len(steps) - 1):
    start_y = y_positions[i] - (rect_height / 2) - space_between_arrows +0.4
    end_y = y_positions[i + 1] + (rect_height / 2) + space_between_arrows +0.5
    ax.arrow(arrow_start_x, start_y, 0, end_y - start_y - space_between_arrows * 2, 
             head_width=0.05, head_length=0.1, fc='black', ec='black')

plt.title("Sequence Diagram of ML Inference Pipeline", fontsize=14)
plt.show()
