import matplotlib.pyplot as plt
#figure 3
''' 
# Your data: each tuple is one group of two values
data = [(160, 11242),
        (156,  9524),
        (128,  6955 )]

# Split into two series
series1 = [d[0] for d in data]
series2 = [d[1] for d in data]

# X locations for the groups
x = range(len(data))
width = 0.35  # width of each bar

# Plot the two bar series, offsetting them so they sit side by side
bars1 = plt.bar([i - width/2 for i in x], series1, width=width, label='Marr-Albus')
bars2 = plt.bar([i + width/2 for i in x], series2, width=width, label='Feed-forward')

# Add significance lines and stars between each pair
for i in x:
    bar1 = bars1[i]
    bar2 = bars2[i]
    # x-coordinates of the centers
    x1 = bar1.get_x() + bar1.get_width()/2
    x2 = bar2.get_x() + bar2.get_width()/2
    # choose a y position just above the taller bar
    y_max = max(bar1.get_height(), bar2.get_height())
    offset = y_max * 0.05
    y = y_max + offset

    # draw the horizontal line with little vertical ticks at the ends
    plt.plot([x1, x1, x2, x2],
             [y,   y+offset, y+offset, y],
             lw=1.5)

    # put a star in the middle
    plt.text((x1 + x2)/2, y + offset*1.1, "*",
             ha='center', va='bottom', fontsize=14)

# Label the x‐axis ticks with the first value of each tuple
plt.xticks(x, ['Linear','Square root', 'Characters'])

plt.xlabel('Trajectory Type')
plt.ylabel('Epochs')
plt.title('Convergence across models')
plt.legend()

plt.tight_layout()
plt.show() 
'''

#figure 4

# Data for the two bars
values = [0.16, 0.06]
labels = ['Marr-Albus', 'Feed forward']

# X positions for the bars
x = range(len(values))

# Create the bar chart
plt.bar(x, values)
plt.xticks(x, labels)
plt.ylabel('Loss')
plt.title('Character Trajectory Performance')

plt.tight_layout()
plt.show()


