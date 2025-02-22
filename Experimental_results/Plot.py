import matplotlib.pyplot as plt

# Create lists to store data for three lines
L_values = [[] for _ in range(3)]
power_values = [[] for _ in range(3)]
colors = ['dodgerblue', 'orange']  # Define line colors
labels = ['Estimated density ratio','True density ratio']  # Define line labels

# Read data from three different files
files = ['/gpfsnyu/home/bx2038/Covariate-shift-CRT/Results/Power_zdiff_est_n1000_0.1.txt', '/gpfsnyu/home/bx2038/Covariate-shift-CRT/Results/Power_zdiff_true_n1000_0.1.txt']

for i, filename in enumerate(files):
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('Zdiff:'):
                _, l_value, _, power_value = line.strip().split()
                L_values[i].append(l_value.replace(',', ''))
                power_values[i].append(float(power_value))

# Create a stylish plot
plt.figure(figsize=(10, 6))  # Set figure size

# Plot three lines
for i in range(2):
    plt.plot(L_values[i], power_values[i], marker='o', linestyle='-', color=colors[i],
             linewidth=2, markersize=8, label=labels[i])

plt.xlabel('Z difference', fontsize=14)  # X-axis label with fontsize
plt.ylabel('Power', fontsize=14)  # Y-axis label with fontsize
plt.ylim(0, 1)  # Set y-axis limits to 0 and 1
plt.title('Power vs Z difference', fontsize=16)  # Title with fontsize
plt.legend(fontsize=12)  # Legend with fontsize
plt.grid(True, linestyle='--', alpha=0.6)  # Add grid lines
plt.savefig('/gpfsnyu/home/bx2038/Covariate-shift-CRT/Plot/True_vs_estimate_zdiff_power.png', dpi=300, bbox_inches='tight')  # Save as a high-resolution image
plt.show()  # Show the plot