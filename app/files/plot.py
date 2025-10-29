import matplotlib.pyplot as plt

# --- Your Experimental Data ---

# ABAC Dataset
abac_log_sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2200]
abac_coverage_raw = [7/87, 28/177, 38/236, 123/310, 182/386, 430/471, 501/516, 545/571, 622/630, 670/670, 806/807, 832/832, 938/938, 1026/1026, 1037/1037, 1118/1118]
abac_coverage_percent = [c * 100 for c in abac_coverage_raw]

# DAC Dataset
dac_log_sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 5000]
dac_coverage_raw = [6/80, 23/145, 39/181, 82/223, 119/269, 159/269, 172/308, 199/320, 186/337, 213/342, 233/360, 250/374, 255/385, 251/384, 267/395, 244/398, 274/418, 254/419]
dac_coverage_percent = [c * 100 for c in dac_coverage_raw]
# Note: your DAC data has an interesting dip and recovery. This plot will show it well.
# Correcting the 84% and 86% values from your text to the calculated values for accuracy.
dac_coverage_percent[15] = (244/398) * 100 # This is ~61.3%, not 84%
dac_coverage_percent[16] = (274/418) * 100 # This is ~65.5%, not 86%
dac_coverage_percent[17] = (254/419) * 100 # This is ~60.6%, not 86%


# --- Create the Plot ---
plt.style.use('seaborn-v0_8-whitegrid') # Use a nice academic style
fig, ax = plt.subplots(figsize=(8, 5)) # Set the figure size

# Plot both lines on the same graph for easy comparison
ax.plot(abac_log_sizes, abac_coverage_percent, marker='o', linestyle='-', label='ABAC Dataset')
ax.plot(dac_log_sizes, dac_coverage_percent, marker='s', linestyle='--', label='DAC Dataset')

# --- Customize for a Professional, Publication-Quality Look ---
ax.set_title('Policy Miner Performance: Coverage vs. Log Size', fontsize=14, fontweight='bold')
ax.set_xlabel('Number of Log Entries', fontsize=12)
ax.set_ylabel('Permission Coverage (\%)', fontsize=12)

# Set the y-axis to go from 0 to 105 for better presentation
ax.set_ylim(0, 105)
ax.set_xlim(0, max(max(abac_log_sizes), max(dac_log_sizes)) + 200) # Give some space on the x-axis

# Add a legend to identify the lines
ax.legend(fontsize=11)

# Make ticks larger
ax.tick_params(axis='both', which='major', labelsize=10)

# --- Save the Figure and Show the Plot ---
# This saves the plot as a high-quality PDF file, perfect for Overleaf
plt.savefig("coverage_comparison.pdf", format='pdf', bbox_inches='tight')

print("Graph saved as coverage_comparison.pdf")

# This will display the plot on your screen when you run the script
plt.show()