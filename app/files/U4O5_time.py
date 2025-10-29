import matplotlib.pyplot as plt

# Data for the first scenario: 2 User Attributes, 3 Object Attributes
logs_2ua_3oa = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000]
time_2ua_3oa = [1.4923, 1.6030, 1.7607, 1.9029, 1.9036, 2.2191, 3.1036, 4.2889, 4.4096, 4.5599, 4.2228, 5.0539, 6.1312]

# Data for the second scenario: 4 User Attributes, 5 Object Attributes
logs_4ua_5oa = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000]
time_4ua_5oa = [3.3882, 5.0303, 5.7774, 5.4513, 6.3075, 6.1793, 7.4654, 8.1880, 12.1394, 14.8671, 21.7779, 29.8260, 46.5459]

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the data for both scenarios
plt.plot(logs_2ua_3oa, time_2ua_3oa, marker='o', linestyle='--', label='UA:2, OA:3 (Less Complex)')
plt.plot(logs_4ua_5oa, time_4ua_5oa, marker='s', linestyle='--', label='UA:4, OA:5 (More Complex)')

# Add titles and labels for clarity
plt.title('')
plt.xlabel('Number of Log Entries')
plt.ylabel('Time Taken (seconds)')

# Add a legend to identify the lines
plt.legend()

# Add a grid for better readability
plt.grid(True)

# Display the plot
plt.show()