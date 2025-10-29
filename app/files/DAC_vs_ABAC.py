import matplotlib.pyplot as plt

# Data for ABAC vs DAC
logs_abac = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000]
abac_coverage = [0, 0, 32, 74, 89, 100, 100, 100, 100, 100, 100, 100, 100]
logs_dac = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000]
dac_coverage = [5, 20, 31, 76, 58, 75, 70, 89, 80, 79, 83, 88, 89]

plt.figure(figsize=(10, 6))
plt.plot(logs_abac, abac_coverage, marker='o', label='ABAC System Logs (2 User, 3 Object Attributes)')
plt.plot(logs_dac, dac_coverage, marker='s', label='DAC System Logs')

plt.title('')
plt.xlabel('Number of Log Entries')
plt.ylabel('Coverage (%)')
plt.legend()
plt.grid(True)
plt.show()