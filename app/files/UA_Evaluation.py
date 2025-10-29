import matplotlib.pyplot as plt

# Data for User Attributes
logs = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
user_attr_2_coverage = [0, 0, 0, 70, 95, 98, 100, 100, 100, 100]
user_attr_3_coverage = [0, 0, 0, 75, 89, 100, 100, 100, 100, 100]
user_attr_4_coverage = [0, 0, 0, 0, 0, 12, 63, 96, 95, 94]


plt.figure(figsize=(10, 6))
plt.plot(logs, user_attr_2_coverage, marker='o', label='2 User Attributes')
plt.plot(logs, user_attr_3_coverage, marker='s', label='3 User Attributes')
plt.plot(logs, user_attr_4_coverage, marker='^', label='4 User Attributes')

plt.title('')
plt.xlabel('Number of Log Entries')
plt.ylabel('Coverage (%)')
plt.legend()
plt.grid(True)
plt.show()