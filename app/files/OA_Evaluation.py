import matplotlib.pyplot as plt

# Data for Object Attributes
logs = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
obj_attr_2_coverage = [0, 0, 0, 70, 95, 98, 100, 100, 100, 100]
obj_attr_3_coverage = [0, 0, 32, 74, 89, 100, 100, 100, 100, 100]
obj_attr_5_coverage = [0, 0, 0, 0, 2, 12, 23, 76, 94, 91]


plt.figure(figsize=(10, 6))
plt.plot(logs, obj_attr_2_coverage, marker='o', label='2 Object Attributes')
plt.plot(logs, obj_attr_3_coverage, marker='s', label='3 Object Attributes')
plt.plot(logs, obj_attr_5_coverage, marker='^', label='5 Object Attributes')

plt.title('')
plt.ylabel('Coverage (%)')
plt.xlabel('Number of Log Entries')
plt.legend()
plt.grid(True)
plt.show()