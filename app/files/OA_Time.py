import matplotlib.pyplot as plt

# Data for execution time based on Object Attributes (User Attributes = 2)
logs_oa = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
obj_attr_2_time = [1.2259, 1.5207, 1.5270, 1.6715, 1.6106, 1.9256, 2.5530, 3.3166, 3.4833, 3.3984]
obj_attr_3_time = [1.4923, 1.6030, 1.7607, 1.9029, 1.9036, 2.2191, 3.1036, 4.2889, 4.4096, 4.5599]
obj_attr_5_time = [1.6116, 2.2638, 2.2994, 2.6654, 2.9571, 3.5551, 3.7724, 4.7969, 6.1406, 6.6604]

plt.figure(figsize=(10, 6))
plt.plot(logs_oa, obj_attr_2_time, marker='o', linestyle='--', label='2 Object Attributes')
plt.plot(logs_oa, obj_attr_3_time, marker='s', linestyle='--', label='3 Object Attributes')
plt.plot(logs_oa, obj_attr_5_time, marker='^', linestyle='--', label='5 Object Attributes')

plt.title('')
plt.xlabel('Number of Log Entries')
plt.ylabel('Time (seconds)')
plt.legend()
plt.grid(True)
plt.show()