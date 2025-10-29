import matplotlib.pyplot as plt

# Data for execution time based on User Attributes (Object Attributes = 2)
logs_ua = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200]
user_attr_2_time = [1.2259, 1.5207, 1.5270, 1.6715, 1.6106, 1.9256, 2.5530, 3.3166, 3.4833, 3.3984, 2.8284]
user_attr_3_time = [1.7468, 1.8456, 1.8313, 1.9494, 2.1001, 2.5549, 3.1118, 3.3867, 4.3553, 5.0274, 4.4688]
logs_ua4 = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200]
user_attr_4_time = [2.5059, 2.6054, 3.0660, 3.1271, 3.1172, 2.9576, 4.0937, 5.8425, 5.7276, 7.6788, 8.9589]

plt.figure(figsize=(10, 6))
plt.plot(logs_ua, user_attr_2_time, marker='o', linestyle='--', label='2 User Attributes')
plt.plot(logs_ua, user_attr_3_time, marker='s', linestyle='--', label='3 User Attributes')
plt.plot(logs_ua4, user_attr_4_time, marker='^', linestyle='--', label='4 User Attributes')

plt.title('')
plt.xlabel('Number of Log Entries')
plt.ylabel('Time (seconds)')
plt.legend()
plt.grid(True)
plt.show()