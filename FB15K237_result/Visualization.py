import numpy as np
import matplotlib.pyplot as plt

labels1 = np.genfromtxt("C:/Users/14114/Desktop/npresult1.txt", dtype=np.float32)
labels2 = np.genfromtxt("C:/Users/14114/Desktop/npresult2.txt", dtype=np.float32)

for index in range(3):
    name_list = ['', '', '', '', '', '', '']
    num_list = labels1[index]
    num_list1 = labels2[index]
    x = list(range(len(num_list)))
    total_width, n = 0.8, 2
    width = total_width / n
    plt.bar(x, num_list, width=width, label='first', fc='y')
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, num_list1, width=width, label='second', tick_label=name_list, fc='r')
    plt.legend()
    plt.show()