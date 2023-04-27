"""Produces two random sets of numbers that can be clearly separated by a linear function.
Displays the dividing line and the two classes (red and blue) of these sets on the graph.
Determines to which class the values belong."""


import matplotlib.pyplot as plt
import numpy as np

N = 5
b = 3

x1 = np.random.random(N)
print(x1, '---x1 = np.random.random(N) --- it is random N vector', '\n')
b = np.random.randint(10)
print(b, '---b = np.random.randint(10)--- it is randint number from 10', '\n')
bv = [b/10 for i in range(N)]
print(bv, '---bv = [b/10 for i in range(N)]--- it is random vector b/10 (N-times)', '\n')
c = [np.random.randint(10)/10 for i in range(N)]
print(c, '---c = [np.random.randint(10)/10 for i in range(N)]---compare with bv', '\n')
x2 = x1 + c + 0.1 + b
print(x2, '---x2 = x1 + c + 0.1', '\n')
C1 = [x1, x2]
print(C1, '---C1 = [x1, x2]', '\n')

x1 = np.random.random(N)
c = [np.random.randint(10)/10 for i in range(N)]
x2 = x1 - c - 0.1 + b
print(x2, '---x2 = x1 - c - 0.1', '\n')
C2 = [x1, x2]
print(C2, '---C2 = [x1, x2]', '\n')

f = [0+b, 1+b]
g = [-1, 1]

W = np.array([-0.5, 0.5, -1])
for i in range(N):
    x = np.array([C2[0][i], C2[1][i], 1])
    y = np.dot(W, x)
    if y >= 0:
        print("class C1")
    else:
        print("class C2")

plt.scatter(C1[0][:], C1[1][:], s=20, c='red')
plt.scatter(C2[0][:], C2[1][:], s=10, c='blue')
plt.plot(f)
plt.plot(g)
plt.grid(True)
plt.show()

