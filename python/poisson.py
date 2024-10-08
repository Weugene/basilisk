from __future__ import annotations

from math import *

import matplotlib.pyplot as plt
import numpy as np

import stream as bas

N = 256
bas.init_grid(N)

a = bas.scalar()
b = bas.scalar()

a.f = lambda x, y: 0.0
b.f = lambda x, y: sin(2.0 * pi * x) * cos(2.0 * pi * y)


bas.poisson(a, b)

x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)
plt.imshow(a.f(X, Y))
plt.show()
