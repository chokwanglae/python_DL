import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "ann"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
    if tight_layout:
        plt.tight_layout()
    if os.path.exists(path) is False:
        os.makedirs(path)
    plt.savefig(path+"\\"+fig_id + ".png", format='png', dpi=300)

def logistic(z):
    return 1/(1+np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def derivative(f, z, eps = 0.000001):
    return (f(z+eps) - f(z-eps))/(2*eps)

z = np.linspace(-5, 5 ,200)
plt.figure(figsize=(11, 4))
plt.subplot(121)
plt.plot(z, np.sign(z), "r-", linewidth=2, label="step")
plt.plot(z, logistic(z), "g--", linewidth=2, label="Logistic")
plt.plot(z, np.tanh(z), "b-", linewidth=2, label="Tanh")
plt.plot(z, relu(z), "m-", linewidth=2, label="ReLU")

plt.grid(True)

plt.legend(loc="center right", fontsize=14)
plt.title("active_function", fontsize=14)
plt.axis([-5, 5, -1.2, 1.2])

save_fig("activation_functions_plot")
plt.show()
