import numpy as np
import plots as p

# o = np.array([0.44, 0.49, 0.48, 0.51, 0.515, 0.52, 0.505, 0.51, 0.52, 0.525])
# s_and_p = np.array([0.44, 0.47, 0.49, 0.51, 0.515, 0.52, 0.505, 0.50, 0.53, 0.538])
# p.plot_learning_curves(o, s_and_p)
# np.save('original', o)
original = np.load('../output/cnn_original.npy')
# s_and_p = np.load('../output/cnn_s_and_p.npy')
reflection = np.load('../output/cnn_reflection.npy')
# rotation = np.load('../output/cnn_rotation.npy')
# p.plot_learning_curves(original, s_and_p, rotation, reflection)
p.plot_learning_curves(original, reflection, original, original)
print("done")
