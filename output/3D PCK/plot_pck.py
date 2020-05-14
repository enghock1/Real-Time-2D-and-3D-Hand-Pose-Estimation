import numpy as np
import matplotlib.pyplot as plt

threshold_list = np.loadtxt('threshold.npy')
pck_baseline = np.loadtxt('pck_baseline.npy')
#pck_proposed = np.loadtxt('pck_proposed.npy')

threshold_list = [i*10 for i in threshold_list]
plt.figure()
plt.plot(threshold_list, pck_baseline, label='baseline (11.93mm)')
plt.plot(threshold_list, pck_proposed, label='proposed method (18.21 mm)')
plt.grid(linestyle=':')
plt.title('Real world testset')
plt.xlabel('error threshold (mm)')
plt.ylabel('3D PCK')
plt.legend(loc='lower right')
plt.savefig('3D_PCK_comparison.png')
