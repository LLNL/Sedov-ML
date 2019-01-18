import matplotlib.pyplot as plt
import pickle
import numpy as np
import matplotlib

matplotlib.rcParams.update({'font.size': 14})

with open('./animation_data','rb') as file:
    data = pickle.load(file)



position = data['position']
pressure = data['pressure']
time = data['time']

plt.show()
for i in range(13,len(position)):
    plt.semilogy(position[i], pressure[i], label='time: {:4.2f} seconds'.format(time[i]))
    # plt.semilogy(position[0], pressure[0], label='time: {:4.2f} seconds'.format(time[0]))
    plt.legend(loc='upper right')
    plt.ylim([0.01, 20])
    plt.xlim([0, 1])
    plt.xlabel('Position (cm)')
    plt.ylabel(r'Pressure (erg/cm$^3$)')
    props = dict(boxstyle='round', facecolor='w', alpha=0.5, edgecolor='grey')
    plt.text(0.47, 4.35, r'Initial Energy: 0.8451 erg', bbox=props)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig('./semilog_plots/image_t{:.3f}.png'.format(time[i]))
    plt.close()
    # pass

