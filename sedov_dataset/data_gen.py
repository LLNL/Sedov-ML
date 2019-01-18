from sedov_main_function import sedov_var_initial_energy_time_gamma
import multiprocessing as mp
import numpy as np
import pickle
import pandas as pd



# gamma_vec = np.arange(1.1,2,0.1)
# gamma_vec = np.array([1.4])
d_gamma= 0.01
gamma_vec = np.arange(1.2,2.0+d_gamma,d_gamma)
def worker(gamma_indx):
    # call the function and return the data
    geometry = 3.0
    n_steps = 2400
    zmax = 4.0
    density_power = 0.0  # uniform density
    output_file = 'test'
    gamma = gamma_vec[gamma_indx]
    energy = np.arange(0.1, 1.0, 0.01)
    time = np.arange(0.1, 10, 0.1)
    return_dict = sedov_var_initial_energy_time_gamma(geometry, density_power, energy, time,
                                                            gamma, output_file, position_hi=zmax, number_steps=n_steps,
                                                            rho_0=1.0)
    return return_dict

if __name__ == '__main__':
    Processes = mp.Pool()
    results = Processes.map(worker,range(0,gamma_vec.size))

    print("done!!")

    print(results)

    data = pd.DataFrame()

    for i in range(0, len(results)):
        temp = pd.DataFrame(results[i])
        data = data.append(temp, ignore_index=True)

    # output_file = open ('./multiprocessing/data_results','wb')
    data.to_csv('../Data_gamma_1.2_2.0_included',sep = ",",index = False)
    # pickle.dump(results,output_file)
    # output_file.close()