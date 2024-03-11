import numpy as np
import torch as tc
import matplotlib.pyplot as plt
import seaborn as sns
from bptt.models import Model
from torch.utils.data import Dataset, DataLoader , Subset
#from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import find_peaks
from itertools import combinations
import h5py
import os

# CRCNS evaluation functions
#
def calculate_coincidences(spike_train_data, spike_train_model, delta=0.004):
    """
    Returns number of coincidences of spikes between two spike trains with precision delta

    arguments:
        spike_train_data: spike train from reference/target data
        spike_train_model: spike train from prediction by model
        delta: precision of coincidence in seconds
    returns:
        Ncoinc: number of coincidences of the two spike trains
    """
    idx_a = 0
    idx_b = 0

    mask_a = np.zeros_like(spike_train_model)

    len_a = len(spike_train_model)
    len_b = len(spike_train_data)

    while idx_a < len_a and idx_b < len_b:
        val_a = spike_train_model[idx_a]
        val_b = spike_train_data[idx_b]

        diff = abs(val_a - val_b)
        if diff <= delta:
            mask_a[idx_a] = 1

        if val_a == val_b:
            idx_a += 1
            idx_b += 1
        else:
            if val_a < val_b:
                idx_a += 1
            else:
                idx_b += 1
    Ncoinc = np.sum(mask_a)

    return Ncoinc


def calculate_gamma(Ncoinc, Ndata, Nmodel, v, delta):
    exp_Ncoinc = 2 * v * delta * Ndata
    alpha = 1 - 2 * v * delta
    gamma = (1/alpha) * abs(Ncoinc - exp_Ncoinc) / (0.5 * (Ndata + Nmodel))
    return gamma

def calculate_performance(gammas, R):
    PA = np.mean(gammas) / R
    return PA

def calculate_reliability(all_data_spike_models, exp_duration, delta):
    Nrep = len(all_data_spike_models)
    gammas = []

    for i, j in combinations(range(Nrep), 2):
        v = len(all_data_spike_models[i])/exp_duration

        Ncoinc = calculate_coincidences(all_data_spike_models[i], all_data_spike_models[j], delta)

        gamma = calculate_gamma(Ncoinc, len(all_data_spike_models[i]), len(all_data_spike_models[j]), v, delta)
        gammas.append(gamma)

    R =  np.mean(gammas)      

    print("----calculate reliability----")
    print("Type of gammas:", type(gammas))
    print("Length of gammas:", len(gammas))
    return R


def bootstrap_error(gammas, R, n=10000):
    Nrep = len(gammas)
    PA = calculate_performance(gammas, R)
    resampled_means = [np.mean(np.random.choice([gamma/R for gamma in gammas], Nrep)) for _ in range(n)]
    #resampled_means = [np.mean(np.random.choice(gammas/R, Nrep)) for _ in range(n)]
    ErrA = np.sqrt(sum([(mu - PA)**2 for mu in resampled_means]) / (n - 1))
    return ErrA

def calculate_data_spike_train(testX, testY):
    # Load the data
    stimulus = testX
    voltage = testY

    # Convert voltage to spikes
    #spikes = np.where(voltage > 0 , 1 , 0)  # replace 0 with your threshold
        
    data_peaks, _ = find_peaks(voltage, height=0.6)
    spikes = np.zeros_like(voltage)
    spikes[data_peaks] = 1

    # Calculate spike times
    spike_times = np.where(spikes == 1)[0]

    # Calculate mean firing rate
    mean_firing_rate = len(spike_times) / len(stimulus)

    # Calculate inter-spike intervals
    isi = np.diff(spike_times)

    # Number of occurring spike times
    num_spikes = len(spike_times)
        
    # Duration of the experiment
    T = len(stimulus)  # replace with the appropriate time unit
            
    # Create subplots
    #fig, axs = plt.subplots(3, figsize=(10, 6))

    # Plot voltage
    #axs[0].plot(voltage)
    #axs[0].set_title('Voltage')
    #axs[0].set_xlabel('Time (ms)')
    #axs[0].set_ylabel('Voltage')
    #axs[0].set_xlim([0, T])

            
    # Plot stimulus
    #axs[1].plot(stimulus)
    #axs[1].set_title('Stimulus')
    #axs[1].set_xlabel('Time (ms)')
    #axs[1].set_ylabel('Stimulus')
    #axs[1].set_xlim([0, T])

    # Plot spike train
    #axs[2].eventplot(spike_times, color='black')
    #axs[2].set_title('Spike Train')
    #axs[2].set_xlabel('Time (ms)')
    #axs[2].set_ylabel('Spikes')
    #axs[2].set_yticks([])
    #axs[2].set_xlim([0, T])

    # Show the plot
    #plt.tight_layout()
    #plt.show()
    #fig.savefig('endplrnn4neuron/BPTT_TF/plots/%s/data_spike_train_%s.png'%(rep,rep))

    # Print statistics
    print(f"-----data spike train repetition: {rep} -----")
    #$print(f"Mean firing rate: {mean_firing_rate} Hz")
    #print(f"Inter-spike intervals: {isi} ms")
    print(f"Number of occurring spike times: {num_spikes}")
    #print(f"spike times: {data_peaks}")
    #print(f"Duration of the experiment: {T}")

    return data_peaks


def calculate_PA(all_data_spikes, all_model_gammas):

    R = calculate_reliability(all_data_spikes,43001, 0.004*fs)

    PA = calculate_performance(all_model_gammas, R)

    ErrA = bootstrap_error(all_model_gammas, R, n=10000)

    end_result = PA - ErrA

    print("Mean of PA: ", np.mean(PA))
    print("PA: ", PA)
    print("R: ", R)
    print("ErrA: ", ErrA)
    print("End result: ", end_result)

    return end_result


mode = 'crcns_data'  # 'real_data'

all_data_spikes = []
all_model_gammas= []
best_model_performance = []
nr = '050'
epoch_nr = '5000'

for i in range(13):
    rep = str(i).zfill(2)  # 00 to 12

    data_path = "Experiments/my_exp/crcns_data/crcns_test_stim_39.npy"
    
    print(f"Loaded data from {data_path}")

    model_path_gen = "results/neuron_w_input_crcns/M26B47tau05T500_39/"
    # Load your model here using model_path_gen

    data_path_train = f"Experiments/my_exp/{mode}/train_data{'_real' if mode=='real_data' else '_crcns'}{rep}.npy"
   
    print(f"Loaded training data from {data_path_train}")

    dname =  'challengeA%s'%(rep)  # 'challengeAfull'
    #file = h5py.File("../../single-neuron-rnn/data/%s.hdf5"%dname, "r")
    file_test = h5py.File("data/%s.hdf5"%dname, "r")
    
    file_train = h5py.File("data/challengeA_train%s.hdf5"%rep)

    trainX, trainY = np.array(file_train['train']).T
    testX, testY = np.array(file_test['test']).T

    Imin = np.min(trainX)/1000  #nA
    Imax = np.max(trainX)/1000  #nA

    vmin = np.min(trainY)
    vmax = np.max(trainY)

    # load data
    test_data = tc.tensor(np.load(data_path))
    train_data = tc.tensor(np.load(data_path_train))

    dt_s = 0.0001

    fs = int(1/dt_s)
    print(fs)

    pend = 40000

    #plt.plot(test_data[:pend,0])
    #plt.plot(train_data[:pend,0])

    #calculate the data spike train, for the current repetiton
    all_data_spikes.append(calculate_data_spike_train(trainX,trainY))

    #print(f"all data spikes: {all_data_spikes}")
    
    #all model entries
    model_entry = []

    # testing to get model spike train

    for epoch in [epoch_nr]:#np.arange(17000,18000,25): 1325 = crcns
        print(epoch)
        for j in [nr]:  #'001','002','003
            #nr = str(j+1).zfill(3)
            
            model_path = model_path_gen + nr
        
            # restore model checkpoint
            m = Model()
            m.init_from_model_path(model_path, epoch=epoch)
            m.eval()
        
            for k,dat in enumerate([train_data]):
                
                T = len(dat)
                inp = dat[:,1].unsqueeze(1)  #tc.tensor([[0]]).float()
                data = dat[:,0].unsqueeze(1)   #data.float()
                
                X, Z = m.generate_free_trajectory(inp,data.float() , T)   # for only spiketimes: tc.tensor([[0]]).float()

                #I_scaler = MinMaxScaler(feature_range=(0, 1)).fit(trainX.reshape(-1,1))
                #v_scaler = MinMaxScaler(feature_range=(0, 1)).fit(trainY.reshape(-1,1))
                #X_og = v_scaler.inverse_transform(X.reshape(-1,1)).flatten() # shape to original shape
                #Z_og = I_scaler.inverse_transform(Z.reshape(-1,1)).flatten() # shape to original shape
                #inp_og = I_scaler.inverse_transform(inp.reshape(-1,1)).flatten()
                #data_og = v_scaler.inverse_transform(data.reshape(-1,1)).flatten()

                max_val = tc.max(X)
                min_val = tc.min(X)

                # Calculate the middle value
                threshold = float((max_val + min_val) / 2)
                
                model_peaks, _ = find_peaks(np.squeeze(X), height=threshold)   # 0.5 for scaled data
                spikes = np.zeros_like(np.squeeze(X))
                spikes[model_peaks] = 1
                
                #spikes = np.where(X > 0.5 , 1 , 0)

                spike_times = np.where(spikes == 1)[0]

                model_peaks_gt, _ = find_peaks(np.squeeze(data), height=threshold)   # 0.5 for scaled data
                spikes_gt = np.zeros_like(np.squeeze(data))
                spikes_gt[model_peaks_gt] = 1
                
                #spikes = np.where(X > 0.5 , 1 , 0)

                spike_times_gt = np.where(spikes_gt == 1)[0]

                #  spike_times = np.where(X > 0.5)[0]
                
                # Calculate mean firing rate
                mean_firing_rate = len(spike_times) / len(dat[:,1])
                
                # Calculate inter-spike intervals
                isi = np.diff(spike_times)
                
                # Number of occurring spike times
                num_spikes = len(spike_times)
                
                # Create subplots
                #fig, axs = plt.subplots(4, figsize=(10, 8))

                # Plot stimulus
                #axs[0].plot(inp_og)
                #axs[0].set_title('Stimulus')
                #axs[0].set_xlabel('Time (ms)')
                #axs[0].set_ylabel('$p_A$ (I)')
                #axs[0].set_xlim([0, T])

                # Plot voltage
                #axs[1].plot(np.linspace(0,len(X_og),len(X_og)),X_og,zorder=10,label='prediction')
                #axs[1].plot(np.linspace(0,len(X_og),len(X_og)),data_og,label='ground truth')
                #axs[1].axhline(threshold*(np.mean(X_og)), color='red', linestyle='--', label='threshold')
                #axs[1].set_title('Voltage')
                #axs[1].set_xlabel('Time (ms)')
                #axs[1].set_ylabel('$V_m$ (V)')
                #axs[1].set_xlim([0, T])
                #axs[1].legend()
            
                # Plot predicted spike train
                #axs[2].eventplot(spike_times, color='black')
                #axs[2].set_title('Predicted Spike Train')
                #axs[2].set_xlabel('Time (ms)')
                #axs[2].set_ylabel('Spikes')
                #axs[2].set_yticks([])
                #axs[2].set_xlim([0, T])

                # Plot ground truth spike train
                # Assuming ground_truth_spike_times is the array of spike times for the ground truth
                #ground_truth_spike_times =  spike_times_gt # replace with your data
                #axs[3].eventplot(ground_truth_spike_times, color='orange')
                #axs[3].set_title('Ground Truth Spike Train')
                #axs[3].set_xlabel('Time (ms)')
                #axs[3].set_ylabel('Spikes')
                #axs[3].set_yticks([])
                #axs[3].set_xlim([0, T])

                # Show the plot
                #plt.tight_layout()
                #plt.show()

                #np.save('dendplrnn4neuron/BPTT_TF/plots/%s/crcns_data_trace%s_%s.npy'%(rep,nr,rep),np.concatenate([X.T*(vmax-vmin)+vmin,Z.T]))

                # Print statistics
                print(f"Model {nr}, repetition {i} performance statistics")
               # print(f"Mean firing rate: {mean_firing_rate} Hz")
               # print(f"Inter-spike intervals: {isi} ms")
                print(f"Number of occurring spike times: {num_spikes}")
               # print(f"spike times: {model_peaks}")
               # print(f"Duration of the experiment: {T}")
                
                v = len(all_data_spikes[i])/T
                Ncoinc = calculate_coincidences(all_data_spikes[i], model_peaks, 0.004*fs)
                all_model_gammas.append(calculate_gamma(Ncoinc, len(all_data_spikes[i]), len(model_peaks), v, 0.004*fs))
                print(f"Gamma of model {nr}, repetition {i}  = {all_model_gammas[i]}")
            
    print(all_model_gammas)

print(f"Length of all the model gammas calculated: {len(all_model_gammas)}")
print(f"Length of all the data spikes calculated: {len(all_data_spikes)}")


best_model_performance = (calculate_PA(all_data_spikes, all_model_gammas))

print(f"The model {nr} predicts ST with a performance value of {best_model_performance}")

# save the results for the evaluation run
directory = "crcns_evaluation"
if not os.path.exists(directory):
    os.makedirs(directory)

# Save the lists in the created directory
np.save(directory + '/all_data_spikes_train.npy', all_data_spikes)
np.save(directory + '/all_model_gammas_train.npy', all_model_gammas)
np.save(directory + '/best_model_performance_train.npy', best_model_performance)

print(f"Data saved in directory: {directory} ")
