import numpy as np
import torch as tc
import matplotlib.pyplot as plt
import seaborn as sns
from bptt.models import Model
from torch.utils.data import Dataset, DataLoader , Subset
#from sklearn.decomposition import PCA
from scipy.signal import find_peaks
from itertools import combinations
import h5py

# CRCNS evaluation functions
#
def calculate_coincidences(spike_train_data, spike_train_model, delta=4):
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
nr = '034'
epoch_nr = 1625

test_data_path = np.load("Experiments/my_exp/crcns_data/crcns_answer_testset_39.npy", allow_pickle=True)

data_path = "Experiments/my_exp/crcns_data/crcns_test_stim_39.npy"
    
print(f"Loaded data from {data_path}")

model_path_gen = "results/neuron_w_input_crcns/M26B47tau05T500_39/"
# Load your model here using model_path_gen

data_path_train = "Experiments/my_exp/crcns_data/train_data_crcnsfull_val_39.npy"
   
print(f"Loaded training data from {data_path_train}")

for i in range(13):
    rep = str(i).zfill(2)  # 00 to 12

    dname = 'challengeAtest_39'
    #file = h5py.File("../../single-neuron-rnn/data/%s.hdf5"%dname, "r")
    file = h5py.File("data/%s.hdf5"%dname, "r")

    #trainX, trainY = np.array(file['train']).T
    testX, testY = np.array(file['test']).T

    Imin = np.min(testX)/1000  #nA
    Imax = np.max(testX)/1000  #nA

    vmin = np.min(testY)
    vmax = np.max(testY)

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
    all_data_spikes.append(test_data_path[i])

    #print(f"all data spikes: {all_data_spikes}")

    # testing to get model spike train

    for epoch in [epoch_nr]:#np.arange(17000,18000,25): 1325 = crcns
        print(epoch)
        for j in [nr]:  #'034'
            
            model_path = model_path_gen + nr
        
            # restore model checkpoint
            m = Model()
            m.init_from_model_path(model_path, epoch=epoch)
            m.eval()
        
            for k,dat in enumerate([test_data]):
                
                T = len(dat)
                inp = dat[:,1].unsqueeze(1)  #tc.tensor([[0]]).float()
                data = dat[:,0].unsqueeze(1)   #data.float()
                
                X, Z = m.generate_free_trajectory(inp,data.float() , T)   # for only spiketimes: tc.tensor([[0]]).float()

                max_val = tc.max(X)
                min_val = tc.min(X)

                # Calculate the middle value
                threshold = float((max_val + min_val) / 2)
            
                model_peaks, _ = find_peaks(np.squeeze(X), height=threshold)  
                spikes = np.zeros_like(np.squeeze(X))
                spikes[model_peaks] = 1

                #spikes = np.where(X > 0.5 , 1 , 0)

                spike_times = np.where(spikes == 1)[0]

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
                #axs[0].plot(dat[:,1])
                #axs[0].set_title('Stimulus')
                #axs[0].set_xlabel('Time (ms)')
                #axs[0].set_ylabel('Stimulus')
                #axs[0].set_xlim([0, T])

                # Plot voltage
                #axs[1].plot(np.linspace(0,len(X),len(X)),X,zorder=10,label='prediction')
                #axs[1].set_title('Voltage')
                #axs[1].set_xlabel('Time (ms)')
                #axs[1].set_ylabel('Voltage')
                #axs[1].set_xlim([0, T])

                # Plot predicted spike train
                #axs[2].eventplot(spike_times, color='black')
                #axs[2].set_title('Predicted Spike Train')
                #axs[2].set_xlabel('Time (ms)')
                #axs[2].set_ylabel('Spikes')
                #axs[2].set_yticks([])
                #axs[2].set_xlim([0, T])

                # Plot ground truth spike train
                # Assuming ground_truth_spike_times is the array of spike times for the ground truth
                #ground_truth_spike_times = test_data_path[i]  # replace with your data
                #axs[3].eventplot(ground_truth_spike_times, color='orange')
                #axs[3].set_title('Ground Truth Spike Train')
                #axs[3].set_xlabel('Time (ms)')
                #axs[3].set_ylabel('Spikes')
                #axs[3].set_yticks([])
                #axs[3].set_xlim([0, T])

                # Show the plot
                #plt.tight_layout()
                #plt.show()
                #np.save('dendplrnn4neuron/BPTT_TF/plots/%s/crcns_test_trace%s_%s.npy'%(rep,nr,rep),np.concatenate([X.T*(vmax-vmin)+vmin,Z[:,:].T]))

                # Print statistics
                print(f"Model {nr}, repetition {i} performance statistics")
                # print(f"Mean firing rate: {mean_firing_rate} Hz")
                # print(f"Inter-spike intervals: {isi} ms")
                print(f"Number of occurring spike times: {num_spikes}")
                # print(f"spike times: {model_peaks}")
                # print(f"Duration of the experiment: {T}")

                print(f"GT length: {len(all_data_spikes[i])}")
                
                v = len(all_data_spikes[i])/T
                Ncoinc = calculate_coincidences(all_data_spikes[i], model_peaks, 0.004*fs)
                all_model_gammas.append(calculate_gamma(Ncoinc, len(all_data_spikes[i]), len(model_peaks), v, 0.004*fs))
                print(f"Gamma of model {nr}, repetition {i}  = {all_model_gammas[i]}")

    print(all_model_gammas)

print(f"Length of all the model gammas calculated: {len(all_model_gammas)}")
print(f"Length of all the data spikes calculated: {len(all_data_spikes)}")


best_model_performance = (calculate_PA(all_data_spikes, all_model_gammas))

np.save(f'crcns_evaluation/best_model_performance_{epoch_nr}.npy', best_model_performance)


print(f"The model {nr} predicts ST with a performance value of {best_model_performance}")
