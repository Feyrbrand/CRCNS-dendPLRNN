# stp is better?
#%%
import matplotlib.pyplot as plt
import matplotlib.image as mim
import numpy as np
import json
import h5py
from bptt.models import Model
import torch as tc
from evaluation.metrics import stp
from scipy.signal import find_peaks

#%% plot train/test data for all CV
measure = 'stp'
mm = np.nanmax

fs = 1/0.00028

for dname in ['ML_Hopf','WB_HOM','WB_SNIC']:
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)

    for i in [0,2,3]:
        dat = np.load("datasets/CV_0_%i/single_neuron_%s/scaled/test_data.npy"%(i,dname))

        '''
        plt.figure(# get fI curve
        freqs = []

        min_curr, max_curr = curr_range
        currents = np.linspace(min_curr, max_curr, res)

        currents_rescaled = currents.copy()

        for i, curr in enumerate(currents):
            )
            plt.plot(dat[:,0])
            plt.xlim([0,1000])
            plt.show()
        '''
        #plt.figure()
        ax1.plot(np.arange(0, len(dat[:,0])/fs, 1/fs),dat[:,0],label='0.%i'%i)
        ax2.plot(np.arange(0, len(dat[:,0])/fs, 1/fs),dat[:,1])
    ax1.legend(title='CV',loc='upper left',bbox_to_anchor=(1,1))
    ax1.set_xlim([0,1])
    ax2.set_xlim([0,1])
    plt.show()


#%% show inference - where to find RNN traces??
# at which epoch? 5000?
# pick best fit? (picked according to noisy mse)


epoch = 5000
run = '001'

for dname in ['ML_Hopf','WB_HOM','WB_SNIC']:
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)

    for i,me in zip([0,2,3],[5000,5000,15000]):

        err = json.load(open("results/CV_0_%i/single_neuron_%s/M26B47tau05T500/results_6_%i_100_noisy.json"%(i,dname,me)))

        epoch = err['epochs'][np.where(np.array(err[measure])==mm(np.array(err[measure])))[0][0]]
        run = np.where(np.array(err[measure])==mm(np.array(err[measure])))[1][0]

        model_path = "results/CV_0_%i/single_neuron_%s/M26B47tau05T500/00%i"%(i,dname,run+1)
        dat = np.load("datasets/CV_0_%i/single_neuron_%s/scaled/test_data.npy"%(i,dname))

        m = Model()
        m.init_from_model_path(model_path, epoch=epoch)
        m.eval()
        inp = dat[:,1]
        T = len(inp)

        X, Z = m.generate_free_trajectory(inp, tc.from_numpy(dat[:,0].reshape(-1,1)), T)
        ax1.plot(np.arange(0, len(X)/fs, 1/fs),X,label='0.%i'%i)
        if i==0:
            ax1.plot(np.arange(0, len(dat[:,0])/fs, 1/fs),dat[:,0],c='k',zorder=-1,linewidth=3)

        #print(stp.spike_time_precision(X, dat[:,0], np.mean(dat[:,0])+np.std(dat[:,0])))
        if 'Hopf' in dname:
            crossing0 = 1.5
        else:
            crossing0=5

        dt = 1/fs

        spikes_pred = (
            find_peaks(np.squeeze(X), height=crossing0, distance=200)[0] * dt
        )

        spikes_ref = find_peaks(dat[:,0], height=crossing0, distance=200)[0] * dt

        ax2.scatter(spikes_pred,np.ones(len(spikes_pred)))
        ax2.scatter(spikes_ref,np.ones(len(spikes_ref)),c='k',zorder=-1)


    ax1.legend(title='CV',loc='upper left',bbox_to_anchor=(1,1))
    ax1.set_xlim([10,11])
    ax2.set_xlim([10,11])
    plt.title(dname)
    plt.show()



#%% plot summary stats
# results_6 vs results_6_noisy

# plot best prc according to this score?

cmap = plt.get_cmap('tab10')
plt.figure()
for j,dname in enumerate(['ML_Hopf','WB_HOM','WB_SNIC']):
    for ii,(i,me) in enumerate(zip([0,2,3],[5000,5000,15000])):
        err = json.load(open("results/CV_0_%i/single_neuron_%s/M26B47tau05T500/results_6_%i_100.json"%(i,dname,me)))
        print(err['mse20'][-1])
        if j==0:
            plt.scatter(j,mm(np.array(err[measure])),c=cmap(ii),label='0.%i'%i)
        else:
            plt.scatter(j,mm(np.array(err[measure])),c=cmap(ii))
        print(np.array(err[measure]).shape)
        print(np.where(np.array(err[measure])==mm(np.array(err[measure]))))
        print(err['epochs'][np.where(np.array(err[measure])==mm(np.array(err[measure])))[0][0]])
plt.xticks([0,1,2])
plt.gca().set_xticklabels(['hopf','hom','snic'])
plt.legend(title='CV')
plt.ylabel('best %s'%measure)
plt.show()

#%%

# results_6 vs results_6_noisy
cmap = plt.get_cmap('tab10')
plt.figure()
for j,dname in enumerate(['ML_Hopf','WB_HOM','WB_SNIC']):
    plt.figure()
    for ii,(i,me) in enumerate(zip([0,2,3],[5000,5000,15000])):
        err1 = json.load(open("results/CV_0_%i/single_neuron_%s/M26B47tau05T500/results_6_%i_100.json"%(i,dname,me)))
        err2 = json.load(open("results/CV_0_%i/single_neuron_%s/M26B47tau05T500/results_6_%i_100_noisy.json"%(i,dname,me)))

        plt.scatter(0,mm(err1[measure]),c=cmap(ii))
        plt.scatter(ii,mm(err2[measure]),c=cmap(ii),label='0.%i'%i)

    plt.xticks([0,1,2])
    plt.gca().set_xticklabels([0,0.2,0.3])
    plt.legend(title='CV')
    plt.ylabel('best %s'%measure)
    plt.show()

#%% evaluated on the correct noise level? so for 0 should be same..
