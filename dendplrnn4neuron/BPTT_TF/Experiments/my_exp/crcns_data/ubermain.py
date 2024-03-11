from multitasking import *


def ubermain(n_runs):
    """
    Specify the argument choices you want to be tested here in list format:
    e.g. args.append(Argument('dim_z', [5, 6], add_to_name_as='z'))
    will test for dimensions 5 and 6 and save experiments under z5 and z6.
    Possible Arguments can be found in main.py

    When using GPU for training (i.e. Argument 'use_gpu 1')  it is generally
    not necessary to specify device ids, tasks will be distributed automatically.
    """
    args = []
    args.append(Argument('experiment', ['neuron_w_input_real']))
    args.append(Argument('data_path', ['Experiments/my_exp/real_data/train_data.npy']))
    args.append(Argument('input_size',[1]))
    args.append(Argument('dim_z', [26], add_to_name_as="M"))
    args.append(Argument('n_bases', [47], add_to_name_as="B"))
    args.append(Argument('fix_obs_model', [1]))
    args.append(Argument('mean_centering', [0]))
    args.append(Argument('learn_z0', [1]))
    args.append(Argument('n_epochs', [100]))
    args.append(Argument('teacher_forcing_interval', [10], add_to_name_as="tau"))
    args.append(Argument('seq_len', [500], add_to_name_as="T"))
    args.append(Argument('latent_model', ['clipped-PLRNN']))
    args.append(Argument('learning_rate', [1e-3]))
    args.append(Argument('gradient_clipping', [10]))
    args.append(Argument('use_gpu', [0]))
    args.append(Argument('run', list(range(1, 1 + n_runs))))
    return args


if __name__ == '__main__':
    import os
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # number of runs for each experiment
    n_runs = 10
    # number of runs to run in parallel
    n_cpu = 10
    # number of processes run parallel on a single GPU
    n_proc_per_gpu = 10

    args = ubermain(n_runs)
    run_settings(*create_tasks_from_arguments(args, n_proc_per_gpu, n_cpu))