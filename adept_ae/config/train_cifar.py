mnist = False
fmnist = False
cifar100 = True
syn = False

baseline = False

cl_worker = 1               
fedavg = False              # True if testing for Federated Averaging
fedavg_tune = False
personalized = False

tau = 20                    # number of local iterations
size = 10                   # number of clients, for cifar-100 set it to 50
EPOCHS = 250            # number of epochs
lazy_epochs = 200         # only update sigma after lazy epochs
tune_epoch = 75

    
class_dist_arr = [3]        # different class distributions that average is taken over, set [1,2,3] and take the average of the results to replicate our results
                            # there are three randomly generated class distributions over the workers
    
sigma_in = 0.2
lambda_p = 1/2/(sigma_in**2)     # lambda = 1/2/(sigma**2)
                                
lr1 = 1e-2           # learning rate for the local models
lr2 = 1e-2           # learning rate for the global model

lr_sigma = 0.0  # learning rate for sigma
assert not (fedavg and personalized)
latent_dims = 50     # run for different latex dimensions
xi = 1e-6                   # hyper prior for sigma


conv = True # cifar100 true, o/w false

# for dataset
homogen = False             # If the dataset is distributed homogenously or not
gen_std = 0.01       # std for synthetic data generation
gen_latent_dims = 5         # latent dimension for synthetic data generation
train_num = 10              # number of training sample for synthetic dataset
classorder = [i for i in range(size*cl_worker)]     # auxiliary vrbl for dataset partition
data_ratio = 1.0 # 0.1 for mnist fashion mnist, 0.25 for cifar
save_location = 'results/CIFAR Results/'