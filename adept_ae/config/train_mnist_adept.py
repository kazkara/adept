mnist = True 
fmnist = False
cifar100 = False
syn = False


cl_worker = 1               
fedavg = False              # True if testing for Federated Averaging
fedavg_tune = False
personalized = False
tau = 20                    # number of local iterations
size = 50                   # number of clients, for cifar-100 set it to 50
EPOCHS = 150
lazy_epochs = 2
class_dist_arr = [1]        # different class distributions that average is taken over, set [1,2,3] and take the average of the results to replicate our results
                            # there are three randomly generated class distributions over the workers
    
sigma_in = 1.0
lambda_p = 1/2/(sigma_in**2)     # lambda = 1/2/(sigma**2)
                                

lr1 = 1e-2           # learning rate for the local models
lr2 = 1e-2           # learning rate for the global model

lr_sigma = 1e-3  # learning rate for sigma
assert not (fedavg and personalized)
latent_dims = 10     # run for different latex dimensions
xi = 1e-6                   # hyper prior for sigma

baseline = False
if baseline:
    assert personalized or fedavg

conv = False

# for dataset
homogen = False             # If the dataset is distributed homogenously or not
gen_std = 0.01       # std for synthetic data generation
gen_latent_dims = 5         # latent dimension for synthetic data generation
train_num = 10              # number of training sample for synthetic dataset
classorder = [i for i in range(size*cl_worker)]     # auxiliary vrbl for dataset partition
data_ratio = 0.1 # 0.1 for mnist fashion mnist, 0.25 for cifar
save_location = 'General Mnist Results/'