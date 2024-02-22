mnist = True #False
fmnist = False




test_iterator = []


K = 1 #ratio of clients that participates in a communication round

save_location = 'General Mnist Results/' #Choose save location
    
            #True for personalzied training, False for Local Training


cl_worker = 1 
fedavg = True              #True if testing for Federated Averaging
fedavg_tune = True          # True if fedavg + fine-tuning

tau = 20            #number of local iterations
size = 50                   #number of clients, for cifar-100 set it to 50

EPOCHS = 100
tune_epoch = 50
class_dist_arr = [1]    # different seeds that average is taken over, set [1,2,3] and take the average of the results to replicate our results
                            
save_details = True         
sigma_in = 0.8
lambda_p = 1/2/(sigma_in**2)     

lambda_coef = 1e-6      # 1e-6 mnist 
personalized = False
lr1 = 1e-3 #lr for personalized model
lr2 = 1e-1 #lr for global model
if not personalized:
    lr_sigma = 0.0  
else:
    lr_sigma = 1e-3  # lr for variance
assert not (fedavg and personalized) #make sure fedavg and personalized not TRUE at the same time



homogen = False             # If the dataset is distributed homogenously or not

train_num = 10
gen_std = 0.01
beta = 1e-5 # 1e-5 for syn
baseline = False
if baseline:
    assert personalized

classorder = [i for i in range(size*cl_worker)]     # auxiliary vrbl for dataset partition



data_ratio= 1.0 # whether 1200 or 600 samples

