import os
import time
import numpy as np
import json
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.multiprocessing import Process
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.fid import FrechetInceptionDistance

from sep_logsumexp import logsumexp_loss
from model import Autoencoder
from model import Autoencoder_gen
from model import ConvAutoencoder
import partition_dataset as partition_dataset

#Set your Cuda devices, recomended: at least 5 GPU's
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5" 

def get_synthetic_data(input_dims=64*64, latent_dims=0, train_num=1, test_num=1, device=None, seed = 1):
    torch.manual_seed(seed)
    model = Autoencoder(input_dims, latent_dims)
    model.to(device)

    signal_std = 0.1
    gen_model = Autoencoder_gen(input_dims, latent_dims)
    gen_model.to(device)
    with torch.no_grad():
        for param in gen_model.parameters():
            param.copy_(torch.zeros_like(param))
            param.add_(torch.normal(mean=0, std=signal_std , size=param.shape, device=device) )

    broadcast_model(gen_model,0)
    torch.manual_seed(dist.get_rank())

    with torch.no_grad():
        for param in gen_model.parameters():
            param.add_( torch.normal(mean=0, std=gen_std , size=param.shape, device=device) )
    
    train_latent_data = torch.normal(mean=0, std=0.5, size=(train_num, latent_dims), device=device)
    test_latent_data = torch.normal(mean=0, std=0.5, size=(test_num, latent_dims), device=device)
    
    train_data = gen_model.lineardec_b(train_latent_data).clone().detach().float() # +  torch.normal(mean=0, std=0.01, size=(train_num, input_dims), device=device)
    test_data = gen_model.lineardec_b(test_latent_data).clone().detach().float() #+  torch.normal(mean=0, std=0.01, size=(test_num, input_dims), device=device)
    
    train_data = train_data.reshape((-1, 1, int(np.sqrt(input_dims)), int(np.sqrt(input_dims))))
    test_data = test_data.reshape((-1, 1, int(np.sqrt(input_dims)), int(np.sqrt(input_dims))))

    train_data = train_data/torch.std(train_data)
    test_data = test_data/torch.std(test_data)

    train_data = torch.utils.data.TensorDataset(train_data, torch.zeros(train_num))
    valid_data = torch.utils.data.TensorDataset(test_data, torch.zeros(test_num))

    if mnist:
        bsz = np.floor(train_num)//20
    else:
        bsz = np.floor(train_num)
    train_iterator = torch.utils.data.DataLoader(train_data, batch_size=int(bsz), shuffle=True) 
    valid_iterator = torch.utils.data.DataLoader(valid_data, batch_size=int(bsz), shuffle=True)

    return train_iterator, valid_iterator, bsz, gen_model

def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            loss = criterion(y_pred, x)
            epoch_loss += loss.item()/x.shape[0]
        
    return epoch_loss / len(iterator)

def evaluate_energy(model, iterator, device):
    epoch_loss = 0
    model.eval()
    criterion = nn.MSELoss(reduction='none')
    criterion = criterion.to(device)
    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            loss = torch.sum(criterion(y_pred, x),dim=(1,2,3))/torch.sum(criterion(torch.zeros_like(x), x),dim=(1,2,3))
            loss = 1-torch.mean(loss)
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate_ssim(model, iterator, device):
    epoch_ssim = 0
    model.eval()
    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            epoch_ssim += structural_similarity_index_measure(y_pred, x).item()

    return epoch_ssim / len(iterator)

def train(rank, test_iterator, class_dist, args):
    seed = args['seed']
    cudnn.benchmark = True
    torch.manual_seed(rank + seed)

    number_of_devices = torch.cuda.device_count()       #Choosing the GPU's to use
    device = torch.device("cuda:{}".format((rank)%number_of_devices))       #Assigning GPU's to clients
    homogen = args['homogen']
    classorder = args['classorder']
    cl_worker = args['cl_worker']
    mnist = args['mnist']
    fmnist = args['fmnist']
    lr1_init = args['lr1']
    lr2 = args['lr2']
    latent_dims = args['latent_dims']
    lr_sigma_init = args['lr_sigma_init']
    H = args['H']
    seed = args['seed']
    if mnist:
        input_dims = 28*28
    else:
        input_dims = 64
    syn = args['syn']
    if mnist:
        syn = False
    test_num = 20
    if syn:
        train_iterator, valid_iterator, bsz, gen_model = get_synthetic_data(input_dims, gen_latent_dims, train_num, test_num, device, seed)
    else:
        if not mnist:
            train_iterator, valid_iterator, bsz = partition_dataset.partition_dataset(homogen,classorder, class_dist, cl_worker, cifar100, data_ratio, tau= H, seed= seed) #Partitioning dataset according to clients
        else:
            train_iterator, valid_iterator, bsz = partition_dataset.partition_dataset_mnist(homogen,classorder, class_dist, cl_worker, data_ratio, fmnist, tau=H, seed= seed) #Partitioning dataset according to clients
    
    # HYPERPARAMETERS
    weight_decay = 0
    K = args['K']
    personalized = args['personalized']
    if personalized:
            lambda_p_init = args['lambda_p']
    else:
        lambda_p_init = 0
        print('Local Training')
    lambda_p = lambda_p_init

    EPOCHS = args['EPOCHS']
    fedavg = args['fedavg']
    fedavg_init = args['fedavg']

    criterion_MSE = nn.MSELoss(reduction='sum')
    criterion_MSE = criterion_MSE.to(device)

    criterion_col = logsumexp_loss()
    criterion_col  = criterion_col.to(device)

    # initialize the local models
    model = Autoencoder(input_dims, latent_dims)
    if conv:
        model = ConvAutoencoder(input_dims, latent_dims)
    model.to(device)
    if fedavg:
        broadcast_model(model,0)
    if baseline:
        model.lineardec_b.weight.data.copy_(gen_model.lineardec_b.weight.data)
        model.lineardec_b.bias.data.copy_(gen_model.lineardec_b.bias.data)

    # initialize the global model
    model_p = Autoencoder(input_dims, latent_dims)
    if conv:
        model_p = ConvAutoencoder(input_dims, latent_dims)
    model_p.to(device)
    broadcast_model(model_p,0)

    #Set optimizer
    optimizer = optim.SGD(model.parameters(), lr = lr1_init, momentum = 0.9)
    optimizer_p = optim.SGD(model_p.parameters(), lr = lr2) 

    # Create empty lists for results
    avg_test_acc_list = []
    avg_test_ssim_list = []
    avg_test_energy_list = []
    avg_train_loss_list = []
    avg_train_loss_p_list = []
    avg_sigma = []
    loss_terms_list = []
    grad_list = []
    sigma_update_list = []

    sync_count = 1      # controlling the coccurance of ommunication rounds
    
    client_part = torch.ones(int(size),device=device) 

    sigma_list = []
    if(lambda_p != 0):
        a = torch.sqrt( torch.tensor( 1/(2*lambda_p), device=device) )
    else:
        a = torch.sqrt( torch.ones( 1, device=device) )
    for p in model.parameters():
        sigma_list.append((torch.zeros_like(p, requires_grad=True, device=device)+a).clone().detach())
        
    # for setting the lazy epochs
    window_len = 1
    improve_rate = 100
    lazy_loss = 0
    sigma_update = False
    lazy_loss_window = torch.zeros(window_len, device=device)

    #Training Epochs        
    for epoch in range(EPOCHS):
        start_time = time.time()

        batch_count = 0
        model.train()
        model_p.train()
    
        if epoch == tune_epoch and fedavg_tune:
            fedavg = False
            personalized = False

        for x, y in train_iterator:
            with torch.no_grad():
                model_p_old = Autoencoder(input_dims,latent_dims)
                if conv:
                    model_p_old = ConvAutoencoder(input_dims, latent_dims)
                model_p_old.to(device)
                nonz = torch.nonzero(client_part)

                if not fedavg:
                    model_p_old.load_state_dict(model_p.state_dict())
                else:
                    model_p_old.load_state_dict(model.state_dict())
                if (sync_count-1)%H==0:
                    if not fedavg:
                        broadcast_model(model_p, nonz[0])
                    else:
                        broadcast_model(model, nonz[0])
                    client_part = torch.cat((torch.ones(int(K*size),device=device),torch.zeros(int(size-K*size),device=device)))     
                    order = np.array(range(len(client_part)))
                    np.random.shuffle(order)
                    client_part[np.array(range(len(client_part)))] = client_part[order]
                    dist.broadcast(client_part,0)

                    if client_part[rank] == 0:
                        if not fedavg:
                            model_p.load_state_dict(model_p_old.state_dict())
                        else:
                            model.load_state_dict(model_p_old.state_dict())
                del model_p_old

                x = x.to(device)
                y = y.to(device)

            model.zero_grad()

            x_hat = model(x)
            sample_loc_var = 0.5
            loss_recon = criterion_MSE(x_hat, x)/x.shape[0] #average recon per sample in a batch
            
            probs = torch.ones(1,device=device)           
            sample_multiplier = 1/train_num # for scaling the effect of regularizer in a batch to overall dataset
            for sigma in sigma_list:
                sigma.requires_grad_(True)
            loss = criterion_col(model, model_p, sigma_list, probs, True, xi)[0] 
            loss.backward()
            if(sigma_update):
                lr_sigma_sq = lr_sigma_init#*0.99**(epoch)
            else:
                lr_sigma_sq = 0
            
            if(epoch >= lazy_epochs) and (sync_count-1) % H == 0 :
                with torch.no_grad():
                    for sigma in sigma_list:
                        torch.nn.utils.clip_grad_norm_(sigma, max_norm=10, norm_type='inf')
                        sigma -= lr_sigma_sq * sigma.grad.data             
                        sigma.grad.zero_()
                        sigma.clamp_(min=1e-3, max=1e2)
                   
            model.zero_grad()
            model_p.zero_grad()

            criterion_KL = model.kl
            criterion_KL = criterion_KL.to(device)
            loss = loss_recon 
            loss_terms = [loss_recon.item()]

            sigma_list_copy = sigma_list.copy()
            sigma_max = torch.max(sigma_list_copy[0])
            for i in range(len(sigma_list_copy)):
                if i%2==0:
                    sigma_max = torch.max(sigma_list_copy[i])
                torch.max(sigma_max, torch.max(sigma_list_copy[i]))

                sigma_list_copy[i].requires_grad_(True)
                sigma_list_copy[i] = sigma_max*torch.ones_like(sigma_list_copy[i])

            if personalized:
                l, a, b = criterion_col(model, model_p, sigma_list, probs, True, xi)
                loss += sample_multiplier*(2*sample_loc_var)  * l 
                loss_terms.append(a)
                loss_terms.append(b)
            loss_terms_list.append(loss_terms)
                
            if client_part[rank] == 1:
                loss.backward()
                train_loc = loss.item()
                if client_part[rank] == 0:
                    model.zero_grad()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type='inf')
                optimizer.step()

            if lambda_p != 0 and not fedavg and client_part[rank] == 1:                       
                grad_list.append([])
                with torch.no_grad():
                    x_hat = model(x)
                
                model_p.zero_grad()
                model.zero_grad()

                loss_logsumexp = criterion_col(model, model_p, sigma_list, probs, False, xi)
                loss_recon = criterion_MSE(x_hat, x)/x.shape[0]
                criterion_KL = model.kl
                criterion_KL = criterion_KL.to(device)
                loss = loss_recon #+ xi*criterion_KL
                if personalized:
                    loss += loss_logsumexp #lambda_p*loss_logsumexp
                loss.backward()
                model.zero_grad()
                if not fedavg:
                    torch.nn.utils.clip_grad_norm_(model_p.parameters(), max_norm=1, norm_type='inf')
                    optimizer_p.step()
      
            # For communication
            with torch.no_grad():
                if fedavg:
                    if client_part[rank] == 0 and (sync_count) % H == 0 :
                        model_temp = Autoencoder(input_dims, latent_dims)
                        if conv:
                            model_temp = ConvAutoencoder(input_dims, latent_dims)
                        model_temp.load_state_dict(model.state_dict())
                        for n,p in model.state_dict().items():
                            p.data.copy_(torch.zeros_like(p))

                    if (sync_count) % H == 0 :
                        average_model(model,K)

                        for sigma in sigma_list:
                            average_parameter(sigma)
                    
                    if client_part[rank] == 0 and (sync_count) % H == 0 :
                        model.load_state_dict(model_temp.state_dict())    
                        del model_temp
                else:
                    if client_part[rank] == 0:
                        model_ptemp = Autoencoder(input_dims, latent_dims)
                        if conv:
                            model_ptemp = ConvAutoencoder(input_dims, latent_dims)
                        model_ptemp.load_state_dict(model_p.state_dict())

                        for n,p in model_p.state_dict().items():
                            p.data.copy_(torch.zeros_like(p))

                    if lambda_p != 0:
                        if (sync_count) % H == 0 :
                            average_model(model_p,K)

                            for sigma in sigma_list:
                                average_parameter(sigma)
                    
                    if client_part[rank] == 0:
                        model_p.load_state_dict(model_ptemp.state_dict())                    
                        del model_ptemp
            sync_count += 1
            if mnist:
                if batch_count == 19: #prevent asyn in updates for mnist
                    break
            batch_count += 1
        
        # record the sigma's
        with torch.no_grad():
            ttmp = 0
            d = 0
            M = 0
            m = 100
            for sigma in sigma_list:
                d += torch.numel(sigma)
            for sigma in sigma_list:
                ttmp += torch.sum(sigma) / d
                M = max(M, torch.max(sigma).item())
                m = min(m, torch.min(sigma).item())
        avg_sigma.append([ttmp.item(), M, m])
    
        model_pcopy = Autoencoder(input_dims, latent_dims)
        if conv:
            model_pcopy = ConvAutoencoder(input_dims, latent_dims)

        model_pcopy.to(device)

        if not fedavg:
            model_pcopy.load_state_dict(model_p.state_dict())
        else:
            model_pcopy.load_state_dict(model.state_dict())
            average_model(model_pcopy,1)
        
        #compute validation loss
        if not fedavg or fedavg_tune:
            val_loss = evaluate(model, valid_iterator, criterion_MSE, device)
            val_ssim = evaluate_ssim(model, valid_iterator, device)
            val_energy = evaluate_energy(model, valid_iterator, device)
            trainend_loss = evaluate(model, train_iterator, criterion_MSE, device)
        else:
            val_loss = evaluate(model_pcopy, valid_iterator, criterion_MSE, device)
            val_ssim = evaluate_ssim(model_pcopy, valid_iterator, device)
            val_energy =  evaluate_energy(model_pcopy, valid_iterator, device)
            trainend_loss = evaluate(model_pcopy, train_iterator, criterion_MSE, device)

        # set the save place
        folder = args['save_location']
        directory = folder + 'class dist ' + str(class_dist) + '/'
        save_place = (directory + 'b_sep_ae_0.01_simple_lrSigma_' + str(lr_sigma_init) + '_' + str(input_dims) + "_" + str(latent_dims,) + 
        '_' + 'genlat' + str(gen_latent_dims) + '_lw'+ str(lambda_p_init) + '_'+ 'lr2'+ str(lr2) +  '_' + 'size' + str(size) + '_' + str(H) + 'H_'+ 'genstd_' + str(gen_std) + '_xi' + str(xi)  + '_trainnum' + str(train_num)  )
        if conv:
            save_place += "conv_"
        if K != 1:
            save_place = save_place + str(K) + 'K_'
        if fedavg_init:
            save_place = save_place + 'fedavg_'
        if fedavg_tune:
            save_place = save_place + 'tune_'
        if homogen:
            save_place = save_place + 'hom'
        else:
            save_place = save_place + 'het' + str(cl_worker) + '_'
        if weight_decay == 0:
            save_place = save_place + 'nowd_'
        save_place = save_place + 'data_' + str(data_ratio) + '_'
        save_place = save_place + 'lr1_' + str(lr1_init)  + '_'
        if baseline:
            save_place = save_place + 'baseline'
        save_place = save_place +'/'

        Path(save_place).mkdir(parents=True, exist_ok=True)

        #averaging the results
        val_ssim_avg = torch.tensor(val_ssim,device=device)
        average_parameter(val_ssim_avg)
        avg_test_ssim_list.append(val_ssim_avg.cpu().numpy().item())

        val_energy_avg = torch.tensor(val_energy,device=device)
        average_parameter(val_energy_avg)
        avg_test_energy_list.append(val_energy_avg.cpu().numpy().item())

        val_avg = torch.tensor(val_loss,device=device)
        average_parameter(val_avg)
        avg_test_acc_list.append(val_avg.cpu().numpy().item())

        train_avg = torch.tensor(trainend_loss,device=device)
        average_parameter(train_avg)
        avg_train_loss_list.append(train_avg.cpu().numpy().item())

        train_loc_avg = torch.tensor(train_loc, device=device)
        average_parameter(train_loc_avg)
        avg_train_loss_p_list.append(train_loc_avg.cpu().numpy().item())

        ### for lazy updates
        lazy_diff = lazy_loss_window[epoch%window_len].item() - train_avg.item()
        lazy_loss_window[epoch%window_len] = train_avg
        if(epoch >= window_len and lazy_diff < improve_rate * lazy_loss):
            sigma_update = True
            sigma_update_list.append(epoch)
        else:
            sigma_update = False
        lazy_loss -= lazy_diff

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        avg_loss_data = val_avg.item()
        avg_ssim_data = val_ssim_avg.item()
        avg_energy_data = val_energy_avg.item()

        if(True):    
            if fedavg:
                print('FedAvg')
            elif not personalized:
                print('Local SGD')
            print('Rank ',dist.get_rank())
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tFP Personalized Train Loss: {trainend_loss:.3f}')
            print(f'\tFP Personalized Train Loss Overall objective : {train_loc:.3f}')
            print(f'\tFP Personalized Validation Loss: {val_loss:.3f}%')
            print(f'\tFP Personalized SSIM: {val_ssim:.3f}%')
            print(f'\tAverage Test Loss: {avg_loss_data:.3f}', ' ', optimizer.param_groups[0]['lr'])
            print('Avg, min, max sigmas:', avg_sigma[-1])
            print(f'\tAverage Test SSIM: {avg_ssim_data:.4f}', ' ', optimizer.param_groups[0]['lr'])
            print(f'\tAverage Test Energy Captured: {avg_energy_data:.4f}')

        save_place_avg_test_txt =  save_place + '_avgtest_loss.txt'
        save_place_avg_test_ssim_txt =  save_place + '_avgtest_ssim.txt'
        save_place_avg_test_energy_txt =  save_place + '_avgtest_energy.txt'
        save_place_avg_train_loss_txt = save_place + '_avgtrain_loss.txt'
        save_place_avg_train_loss_p_txt = save_place + '_avgtrain_loss_p.txt'
        save_place_avg_sigma_txt = save_place + '_avg_sigma.txt'
        save_place_loss_terms_txt = save_place + '_loss_terms.txt'
        save_place_sigma_update_txt = save_place + '_sigma_update.txt'
        save_place_grad_list_txt = save_place + '_grad_list.txt'

        with open(save_place_avg_test_txt, 'w') as filehandle:
            json.dump(avg_test_acc_list, filehandle)
        with open(save_place_avg_test_ssim_txt, 'w') as filehandle:
           json.dump(avg_test_ssim_list, filehandle)
        with open(save_place_avg_test_energy_txt, 'w') as filehandle:
           json.dump(avg_test_energy_list, filehandle)
        with open(save_place_avg_train_loss_txt, 'w') as filehandle:
            json.dump(avg_train_loss_list, filehandle)
        with open(save_place_avg_train_loss_p_txt, 'w') as filehandle:
            json.dump(avg_train_loss_p_list, filehandle)
        with open(save_place_avg_sigma_txt, 'w') as filehandle:
            json.dump(avg_sigma, filehandle)
        with open(save_place_loss_terms_txt, 'w') as filehandle:
            json.dump(loss_terms_list, filehandle)
        with open(save_place_sigma_update_txt, 'w') as filehandle:
            json.dump(sigma_update_list, filehandle)       
        with open(save_place_grad_list_txt, 'w') as filehandle:
            json.dump(grad_list, filehandle)


def epoch_time(start_time, end_time): 
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def broadcast_model(model, src):
    for n,p in model.state_dict().items():
        dist.broadcast(p.data, src)

def average_model(model,K):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for n,p in model.state_dict().items():
        dist.all_reduce(p.data, op=dist.ReduceOp.SUM)
        p.data /= int(size*K)

def avg_of_sq(avg_of_sq_list,K):
    size = float(dist.get_world_size())
    for p in avg_of_sq_list:
        p.copy_(p**2)
        dist.all_reduce(p, op=dist.ReduceOp.SUM)
        p /= int(size*K)

def sq_of_avg(sq_of_avg_list,K):
    size = float(dist.get_world_size())
    for p in sq_of_avg_list:
        dist.all_reduce(p, op=dist.ReduceOp.SUM)
        p /= int(size*K)
        p.copy_(p**2)
        
def average_parameter(parameter):
    """ Accuracy averaging. """
    size = float(dist.get_world_size())
    dist.all_reduce(parameter, op=dist.ReduceOp.SUM)
    parameter /= int(size)

def average_parameter_max(parameter):
    """ Accuracy averaging. """
    size = float(dist.get_world_size())
    param_max = torch.max(parameter)
    parameter = param_max * torch.ones_like(param_max)
    dist.all_reduce(parameter, op=dist.ReduceOp.SUM)
    parameter /= int(size)
    
def init_process(rank, test_iterator, class_dist, args, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, test_iterator, class_dist, args)

if __name__ == "__main__":
    # set the dataset 
    mnist = True 
    fmnist = False
    cifar100 = False
    syn = False

    test_iterator = []

    K = 1       #ratio of clients that participates in a communication round
    # set the save path

    if cifar100:
        save_location = 'results/CIFAR Results/' #Choose save location
    elif mnist:
        if fmnist:
            save_location = 'results/Fashion Mnist Results/'
        else:
            save_location = 'results/Mnist Results/' #Choose save location
    else:
        save_location = 'results/Synthetic Results/'

    
    cl_worker = 1               
    fedavg = False              # True if testing for Federated Averaging
    fedavg_tune = False

    tau = 20                    # number of local iterations
    size = 50                   # number of clients, for cifar-100 set it to 50
    if mnist or syn:
        EPOCHS = 2            # number of epochs
        lazy_epochs = 2         # only update sigma after lazy epochs
        tune_epoch = 75
    else:
        EPOCHS = 250
        lazy_epochs = 200
        tune_epoch = 100
        
    class_dist_arr = [1]        # different class distributions that average is taken over, set [1,2,3] and take the average of the results to replicate our results
                                # there are three randomly generated class distributions over the workers
        
    sigma_in = 0.2
    lambda_p = 1/2/(sigma_in**2)     # lambda = 1/2/(sigma**2)
                                    
    personalized = True
    lr1 = 1e-2           # learning rate for the local models
    lr2 = 1e-2           # learning rate for the global model
    if not personalized:
        lr_sigma = 0.0   # learning rate for sigma
    else:
        lr_sigma = 1e-3  # learning rate for sigma
    assert not (fedavg and personalized)
    latent_dims = 10     # run for different latex dimensions
    xi = 1e-5                   # hyper prior for sigma
    
    baseline = False
    if baseline:
        assert personalized
    if cifar100:
        conv = True             # type of the autoencoders, true for convolution models
    else:
        conv = False

    # for dataset
    homogen = False             # If the dataset is distributed homogenously or not
    processes = []              # For pytorch.distributed
    gen_std = 0.01       # std for synthetic data generation
    gen_latent_dims = 5         # latent dimension for synthetic data generation
    train_num = 10              # number of training sample for synthetic dataset
    classorder = [i for i in range(size*cl_worker)]     # auxiliary vrbl for dataset partition
    data_ratio = 0.25 # 0.1 for mnist fashion mnist, 0.25 for cifar

    ### config override
    exec(open('configurator.py').read()) # overrides from command line or config file


    for ind in range(len(class_dist_arr)): #different sampling of data to clients

        class_dist = class_dist_arr[ind]
        seed = class_dist-1
        args = {'personalized': personalized, 'fedavg': fedavg, 'homogen': homogen,  
                'size': size, 'H': tau ,'EPOCHS': EPOCHS, 'cl_worker' : cl_worker,
                'classorder': classorder, 'mnist':mnist,
                    'save_location' : save_location, 'lambda_p': lambda_p,
                    'K':K, 'fedavg_tune':fedavg_tune, 'lr1':lr1, 'lr2': lr2, 'latent_dims': latent_dims, 'fmnist': fmnist, 'lr_sigma_init': lr_sigma,
                    'syn':syn, 'seed': seed}

        print(class_dist)
        for rank in range(size):
            p = Process(target=init_process, args=(rank, test_iterator, class_dist, args, train))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

