

import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from torchvision.utils import save_image

from random import Random
from torch.multiprocessing import Process
from torch.autograd import Variable
import torch.nn.utils.clip_grad as clip_grad
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.utils import make_grid

from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.fid import FrechetInceptionDistance

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

from sep_logsumexp import logsumexp_loss

import copy
import random
import time
import math

import torch.backends.cudnn as cudnn
import partition_dataset as partition_dataset

from unet import BasicUNet

#Set your Cuda devices, recomended: at least 5 GPU's
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5" 

def corrupt(x, amount):
  """Corrupt the input `x` by mixing it with noise according to `amount`"""
  noise = torch.rand_like(x)
  amount = amount.view(-1, 1, 1, 1) # Sort shape so broadcasting works
  return x*(1-amount) + noise*amount 




def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def evaluate(model, iterator, criterion, device):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        
        for (x, y) in iterator:

            x = x.to(device)
            noise_amount = torch.rand(x.shape[0]).to(device) # Pick random noise amounts
            noisy_x = corrupt(x, noise_amount) # Create our noisy x
            y = y.to(device)

            y_pred = model(x)

            loss = criterion(y_pred, x)


            epoch_loss += loss.item()/x.shape[0]
        
    return epoch_loss / len(iterator)

def evaluate_bin(model, iterator, criterion, device):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        
        for (x, y) in iterator:

            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            y_pred_bin = (y_pred>0.5)*torch.ones(y_pred.shape,device=device)
            loss = criterion(y_pred_bin, x)


            epoch_loss += loss.item()/x.shape[0]

        
    return epoch_loss / len(iterator)

def evaluate_energy(model, iterator, device):
    
    epoch_loss = 0
    epoch_acc = 0
    
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

def evaluate_energy_bin(model, iterator, device):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    criterion = nn.MSELoss(reduction='none')
    criterion = criterion.to(device)
    with torch.no_grad():
        
        for (x, y) in iterator:

            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            y_pred_bin = (y_pred>0.5)*torch.ones(y_pred.shape,device=device)
            loss = torch.sum(criterion(y_pred_bin, x),dim=(1,2,3))/torch.sum(criterion(torch.zeros_like(x), x),dim=(1,2,3))
            loss = 1-torch.mean(loss)


            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator) 

def evaluate_ssim(model, iterator, device):
    
    epoch_ssim = 0
    epoch_acc = 0
    ssim = 0
    model.eval()
    
    with torch.no_grad():
        
        for (x, y) in iterator:
            
            x = x.to(device)
            y = y.to(device)
            

            y_pred = model(x)
            
            epoch_ssim += structural_similarity_index_measure(y_pred, x).item()

        
    return epoch_ssim / len(iterator) 



def train(rank, test_iterator, class_dist, args):

    cudnn.benchmark = True
    torch.manual_seed(rank+class_dist-1)
    torch.cuda.manual_seed(rank+class_dist-1) 


    number_of_devices = torch.cuda.device_count() #Choosing the GPU's to use
    device = torch.device("cuda:{}".format((rank)%number_of_devices)) #Assigning GPU's to clients
    homogen = args['homogen']
    classorder = args['classorder']
    cl_worker = args['cl_worker']
    mnist = args['mnist']
    fmnist = args['fmnist']
    lr1_init = args['lr1']
    lr2 = args['lr2']
    lr_sigma_init = args['lr_sigma_init']
 
    if mnist:
        input_dims = 28*28
    else:
        input_dims = 64
    
  
            
    train_iterator, valid_iterator, bsz = partition_dataset.partition_dataset_mnist(homogen,classorder, class_dist, cl_worker, data_ratio, fmnist, tau) #Partitioning dataset according to clients


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
    H = args['H']
    fedavg_init = args['fedavg']
    
    
    # inp and out channels
    inc = 1
    outc = 1
    

    criterion_MSE = nn.MSELoss(reduction='sum')
    criterion_MSE = criterion_MSE.to(device)

    criterion_energy = nn.MSELoss(reduction='none')
    criterion_energy = criterion_energy.to(device)

    criterion_col = logsumexp_loss()
    criterion_col  = criterion_col.to(device)


    model = BasicUNet(inc, outc)

    model.to(device)
    if fedavg:
        broadcast_model(model,0)



    model_p = BasicUNet(inc,outc)
    model_p.to(device)
    
    broadcast_model(model_p,0)


    #Set optimizer
    

    optimizer = optim.Adam(model.parameters(), lr = lr1_init, weight_decay=weight_decay)
    optimizer_p = optim.Adam(model_p.parameters(), lr = lr2)
   
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
        milestones=[75], gamma = 0.1, last_epoch=-1)
    lr_scheduler_p = torch.optim.lr_scheduler.MultiStepLR(optimizer_p,
        milestones=[75], gamma = 0.1, last_epoch=-1)



    best_test_acc = 0
    best_avg_test_loss = 0
    best_pers_test_acc = 0
    best_pers_val_loss = 1000
    
    count = 0 
    tune_mode = False


    #Create empty lists for results

    train_acc_list =[]
    train_loss_list =[]

    test_loss_list =[]
    test_acc_list =[]

    global_test_loss_list =[]
    global_test_acc_list =[]

    train_acc_list_p =[]
    train_loss_list_p =[]

    test_loss_list_p =[]
    test_acc_list_p =[]

    train_acc_list_q =[]
    train_loss_list_q =[]

    ## valditations
    global_val_loss_list =[]
    global_val_acc_list =[]
    
    val_loss_list =[]
    val_ssim_list = []
    val_energy_list = []
    val_acc_list =[]

    val_loss_bin_list =[]
    val_ssim_bin_list = []
    val_energy_bin_list = []
    val_acc_bin_list =[]

    val_acc_list_q =[]
    
    avg_test_acc_list_q = []
    avg_test_acc_list = []
    avg_test_ssim_list = []
    avg_test_energy_list = []

    avg_test_acc_list_q = []
    avg_test_acc_bin_list = []
    avg_test_ssim_bin_list = []
    avg_test_energy_bin_list = []

    avg_train_loss_list = []
    avg_train_loss_p_list = []
    avg_train_loss_list_q = []

    avg_sigma = []

    sync_count = 1
    
    client_part = torch.ones(int(size),device=device) 

    sigma_list = []
    if(lambda_p != 0):
        a = torch.sqrt( torch.tensor( 1/(2*lambda_p), device=device) )
    else:
        a = torch.sqrt( torch.ones( 1, device=device) )
    for p in model.parameters():
        sigma_list.append((torch.zeros_like(p, requires_grad=True, device=device)+a).clone().detach())

    loss_terms_list = []
    grad_list = []

    sigma_update_list = []

    # for lazy update
    lazy_epochs = 2
    window_len = 1
    lazy_loss = 0
    lazy_loss_window = torch.zeros(window_len, device=device)

    #Training Epochs        
    for epoch in range(EPOCHS):

        start_time = time.time()
                

        epoch_loss = 0.0
        epoch_loss_p = 0.0

        batch_count = 0
        model.train()
        model_p.train()

        global_test_loss = 0
        global_test_acc = 0
    
        lr1 = optimizer.param_groups[0]['lr']
        if epoch == tune_epoch and fedavg_tune:
            fedavg = False
            personalized = False

        start_time = time.time()
        for x, y in train_iterator:
            
            with torch.no_grad():
                model_p_old = BasicUNet(inc,outc) 

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
                noise_amount = torch.rand(x.shape[0]).to(device) # Pick random noise amounts
                noisy_x = corrupt(x, noise_amount) # Create our noisy x
                y = y.to(device)

                

            model.zero_grad()

            x_hat = model(noisy_x)
            x_hat_p = model_p(noisy_x)
            
            

            loss_recon = criterion_MSE(x_hat, x)/x.shape[0] #average recon per sample in a batch

            probs = torch.ones(1,device=device)

            
            train_num = len(train_iterator.dataset)
            sample_multiplier = 1/train_num # for scaling the effect of regularizer in a batch to overall dataset
            for sigma in sigma_list:
                sigma.requires_grad_(True)
            loss = criterion_col(model, model_p, sigma_list, probs, True, xi)[0] 
            loss.backward()
            lr_sigma_sq = lr_sigma_init
            
            if(epoch >= lazy_epochs) and (sync_count-1) % H == 0 :
                with torch.no_grad():
                    for sigma in sigma_list:
                        torch.nn.utils.clip_grad_norm_(sigma, max_norm=10, norm_type='inf')
                        sigma -= lr_sigma_sq * sigma.grad.data             
                        sigma.grad.zero_()
                        sigma.clamp_(min=1e-3, max=1e2)
                  

            model.zero_grad()
            model_p.zero_grad()


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
                loss += 1* sample_multiplier  * l #* x.shape[0]
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

            #Update global model for the personalized case
            

            if lambda_p != 0 and not fedavg and client_part[rank] == 1:        
                
                grad_list.append([])

                with torch.no_grad():
                    x_hat = model(x)
                    x_hat_p = model_p(x)
                
                model_p.zero_grad()
                model.zero_grad()
                


                loss_logsumexp = criterion_col(model, model_p, sigma_list, probs, False, xi)
                loss_recon = criterion_MSE(x_hat, x)/x.shape[0]

                loss = loss_recon #+ xi*criterion_KL
                if personalized:
                    loss += loss_logsumexp #lambda_p*loss_logsumexp


                loss.backward()
    
                model.zero_grad()
                if not fedavg :#and (sync_count-1) % H == 0:
                    torch.nn.utils.clip_grad_norm_(model_p.parameters(), max_norm=1, norm_type='inf')
                    optimizer_p.step()
                    
                
                epoch_loss_p += loss.item()   
                

                
                
                
            # For communication
            with torch.no_grad():
                if fedavg:
                    if client_part[rank] == 0 and (sync_count) % H == 0 :
                        model_temp = BasicUNet(inc,outc)  

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
                        model_ptemp = BasicUNet(inc,outc)  

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

            epoch_loss += loss_recon.item()
            
            sync_count += 1
            
            # if mnist and not fmnist:
                
            if batch_count == tau-1:
                break
                
            batch_count += 1


            

        lr_scheduler_p.step()
        lr_scheduler.step()
        
        # sigma stats computation for logging:
        with torch.no_grad():
            ttmp = 0
            d = 0
            M = 0
            m = 100
            for sigma in sigma_list:
                d += torch.numel(sigma)
            for sigma in sigma_list:
                #print(sigma.grad.data)
                ttmp += torch.sum(sigma) / d
                M = max(M, torch.max(sigma).item())
                m = min(m, torch.min(sigma).item())
            #print(epoch, 'sigma', ttmp.item(), M)
        avg_sigma.append([ttmp.item(), M, m])
        
        
        
        model_pcopy = BasicUNet(inc,outc)  



        model_pcopy.to(device)

        if not fedavg:
            model_pcopy.load_state_dict(model_p.state_dict())
        else:
            model_pcopy.load_state_dict(model.state_dict())
            average_model(model_pcopy,1)
            #broadcast_model(model_pcopy, nonz[0])
        
        #compute validation loss
        if not fedavg or fedavg_tune:
            val_loss = evaluate(model, valid_iterator, criterion_MSE, device)
            val_energy = evaluate_energy(model, valid_iterator, device)
            

            trainend_loss = evaluate(model, train_iterator, criterion_MSE, device)
            
        else:
            val_loss = evaluate(model_pcopy, valid_iterator, criterion_MSE, device)
            val_energy =  evaluate_energy(model_pcopy, valid_iterator, device)

            trainend_loss = evaluate(model_pcopy, train_iterator, criterion_MSE, device)



        


        global_test_loss = evaluate(model_pcopy, valid_iterator, criterion_MSE, device)
        global_test_acc = torch.tensor(global_test_acc,device=device)
        average_parameter(global_test_acc)

        train_loss_list.append(trainend_loss)

        

        global_test_loss_list.append(global_test_loss)

        val_loss_list.append(val_loss)
        val_energy_list.append(val_energy)


        
        folder = args['save_location']
        directory = folder + 'class dist ' + str(class_dist) + '/'
        


       
        save_place = (directory + 'b_sep_diff_0.01_simple_lrSigma_' + str(lr_sigma_init) + '_' + str(input_dims)  + 
        '_' + '_lw'+ str(lambda_p_init) + '_'+ 'lr2'+ str(lr2) +  '_' + 'size' + str(size) + '_' + str(H) + 'H_'+ 'genstd_' + str(gen_std) + '_beta' + str(xi)  + '_trainnum' + str(1000)  )
       
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
        


        if global_test_loss > best_avg_test_loss:
            best_avg_test_loss = global_test_loss


        val_energy_avg = torch.tensor(val_energy,device=device)
        average_parameter(val_energy_avg)
        avg_test_energy_list.append(val_energy_avg.cpu().numpy().item())


        val_avg = torch.tensor(val_loss,device=device)
        average_parameter(val_avg)
        avg_test_acc_list.append(val_avg.cpu().numpy().item())


        
        
        
        if val_avg < best_pers_val_loss:
            best_model = BasicUNet(inc,outc)  

            best_model.to(device)
            best_pers_val_loss = val_avg
            if fedavg:
                best_model.load_state_dict(model_pcopy.state_dict())
            else:
                best_model.load_state_dict(model.state_dict())
            print('Best model updated...')

        


        train_avg = torch.tensor(trainend_loss,device=device)
        average_parameter(train_avg)
        avg_train_loss_list.append(train_avg.cpu().numpy().item())

        train_loc_avg = torch.tensor(train_loc, device=device)
        average_parameter(train_loc_avg)
        avg_train_loss_p_list.append(train_loc_avg.cpu().numpy().item())


        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        avg_loss_data = val_avg.item()
        # avg_ssim_data = val_ssim_avg.item()
        avg_energy_data = val_energy_avg.item()



        global_test_acc_list.append(global_test_loss)
        



        d=0
        for p in model.parameters():
            d += torch.numel(p)
        print('d=',d)


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
           
            print(f'\tAverage Test Loss: {avg_loss_data:.3f}', ' ', optimizer.param_groups[0]['lr'])
            print('Avg, min, max sigmas:', avg_sigma[-1])
           
            print(f'\tAverage Test Energy Captured: {avg_energy_data:.4f}')
 
        save_place_test_txt =  save_place + '_test_rank' + str(rank)+ '.txt'
        save_place_trn_txt =  save_place + '_train_rank' + str(rank) + '.txt'
        save_place_trnloss_txt = save_place + '_trainloss_rank' + str(rank) + '.txt'

        save_place_globaltest_txt =  save_place + '_global_rank' + str(rank) + '.txt'

        save_place_trnloss_q_txt =  save_place + '_trainloss_q_rank' + str(rank) + '.txt'
        save_place_test_q_txt =  save_place + '_test_quantized_rank' + str(rank)+ '.txt'
        
        save_place_val_txt =  save_place + '_val_rank' + str(rank)+ '.txt'

        save_place_avg_test_q_txt =  save_place + '_avgtest_quantized.txt'
        save_place_avg_test_txt =  save_place + '_avgtest_loss.txt'
        save_place_avg_test_ssim_txt =  save_place + '_avgtest_ssim.txt'
        save_place_avg_test_energy_txt =  save_place + '_avgtest_energy.txt'

        save_place_avg_test_bin_txt =  save_place + '_avgtest_loss_bin.txt'
        save_place_avg_test_ssim_bin_txt =  save_place + '_avgtest_ssim_bin.txt'
        save_place_avg_test_energy_bin_txt =  save_place + '_avgtest_energy_bin.txt'

        save_place_avg_train_loss_q_txt = save_place + '_avgtrain_loss_quantized.txt'
        save_place_avg_train_loss_txt = save_place + '_avgtrain_loss.txt'

        save_place_avg_train_loss_p_txt = save_place + '_avgtrain_loss_p.txt'
        
        save_place_avg_sigma_txt = save_place + '_avg_sigma.txt'
        save_place_loss_terms_txt = save_place + '_loss_terms.txt'
        save_place_sigma_update_txt = save_place + '_sigma_update.txt'
        save_place_grad_list_txt = save_place + '_grad_list.txt'

        

        with open(save_place_avg_test_txt, 'w') as filehandle:
            json.dump(avg_test_acc_list, filehandle)



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

    save_images(best_model, valid_iterator, save_place, device)


                
        
def save_images(model, iterator, directory, device):

    rank =  dist.get_rank()
    save_dic = directory  + "images/" + str(rank) + "/"
    save_dic_original = directory + "images/" + str(rank) + "/original/"
    if personalized:
        save_dic += "personalized/"
    elif fedavg:
        save_dic += "fedavg/"
    else:
        save_dic += "local/"

    
    if not os.path.exists(save_dic):
        os.makedirs(save_dic)
    
    if not os.path.exists(save_dic_original):
        os.makedirs(save_dic_original)
    torch.manual_seed(rank)
    torch.cuda.manual_seed(rank)
    np.random.seed(rank)

    model.eval()
    no_of_images = 200
    n_steps = 500
    with torch.no_grad():
        k = 0
        for (x, y) in iterator:

            x = x.to(device)


            for i in range(x.shape[0]):

                torch.save(x[i], save_dic_original + "img{}".format(k))

                k += 1

        for i in range(no_of_images):
            if mnist:
                x = torch.rand(1, 1, 28, 28).to(device)
 
            for j in range(n_steps):
                pred = model(x)
                mix_factor = 1/(n_steps - j)
                x = x*(1-mix_factor) + pred*mix_factor

            torch.clamp_(x,0,1)
            torch.save(x, save_dic + "img{}".format(i))

       




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


    #dataset 
    mnist = True #False
    fmnist = False
    if fmnist:
        mnist = True



    test_iterator = []


    K = 1 #ratio of clients that participates in a communication round

    save_location = 'results/Mnist Results/' #Choose save location
        
    
    cl_worker = 1 
    fedavg = False              #True if testing for Federated Averaging
    fedavg_tune = False          # True if fedavg + fine-tuning

    tau = 20            #number of local iterations
    size = 50                   #number of clients
    
    EPOCHS = 1
    tune_epoch = 50
    pers_epoch = 103
    class_dist_arr = [1,2,3]    # different seeds that average is taken over, set [1,2,3] and take the average of the results to replicate our results
                                
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
   
    processes = []              # For pytorch.distributed
    train_num = 10
    gen_std = 0.01
    xi = 1e-5 # 1e-5 for syn
    baseline = False
    if baseline:
        assert personalized

    classorder = [i for i in range(size*cl_worker)]     # auxiliary vrbl for dataset partition
    
    
    data_ratio= 1.0 # 1.0 for 1200 or 0.5 for 600 samples



    
    exec(open('configurator.py').read()) # overrides from command line or config file


    
    args = {'personalized': personalized, 'fedavg': fedavg, 'homogen': homogen, 
            'size': size, 'H': tau ,'EPOCHS': EPOCHS, 'cl_worker' : cl_worker ,
            'classorder': classorder,  'mnist':mnist, 
                'save_location' : save_location, 'save_details': save_details,  'lambda_p': lambda_p, 'lambda_coef': lambda_coef,
                    'K':K, 'fedavg_tune':fedavg_tune, 'lr1':lr1,  'lr2': lr2, 'fmnist': fmnist, 'lr_sigma_init': lr_sigma}

    for ind in range(len(class_dist_arr)): #different partition of data to clients

            class_dist = class_dist_arr[ind]
            print(class_dist)
            for rank in range(size):
                p = Process(target=init_process, args=(rank, test_iterator, class_dist, args, train))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

