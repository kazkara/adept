If you would like to use our work in your research please cite as:

@misc{ozkara2024hierarchical,
      title={Hierarchical Bayes Approach to Personalized Federated Unsupervised Learning}, 
      author={Kaan Ozkara and Bruce Huang and Ruida Zhou and Suhas Diggavi},
      year={2024},
      eprint={2402.12537},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}


=======================================================================



Dependencies
------------
Python 			  3.8
cudatoolkit               10.1.243
numpy                     1.18.5
pytorch                   1.5.1
scipy                     1.5.0
torchvision               0.6.1


Directory structure
-------------------
The codes folder consists of the following files/folder:

- /adept_diffusion/

    - main_diffusion.py: main file to run diffusion model experiments

    - unet.py: contains unet model to be used

    - partition_dataset.py: partition datasets to introduce heterogeneity, so that each client has samples from one class

    - evaluate_fid.py: contains code to evaluate FID and KID metrics over the saved datasets

    - sep_logsumexp.py: includes the regularization part of the loss function to differentiate over global model and variance

- /adept_ae/

    - main.py: main file to run AE experiments

    - model.py: contains AE models to be used

    - partition_dataset.py: partition datasets to introduce heterogeneity, so that each client has samples from one class

    - sep_logsumexp.py: includes the regularization part of the loss function to differentiate over global model and variance


Running code
------------

To run our code for each case, we need to manually provide the paramaters in config files under config/ folder:

- main_diffusion.py 
    Parameters are set in config files under config/. 
    - personalized: If this parameter set TRUE, then personalized algortihm runs; if set FALSE clients will not collaborate (Local Training)
    - fedavg, fedavg_tune: Boolean; If these parameters are set TRUE, then Federated Averaging + fine-tuning method runs.
    - if both personalized and fedavg are FALSE, local training runs.
    - data_ratio: determines the subsampling amount of data, e.g. let N be the total number of samples, N/m*data_ratio gives the number of samples per client
    

    Output: outputs a folder (can be chosen through save_location parameter) that contains ~200 generated images and 200 true test images s.t. evaluate_fid.py
    can be used to determine metric values. On the same folder it also outputs training and test losses.

    example use in adept_diffusion: python main_diffusion.py config/train_mnist_dgm.py 
    
- main.py
    Parameters are set in config files under config/. 
    - personalized: If this parameter set TRUE, then personalized algortihm runs; if set FALSE clients will not collaborate (Local Training)
    - fedavg: Boolean; If this parameters is set TRUE, then Federated Averaging method runs.
    - if both personalized and fedavg are FALSE, local training runs.
    - syn:  If this is TRUE and other datasets are FALSE, then the experiment will run for Synthetic dataset.
    - mnist: If this is TRUE and other datasets are FALSE, then the experiment will run for MNIST.
    - fmnist: If this is TRUE and the experiment will run for Fashion MNIST.
    - cifar100: If this is TRUE and others are FALSE the experiment will run for CIFAR-10/100.
    - latent_dims_list: determines the latent dimensionality
    - data_ratio: determines the subsampling amount of data, e.g. let N be the total number of samples, N/m*data_ratio gives the number of samples per client

    Output: outputs a folder (can be chosen through save_location parameter) that contains training and test losses, and captured energy per iteration.

    example use in adept_ae: python main.py config/train_fmnist.py 

		
* Occasionally, during the first run when data is not yet downloaded there occurs to be a bug in pytorch.distributed causing Broken pipe, running the script one more time fixes it.

* Please see the comments in the code for detailed information about hyperparameters. 

* We recommend using at least 6 GPUs, the list of GPUs can be modified at the beginning of the code; currently it is adjusted for 6GPUs. To repeat our results, 3 runs for different seeds should be done (class_dist = 1,2,3 refers to this).

