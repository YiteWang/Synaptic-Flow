from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn

def prune_loop(model, loss, pruner, dataloader, device, sparsity, 
               schedule, scope, epochs, reinitialize=False, train_mode=False, shuffle=False):
    r"""Applies score mask loop iteratively to a final sparsity level.
    """
    # Set model to train or eval mode
    model.train()
    if not train_mode:
        model.eval()

    # Prune model
    for epoch in tqdm(range(epochs)):
        pruner.score(model, loss, dataloader, device)
        if schedule == 'exponential':
            sparse = sparsity**((epoch + 1) / epochs)
        elif schedule == 'linear':
            sparse = 1.0 - (1.0 - sparsity)*((epoch + 1) / epochs)
        pruner.mask(sparse, scope)
    
    # Reainitialize weights
    if reinitialize:
        model._initialize_weights()

    # Shuffle masks
    if shuffle:
        pruner.shuffle()

    for i in model.modules():
        if isinstance(i, nn.BatchNorm2d):
            i.reset_running_stats()
            
    # Confirm sparsity level
    remaining_params, total_params = pruner.stats()
    # print('remain parameters: {} '.format(remaining_params))
    # print('total parameters: {} '.format(total_params))
    # print('supposed parameters: {} '.format(total_params*sparsity))
    if np.abs(remaining_params - total_params*sparsity) >= 5:
        print("ERROR: {} prunable parameters remaining, expected {}".format(remaining_params, total_params*sparsity))
        quit()
