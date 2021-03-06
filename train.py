import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import sv_utils
import torch.nn as nn
import os

def train(model, loss, optimizer, dataloader, device, epoch, verbose, log_interval=10):
    model.train()
    total = 0
    correct1 = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        train_loss = loss(output, target)
        train_loss.backward()
        optimizer.step()

        total += train_loss.item() * data.size(0)
        _, pred = output.topk(5, dim=1)
        correct = pred.eq(target.view(-1, 1).expand_as(pred))
        correct1 += correct[:,:1].sum().item()
        if verbose & (batch_idx % log_interval == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \t'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), train_loss.item()))
    if verbose:
        print('Train Epoch: {} \tTrain accuracy: {:.6f} \t'.format(
            epoch, 100. * correct1 / len(dataloader.dataset)))
    return total / len(dataloader.dataset)

def eval(model, loss, dataloader, device, verbose):
    model.eval()
    total = 0
    correct1 = 0
    correct5 = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total += loss(output, target).item() * data.size(0)
            _, pred = output.topk(5, dim=1)
            correct = pred.eq(target.view(-1, 1).expand_as(pred))
            correct1 += correct[:,:1].sum().item()
            correct5 += correct[:,:5].sum().item()
    average_loss = total / len(dataloader.dataset)
    accuracy1 = 100. * correct1 / len(dataloader.dataset)
    accuracy5 = 100. * correct5 / len(dataloader.dataset)
    if verbose:
        print('Evaluation: Average loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%)'.format(
            average_loss, correct1, len(dataloader.dataset), accuracy1))
    return average_loss, accuracy1, accuracy5

def train_eval_loop(model, loss, optimizer, scheduler, train_loader, test_loader, device, epochs, verbose, args, size_hook):

    test_loss, accuracy1, accuracy5 = eval(model, loss, test_loader, device, verbose)
    rows = [[np.nan, test_loss, accuracy1, accuracy5]]
    training_sv = []
    training_sv_avg = []
    training_sv_std = []
    for epoch in tqdm(range(epochs)):
        train_loss = train(model, loss, optimizer, train_loader, device, epoch, verbose)
        test_loss, accuracy1, accuracy5 = eval(model, loss, test_loader, device, verbose)
        row = [train_loss, test_loss, accuracy1, accuracy5]
        scheduler.step()
        rows.append(row)
        if args.compute_sv and epoch % args.save_every == 0:
            sv, sv_avg, sv_std = sv_utils.get_sv(model, size_hook)
            training_sv.append(sv)
            training_sv_avg.append(sv_avg)
            training_sv_std.append(sv_std)
            np.save(os.path.join(args.result_dir, 'sv.npy'), training_sv)
            np.save(os.path.join(args.result_dir, 'sv_avg.npy'), training_sv_avg)
            np.save(os.path.join(args.result_dir, 'sv_std.npy'), training_sv_std)

    columns = ['train_loss', 'test_loss', 'top1_accuracy', 'top5_accuracy']
    return pd.DataFrame(rows, columns=columns)


