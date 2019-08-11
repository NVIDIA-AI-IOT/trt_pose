import argparse
import subprocess
import torch
import torchvision
import os
import torch.optim
import tqdm
import apex.amp as amp
import time
import json
import pprint
import torch.nn.functional as F
from .coco import CocoDataset
from .models import MODELS

OPTIMIZERS = {
    'SGD': torch.optim.SGD,
    'Adam': torch.optim.Adam
}

EPS = 1e-6


def set_lr(optimizer, lr):
    for p in optimizer.param_groups:
        p['lr'] = lr
        
        
def save_checkpoint(model, directory, epoch):
    if not os.path.exists(directory):
        os.mkdir(directory)
    filename = os.path.join(directory, 'epoch_%d.pth' % epoch)
    print('Saving checkpoint to %s' % filename)
    torch.save(model.state_dict(), filename)

    
def write_log_entry(logfile, epoch, train_loss, test_loss):
    with open(logfile, 'a+') as f:
        logline = '%d, %f, %f' % (epoch, train_loss, test_loss)
        print(logline)
        f.write(logline + '\n')
        
device = torch.device('cuda')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    
    print('Loading config %s' % args.config)
    with open(args.config, 'r') as f:
        config = json.load(f)
        pprint.pprint(config)
        
    # LOAD DATASETS
    
    train_dataset_kwargs = config["train_dataset"]
    train_dataset_kwargs['transforms'] = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_dataset_kwargs = config["test_dataset"]
    test_dataset_kwargs['transforms'] = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = CocoDataset(**train_dataset_kwargs)
    test_dataset = CocoDataset(**test_dataset_kwargs)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        **config["train_loader"]
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        **config["test_loader"]
    )
    
    model = MODELS[config['model']['name']](**config['model']['kwargs']).to(device)
    
    optimizer = OPTIMIZERS[config['optimizer']['name']](model.parameters(), **config['optimizer']['kwargs'])
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    
    for epoch in range(config["epochs"]):
        
        if str(epoch) in config['lr_schedule']:
            new_lr = config['lr_schedule'][str(epoch)]
            print('Adjusting learning rate to %f' % new_lr)
            set_lr(optimizer, new_lr)
        
        if epoch % config['checkpoints']['interval'] == 0:
            save_checkpoint(model, config['checkpoints']['directory'], epoch)
            
           
        
        train_loss = 0.0
        model = model.train()
        for image, cmap, paf in tqdm.tqdm(iter(train_loader)):
            image = image.to(device)
            cmap = cmap.to(device)
            paf = paf.to(device)
            
            optimizer.zero_grad()
            cmap_out, paf_out = model(image)
            
            loss = F.mse_loss(cmap_out, cmap) + F.mse_loss(paf_out, paf)
            
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            
            optimizer.step()
            train_loss += float(loss)
            
        train_loss /= len(train_loader)
        
        test_loss = 0.0
        model = model.eval()
        for image, cmap, paf in tqdm.tqdm(iter(test_loader)):
      
            with torch.no_grad():
                image = image.to(device)
                cmap = cmap.to(device)
                paf = paf.to(device)

                cmap_out, paf_out = model(image)

                loss = F.mse_loss(cmap_out, cmap) + F.mse_loss(paf_out, paf)

                test_loss += float(loss)
        test_loss /= len(test_loader)
        
        write_log_entry(config['logfile'], epoch, train_loss, test_loss)