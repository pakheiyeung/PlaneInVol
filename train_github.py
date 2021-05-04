import torch
from torch.utils import data
from torch import nn
import torch.optim as optim
import glob
import os
import numpy as np
from random import shuffle
from tensorboardX import SummaryWriter
import platform
import random



from model import *
from dataset import Dataset
from dataset import Dataset as Dataset_prop



if __name__ ==  '__main__':
    torch.manual_seed(10)
    torch.cuda.manual_seed(10)
    np.random.seed(10)
    random.seed(10)
    
    'Import files names'
    ### files_list should contain the the list of diretories of the .mat files, i.e. [/path/to/mat1, /path/to/mat2, ...]
    ### For each .mat file, it should contain the 'img_brain' that stores the HxWxD image volume 
    ### and 'img_brain_mask' that stores the HxWxD skull binary mask of the volume.
    ### If your file format is different, you may have to modify the corresponding part in the dataset.py
    
    files_list = []
    save_path = ''
    #################################
    ### input your files_list and change the save_path here
    #################################
    def takeSecond(elem):
        pos = elem.find('d')
        return elem[pos-8:pos-4]
            
    files_training = []
    files_validation = []
    for mat_path in files_list:
        files = sorted(glob.glob(mat_path), key=takeSecond)
        num_files = len(files)
        files_training += files[0:int(0.7*num_files)]
        files_validation += files[int(0.7*num_files):int(0.8*num_files)]
    
    'Path and files for saving the models'
    saved_model = os.path.join(save_path, 'best_model.pth')
    current_model = os.path.join(save_path, 'current_model.pth')
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    'CUDA for PyTorch'
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
#    torch.backends.cudnn.benchmark = False

    'Parameters setting'
    batch_size = 60
    params_dataset =  {'batch_size': batch_size,
                      'sample_num':50,
                      'dist_num':15,
                      'mode':'training'}
    params = {'batch_size': int(batch_size/batch_size),
              'shuffle': True,
              'num_workers': 0,
              'drop_last':True}     # for data generator
    num_classes = [3,3,3]   # xyz for 3 points
    learning_rate = 0.0001
    momentum = 0.9
    loss_weight = 10.0      # more wieght for the center point than the two corners
    max_epochs = 200



    'Initialize the model'
    model = Proposed_vgg(make_layers_instance_norm(), num_classes=num_classes, fc_size = 512, device=device).to(device)
    model.apply(weight_init)


    'Loss and optimizer'
    patience = 3
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=patience, verbose=True)
    

    'Log'
    writer = SummaryWriter(os.path.join(save_path, 'log'))

    'Early stop when loss plateu'
    stop_count = 0
    def stop_early(best, current, count):
        stop = False
        if current<=best:
            count=0
        else:
            count+=1
            if count>=2*patience+1:
                stop=True

        return stop, count

    running_loss_feed = 0.0
    running_loss_y1 = 0.0
    running_loss_y2 = 0.0
    running_loss_y3 = 0.0
    running_var = 0.0
    total_loss_feed = 0.0
    total_loss_y1 = 0.0
    total_loss_y2 = 0.0
    total_loss_y3 = 0.0
    best_loss_on_test = np.Infinity
    loss_on_test = {}
    count = 0
    total_count = 0
    
    for epoch in range(max_epochs):
#        if epoch<=10:
#            continue

        'Training'
        if epoch>0:
            shuffle(files_training)
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
        print(save_path)
        
        model = model.train()
        interval = 10   # number of volumes for each sub epoch
        for sub in range(0, len(files_training), interval):
            sub_list = files_training[sub:sub+interval]                
            training_set = Dataset_prop(sub_list, **params_dataset)
            training_set.shuffle_list()
            training_generator = data.DataLoader(training_set, **params)
            i=0
            for (local_batch, local_y1, local_y2, local_y3) in training_generator:
                # Transfer to GPU
                local_batch = local_batch.to(device=device, dtype=torch.float)
                local_y1 = torch.squeeze(local_y1).to(device=device, dtype=torch.float)
                local_y2 = torch.squeeze(local_y2).to(device=device, dtype=torch.float)
                local_y3 = torch.squeeze(local_y3).to(device=device, dtype=torch.float)
                
                if epoch==0 and sub==0 and i==0:
                    print(local_batch.size(), local_y2.size())

                # Model computations
                # zero the parameter gradients
                optimizer.zero_grad()
            
                # forward + backward + optimize
                y1_pred, y2_pred, y3_pred, _ = model(local_batch)
                                
                y1_pred = torch.squeeze(y1_pred)
                y2_pred = torch.squeeze(y2_pred)
                y3_pred = torch.squeeze(y3_pred)
                

                loss_1 = criterion(y1_pred, local_y1)
                loss_2 = criterion(y2_pred, local_y2)
                loss_3 = criterion(y3_pred, local_y3)
                
                loss = (loss_1+loss_2/loss_weight+loss_3/loss_weight)/3.0#+var_pred/10
                
                i+=1
#                print(i, loss)
                loss.backward()
                optimizer.step()
                
                
                # print statistics
                running_loss_feed += loss.item()
                running_loss_y1 += loss_1.item()
                running_loss_y2 += loss_2.item()
                running_loss_y3 += loss_3.item()
                total_loss_feed += loss.item()
                total_loss_y1 += loss_1.item()
                total_loss_y2 += loss_2.item()
                total_loss_y3 += loss_3.item()
                
                total_count += 1
                if i % 100 == 99:    # print every 2000 mini-batches
                    print('[%d, %2d, %5d] loss: %.3f center: %.3f left: %.3f right: %.3f learning rate: %.4e' %
                          (epoch + 1, sub, i + 1, running_loss_feed / 100, running_loss_y1 / 100, running_loss_y2 / 100, running_loss_y3 / 100, current_lr))
                    # write to log
                    writer.add_scalar('training/total_loss', running_loss_feed / 100, count)
                    writer.add_scalars('training/group',{'center': running_loss_y1 / 100,
                                                         'left': running_loss_y2 / 100,
                                                         'right': running_loss_y3 / 100}, count)
                    writer.add_scalar('training/lr', current_lr, count)

                    running_loss_feed = 0.0
                    running_loss_y1 = 0.0
                    running_loss_y2 =0.0
                    running_loss_y3 = 0.0
                    
                    count+=1
                    
            running_loss_feed = 0.0
            running_loss_y1 = 0.0
            running_loss_y2 =0.0
            running_loss_y3 = 0.0
            
            
        # write to log
        writer.add_scalar('training_overall/total_loss', total_loss_feed / total_count, epoch+1)
        writer.add_scalars('training_overall/group',{'center': total_loss_y1 / total_count,
                                                     'left': total_loss_y2 / total_count,
                                                     'right': total_loss_y3 / total_count}, epoch+1)
        writer.add_scalar('training_overall/lr', current_lr, epoch+1)
        total_loss_feed = 0.0
        total_loss_y1 = 0.0
        total_loss_y2 = 0.0
        total_loss_y3 = 0.0
        
        
        total_count = 0
        torch.save(model.state_dict(), current_model)


        'Validation'
        validation_set = Dataset(files_validation, **params_dataset)
        validation_generator = data.DataLoader(validation_set, **params)
        optimizer.zero_grad()
        model = model.eval()
        
        with torch.set_grad_enabled(False):
            i=0
           
            for (local_batch, local_y1, local_y2, local_y3) in validation_generator:
                # Transfer to GPU
                local_batch = local_batch.to(device=device, dtype=torch.float)
                local_y1 = torch.squeeze(local_y1).to(device=device, dtype=torch.float)
                local_y2 = torch.squeeze(local_y2).to(device=device, dtype=torch.float)
                local_y3 = torch.squeeze(local_y3).to(device=device, dtype=torch.float)
                
                # Model computations
                y1_pred, y2_pred, y3_pred, _ = model(local_batch)
                
                y1_pred = torch.squeeze(y1_pred)
                y2_pred = torch.squeeze(y2_pred)
                y3_pred = torch.squeeze(y3_pred)

                loss_1 = criterion(y1_pred, local_y1)
                loss_2 = criterion(y2_pred, local_y2)
                loss_3 = criterion(y3_pred, local_y3)
                
                loss = (loss_1+loss_2/loss_weight+loss_3/loss_weight)/3.0
                
                i+=1

                # print statistics
                running_loss_feed += loss.item()
                running_loss_y1 += loss_1.item()
                running_loss_y2 += loss_2.item()
                running_loss_y3 += loss_3.item()
                
                
            loss_save={}
            loss_save.update({'overall':running_loss_feed / (i+1),
                              'y1':running_loss_y1 / (i+1),
                              'y2':running_loss_y2 / (i+1),
                              'y3':running_loss_y3 / (i+1),
                              'lr': current_lr})
            loss_on_test.update({epoch+1:loss_save})

            # Display every validation loss
            keys = [k for k in loss_on_test]
            keys.sort()
            for key in keys:
                print('(validation %d) loss: %.3f center: %.3f left: %.3f right: %.3f learning rate: %.4e' %
                      (key, loss_on_test[key]['overall'], loss_on_test[key]['y1'], loss_on_test[key]['y2'], loss_on_test[key]['y3'], loss_on_test[key]['lr']))
            
            writer.add_scalar('validation/total_loss', running_loss_feed / (i+1), epoch+1)
            writer.add_scalars('validation/group',{'center': running_loss_y1 / (i+1),
                                                     'left': running_loss_y2 / (i+1),
                                                     'right': running_loss_y3 / (i+1)}, epoch+1)
            writer.add_scalar('validation/lr', current_lr, epoch+1)
            
            # Save the model if the loss is the lowest
            current_loss = running_loss_feed / (i+1)
            if  (running_loss_feed / (i+1)) < best_loss_on_test:
                best_loss_on_test =  (running_loss_feed / (i+1))
                print("  **")
                torch.save(model.state_dict(), saved_model)
            scheduler.step(torch.tensor([running_loss_feed / (i+1)]).to(device=device, dtype=torch.float))
            
            running_loss_feed = 0.0
            running_loss_y1 = 0.0
            running_loss_y2 = 0.0
            running_loss_y3 = 0.0
            
            
        # Stop early
        stop, stop_count = stop_early(best_loss_on_test, current_loss, stop_count)
        if stop:
            break