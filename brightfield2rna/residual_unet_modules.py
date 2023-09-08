import sys
import os
from optparse import OptionParser
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import random
import pickle

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable as var

import tifffile as tf

from tqdm import tqdm

import random as rd

import pickle

padding = (2,2)
kernel = (5,5)

pix = 50
n_img = 4


'''Building Blocks'''
class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, kernel_Size):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_Size, stride=1, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_Size, stride=2, padding=padding)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_Size):
        super(inconv, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_Size, stride=1, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_Size, stride=1, padding=padding)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_Size):
        super(down, self).__init__()
        self.mpconv = double_conv(in_ch, out_ch, kernel_Size)
        self.inconv = inconv(out_ch, out_ch, kernel_Size)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_Size, stride=2, padding=padding)
        self.dropout = nn.Dropout(p=0.02)
#            nn.MaxPool2d(2), # this kernel size could be changed       

    def forward(self, x):
        x1 = self.mpconv(x)
        x2 = self.conv(x)
        x = torch.add(x1,x2)
        x = self.dropout(x)
        x3 = self.inconv(x)
        x = torch.add(x,x3)
        x = self.dropout(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, cat_ch, out_ch, kernel_size, bilinear=False):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size, stride=2, padding=padding)

        self.conv = inconv(in_ch+cat_ch, out_ch, kernel_size)
        self.BN1 = nn.BatchNorm2d(in_ch)
        self.BN2 = nn.BatchNorm2d(out_ch)
        self.ReLU = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.02)

    def forward(self, x0, x2):
        x1 = self.BN1(x0)
        x1 = self.ReLU(x1)
        x1 = self.up(x1)
        x1 = self.BN1(x1)
        x1 = self.ReLU(x1)
        x3 = self.up(x0)
        x1 = torch.add(x1,x3)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
      #  diffZ = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        #                diffZ // 2, diffZ - diffZ//2)) # not sure it is correct

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.dropout(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_Size):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_Size, stride=1, padding=padding)
        self.inconv = inconv(in_ch, out_ch, kernel_Size)
        self.dropout = nn.Dropout(p=0.02)

    def forward(self, x):
        x2 = self.inconv(x)
        x3 = self.conv(x)
        x = torch.add(x2,x3)
        x = self.dropout(x)
        return x

''' Unet Design'''
class UNet_3(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_3, self).__init__()
#        self.inc = inconv(n_channels, 4,(3,3))

        self.down1 = down(n_channels, 8, kernel)
        self.down2 = down(8, 16, kernel)
        self.down3 = down(16, 32, kernel)
        self.down4 = down(32, 64, kernel)
        self.down5 = down(64, 128, kernel)
        self.up1 = up(128,64,200,kernel)
        self.up2 = up(200,32,180,kernel)
        self.up3 = up(180,16,160,kernel)
        self.up4 = up(160,8,140,kernel)
        self.up5 = up(140,n_channels,120,kernel)
        
        self.outc = outconv(120, n_classes,kernel)

    def forward(self, x):
#        x1 = self.inc(x)
        x0 = x

        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x = self.up1(x5,x4)
        x = self.up2(x,x3)
        x = self.up3(x,x2)
        x = self.up4(x,x1)
        x = self.up5(x,x0)

        x = self.outc(x)
        return x

'''loss function'''
class npcc_lossFunc(nn.Module):
    def _init_(self):
        super(npcc_lossFunc,self).__init__()
        return

    def forward(self, G, T):

        fsp = G - torch.mean(G,0)
        fst = T - torch.mean(T,0)
        devP = torch.std(G,0,True,False)
        devT = torch.std(T,0,True, False)

        npcc_loss=(-1)*torch.mean(fsp*fst,0)/torch.clamp(devP*devT,1e-7,None)
        npcc_loss=torch.mean(npcc_loss,0)
        return npcc_loss

    def backward(self,grad_output):
        return grad_output
    
class mse_lossFunc(nn.Module):
    def _init_(self):
        super(mse_lossFunc,self).__init__()
        return

    def forward(self, G, T):
        loss = nn.MSELoss()
        mse_loss = loss(G,T)
        
        return mse_loss

    def backward(self,grad_output):
        return grad_output
        
class mae_lossFunc(nn.Module):
    def _init_(self):
        super(mae_lossFunc,self).__init__()
        return

    def forward(self, G, T):
        loss = nn.L1Loss()
        mae_loss = loss(G,T)
        return mae_loss

    def backward(self,grad_output):
        return grad_output
    
    
'''data loading'''
def CreateDataset(data_path, test_percent, input_c, gt_c):
    Train_dataset = []
    
    paths_list = os.listdir(os.path.join(data_path, input_c))
    gt_paths_list = os.listdir(os.path.join(data_path, gt_c))
    
    train_len = int(len(paths_list)*(1-test_percent))
    test_len = len(paths_list)-train_len
    Train_idxes = range(0,train_len)
    Test_idxes = range(train_len, len(paths_list))
  
    for idx in tqdm(Train_idxes):
        inputpath = os.path.join(data_path, input_c, paths_list[idx])
        gtpath = os.path.join(data_path, gt_c, gt_paths_list[idx])
        if (os.path.exists(inputpath) == False):
            return -1
        else:
            if (paths_list[idx].find('.tiff') == -1) or (gt_paths_list[idx].find('.tiff') == -1):
                continue
            img = tf.imread(inputpath)
            if np.any(np.isnan(img)):
                continue
            gt = tf.imread(gtpath)
            
            # sample random regions, upsample 4x
            
            w = img.shape[1]
            for i in range(n_img):
                r_idx = rd.randint(0,w-pix)
                r_idy = rd.randint(0,w-pix)
                img_cr = img[:,r_idx:r_idx+pix,r_idy:r_idy+pix]
                gt_cr = gt[r_idx:r_idx+pix,r_idy:r_idy+pix]
                
                Train_dataset.append([img_cr,gt_cr])
                
                # flip
                Train_dataset.append([np.flip(img_cr,1),np.flip(gt_cr,0)])
                Train_dataset.append([np.flip(img_cr,2),np.flip(gt_cr,1)])
                
                # rotate
                Train_dataset.append([np.rot90(img_cr, axes=(1,2)),np.rot90(gt_cr, axes=(0,1))])
                Train_dataset.append([np.rot90(img_cr, axes=(2,1)),np.rot90(gt_cr, axes=(1,0))])
                
#             Train_dataset.append([img,gt])
    Test_dataset = []
    for ii in tqdm(Test_idxes):
        inputpath = os.path.join(data_path, input_c, paths_list[ii])
        gtpath = os.path.join(data_path, gt_c, gt_paths_list[ii])
        if (os.path.exists(inputpath) == False):# or (os.path.exists(gtpath)) is False:
            return -1
        else:
            if (paths_list[idx].find('.tiff') == -1) or (gt_paths_list[idx].find('.tiff') == -1):
                continue
            img1 = tf.imread(inputpath)
            if np.any(np.isnan(img1)):
                continue
            gt1 = tf.imread(gtpath)
            
            # sample random regions, upsample 4x
            w = img1.shape[1]
            for i in range(n_img):
                r_idx = rd.randint(0,w-pix)
                r_idy = rd.randint(0,w-pix)
                img1_cr = img1[:,r_idx:r_idx+pix,r_idy:r_idy+pix]
                gt1_cr = gt1[r_idx:r_idx+pix,r_idy:r_idy+pix]
                
                Test_dataset.append([img1_cr,gt1_cr])
                
                # flip
                Test_dataset.append([np.flip(img1_cr,1),np.flip(gt1_cr,0)])
                Test_dataset.append([np.flip(img1_cr,2),np.flip(gt1_cr,1)])

                # rotate
                Test_dataset.append([np.rot90(img1_cr, axes=(1,2)),np.rot90(gt1_cr, axes=(0,1))])
                Test_dataset.append([np.rot90(img1_cr, axes=(2,1)),np.rot90(gt1_cr, axes=(1,0))])
                
#             Test_dataset.append([img1,gt1])
    return Train_dataset, Test_dataset

def split_train_val(dataset, val_percent=0.05):
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    random.shuffle(dataset)
    return {'train': dataset[:-n], 'val': dataset[-n:]}

def batch(iterable, batch_size):
    """Yields lists by batch"""
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b
        
def save_loss_plot(train_loss, val_loss):
    plt.clf()
    x = range(1,len(train_loss)+1)
#     x = np.arange(1,501)

    y1 = np.array(train_loss)
    y2 = np.array(val_loss)

    plt.plot(x, y1, color="r", linestyle="-", linewidth=1, label="training_loss")
    plt.plot(x, y2, color="b", linestyle="-", linewidth=1, label="validation_loss")

    plt.xlabel("epochs")
    plt.ylabel("loss")

    plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))

    plt.title("Training Curve")

    plt.savefig('Training curve.png')

#     plt.show()
        
        
'''Training'''
def train_net(net, epochs, batch_size, lr, val_percent, save_cp, gpu, Train_dataset, Test_dataset, loss, save_path):# training function
#     test_percent = 0.1
# import data
     
#     [Train_dataset, Test_dataset] = CreateDataset(data_path, test_percent, input_c, gt_c)
    iddataset = split_train_val(Train_dataset, val_percent) # shuffle the data
    train = iddataset['train']
    val = iddataset['val']
# make direction for training results
    resultFolder = save_path
    if (os.path.exists(resultFolder)) is False:
        os.makedirs(resultFolder)
    else:
        pass

    print('''
        Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(iddataset['train']),
               len(iddataset['val']), str(save_cp), str(gpu)))
# optimizer and loss functions
    N_train = len(iddataset['train'])
    optimizer_SGD = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay = 0.0005)
    optimizer_adam = optim.Adam(net.parameters(), lr=lr, betas=[0.9,0.999], weight_decay = 0.0)
    
    if (loss=='mse'):
        criterion = mse_lossFunc()
    elif (loss=='npcc'):
        criterion = npcc_lossFunc()
    elif (loss=='mae'):
        criterion = mae_lossFunc()
        
    if gpu:
        criterion.cuda()
    
    train_loss = []
    val_loss = []
    test_loss = []

    for epoch in range(epochs):
# training
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()
#        iddataset = split_train_val(Train_dataset, val_percent) # shuffle the data    
#        train = iddataset['train']
#        val = iddataset['val']
        epoch_loss = 0

        for i, b in enumerate(batch(train, batch_size)):
            imgs = np.array([i[0] for i in b]).astype(np.float32)
            gts = np.array([i[1] for i in b]).astype(np.float32)
            
            imgs = torch.from_numpy(imgs)
            gts = torch.from_numpy(gts)
            if gpu:
                imgs = imgs.cuda()
                gts = gts.cuda()
            gts_pred = net(imgs)
#            print(gts_pred.shape)
            gts_pred_flat = gts_pred.view(-1)
            gts_flat = gts.view(-1)

            loss = criterion(gts_pred_flat, gts_flat)
            epoch_loss += loss.item()
            print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train, loss.item()))

            optimizer_adam.zero_grad()
            loss.backward()
            optimizer_adam.step()

        if epoch % 10 == 0:
            netFile = resultFolder + '/net_'+str(epoch)+'.pth'
            torch.save(net,netFile)
        else:
            pass

        print('Epoch finished ! Training Loss: {}'.format(epoch_loss / i))
        train_loss.append(epoch_loss / i)
## validation
        net.eval()
        val_epoch_loss = 0
#        batchsize = 1

        for i, b in enumerate(batch(val, 1)): 
            img = np.array([i[0] for i in b]).astype(np.float32)
            gt = np.array([i[1] for i in b]).astype(np.float32)
            img = torch.from_numpy(img)
            gt = torch.from_numpy(gt)
            if gpu:
                img = img.cuda()
                gt = gt.cuda()
            gt_pred = net(img)
            gt_pred_flat = gt_pred.view(-1)
            gt_flat = gt.view(-1)

            loss = criterion(gt_pred_flat, gt_flat)
            val_epoch_loss += loss.item()
        print('Validation Loss: {}'.format(val_epoch_loss/i))
        val_loss.append(val_epoch_loss/i)
        
        save_loss_plot(train_loss, val_loss)

# Testing
    net.eval()
    test_loss = 0
    idx = 0

    resultFolder_last = resultFolder+'/epoch_last'
    if not os.path.isdir(resultFolder_last):
        os.mkdir(resultFolder_last)
    if not os.path.isdir(resultFolder_last+'/test'):
        os.mkdir(resultFolder_last+'/test')
        
    for j, b in enumerate(batch(Test_dataset,1)):
        img1 = np.array([j[0] for j in b]).astype(np.float32)
        gt1 = np.array([j[1] for j in b]).astype(np.float32)
        img1 = torch.from_numpy(img1)
        gt1 = torch.from_numpy(gt1)
        idx += 1
        if gpu:
            img1 = img1.cuda()
            gt1 = gt1.cuda()
        
        gt_pred_1 = net(img1)
        gt1_pred = torch.Tensor.cpu(gt_pred_1)
        phi_pred = var(gt1_pred).numpy()
        testFile = resultFolder_last+'/Prediction'+str(idx)+'.pkl'
        pickle.dump(phi_pred, open(testFile, "wb"))
#         scio.savemat(testFile, {'gt1_pred':phi_pred})
        
        gt2 = torch.Tensor.cpu(gt1)
        gt2= var(gt2).numpy()
        gtFile = resultFolder_last+'/GT_'+str(idx)+'.pkl'
        pickle.dump(gt2, open(gtFile, "wb"))
#         scio.savemat(gtFile, {'gt':gt2})

        img2 = torch.Tensor.cpu(img1)
        img2= var(img2).numpy()
        inpFile = resultFolder_last+'/Input_'+str(idx)+'.pkl'
        pickle.dump(img2, open(inpFile, "wb"))
#         scio.savemat(inpFile, {'inp':img2})

        gt1_pred_flat = gt_pred_1.view(-1)
        gt1_flat = gt1.view(-1)
        loss = criterion(gt1_pred_flat, gt1_flat)
        test_loss += loss.item()

    print('Testing Loss: {}'.format(test_loss/j))
    netFile = resultFolder + '/net_pred.pth'
    torch.save(net, netFile)

    y1 = np.array(train_loss)
    y2 = np.array(val_loss)

    NewFile_1 = resultFolder_last + '/training_loss.pkl'
    pickle.dump(y1, open(NewFile_1, "wb"))
#     scio.savemat(NewFile_1, {'y1':y1})

    NewFile_2 = resultFolder_last + '/validation_loss.pkl'
    pickle.dump(y2, open(NewFile_2, "wb"))
#     scio.savemat(NewFile_2, {'y2':y2})

    return train_loss, val_loss

def predict_from_net(net, train, val, test, resultFolder, gpu):
    if not os.path.isdir(resultFolder):
        os.mkdir(resultFolder)
    for i, b in enumerate(batch(train, 1)):
        folder_path = resultFolder+'/train'
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)

        imgs = np.array([i[0] for i in b]).astype(np.float32)
        gts = np.array([i[1] for i in b]).astype(np.float32)
        imgs = torch.from_numpy(imgs)
        gts = torch.from_numpy(gts)
        if gpu:
            imgs = imgs.cuda()
            gts = gts.cuda()
            gts_pred_1 = net(imgs)

            gts_pred = torch.Tensor.cpu(gts_pred_1)
            gts_preds = var(gts_pred).numpy()
            
            gtFile = folder_path+'/Prediction_'+str(i)+'.pkl'
            pickle.dump(gts_preds, open(gtFile, "wb"))
            
            gt2 = torch.Tensor.cpu(gts)
            gt2= var(gt2).numpy()
            gtFile = folder_path+'/GT_'+str(i)+'.pkl'
            pickle.dump(gt2, open(gtFile, "wb"))

            img2 = torch.Tensor.cpu(imgs)
            img2= var(img2).numpy()
            inpFile = folder_path+'/Input_'+str(i)+'.pkl'
            pickle.dump(img2, open(inpFile, "wb"))

    for i, b in enumerate(batch(val, 1)):
        folder_path = resultFolder+'/val'
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)

        imgs = np.array([i[0] for i in b]).astype(np.float32)
        gts = np.array([i[1] for i in b]).astype(np.float32)
        imgs = torch.from_numpy(imgs)
        gts = torch.from_numpy(gts)
        if gpu:
            imgs = imgs.cuda()
            gts = gts.cuda()
            gts_pred_1 = net(imgs)

            gts_pred = torch.Tensor.cpu(gts_pred_1)
            gts_preds = var(gts_pred).numpy()
            
            gtFile = folder_path+'/Prediction_'+str(i)+'.pkl'
            pickle.dump(gts_preds, open(gtFile, "wb"))
            
            gt2 = torch.Tensor.cpu(gts)
            gt2= var(gt2).numpy()
            gtFile = folder_path+'/GT_'+str(i)+'.pkl'
            pickle.dump(gt2, open(gtFile, "wb"))

            img2 = torch.Tensor.cpu(imgs)
            img2= var(img2).numpy()
            inpFile = folder_path+'/Input_'+str(i)+'.pkl'
            pickle.dump(img2, open(inpFile, "wb"))

    for i, b in enumerate(batch(test,1)):
        folder_path = resultFolder+'/test'
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)

        imgs = np.array([i[0] for i in b]).astype(np.float32)
        gts = np.array([i[1] for i in b]).astype(np.float32)
        imgs = torch.from_numpy(imgs)
        gts = torch.from_numpy(gts)
        if gpu:
            imgs = imgs.cuda()
            gts = gts.cuda()
            gts_pred_1 = net(imgs)

            gts_pred = torch.Tensor.cpu(gts_pred_1)
            gts_preds = var(gts_pred).numpy()
            
            gtFile = folder_path+'/Prediction_'+str(i)+'.pkl'
            pickle.dump(gts_preds, open(gtFile, "wb"))
            
            gt2 = torch.Tensor.cpu(gts)
            gt2= var(gt2).numpy()
            gtFile = folder_path+'/GT_'+str(i)+'.pkl'
            pickle.dump(gt2, open(gtFile, "wb"))

            img2 = torch.Tensor.cpu(imgs)
            img2= var(img2).numpy()
            inpFile = folder_path+'/Input_'+str(i)+'.pkl'
            pickle.dump(img2, open(inpFile, "wb"))