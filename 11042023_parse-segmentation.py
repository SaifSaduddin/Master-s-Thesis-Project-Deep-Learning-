# from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
import sys
import numpy as np
from skimage import segmentation
import torch.nn.init
import os
import shutil

use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation')
parser.add_argument('--nChannel', metavar='N', default=100, type=int,
                    help='number of channels')
parser.add_argument('--maxIter', metavar='T', default=1000, type=int,
                    help='number of maximum iterations')
parser.add_argument('--minLabels', metavar='minL', default=3, type=int,
                    help='minimum number of labels')
parser.add_argument('--lr', metavar='args.lr', default=0.1, type=float,
                    help='learning rate')
parser.add_argument('--nConv', metavar='M', default=2, type=int,
                    help='number of convolutional layers')
parser.add_argument('--num_superpixels', metavar='K', default=10000, type=int,
                    help='number of superpixels')
parser.add_argument('--compactness', metavar='C', default=100, type=float,
                    help='compactness of superpixels')
parser.add_argument('--visualize', metavar='1 or 0', default=1, type=int,
                    help='visualization flag')
parser.add_argument('--orignal_data_folder', metavar='ORIGNAL_FILENAME',
                    help='input image file name', required=True)
parser.add_argument('--segmented_data', metavar='SAVE_FILENAME',
                    help='input image file name', required=True)
parser.add_argument('--results_folder', metavar='RECORD_SAVE_FILE',
                    help='output data location txt file', required=True)

args = parser.parse_args()


# CNN model
class MyNet(nn.Module):
    def __init__(self, input_dim):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, args.nChannel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(args.nChannel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(args.nConv - 1):
            self.conv2.append(nn.Conv2d(args.nChannel, args.nChannel, kernel_size=3, stride=1, padding=1))
            self.bn2.append(nn.BatchNorm2d(args.nChannel))
        self.conv3 = nn.Conv2d(args.nChannel, args.nChannel, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(args.nChannel)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        for i in range(args.nConv - 1):
            x = self.conv2[i](x)
            x = F.relu(x)
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x


# args.orignal_data_folder = 'D:\AI\code\Azure_notebook_data\medical images predictions\dataset\orignal'
# args.segmented_data = 'D:\AI\code\Azure_notebook_data\medical images predictions\dataset\args.segmented_data'


if os.path.exists(args.segmented_data):
    shutil.rmtree(args.segmented_data)
    
if os.path.exists(args.results_folder): ###########3
    shutil.rmtree(args.results_folder)    ################3


for root, folders, path in os.walk(args.orignal_data_folder):
#     print(root)
#     print(folders)
#     print(path)

    for folder in folders:
        save_folder = os.path.join(args.segmented_data, folder)
        result_location = os.path.join(args.results_folder, folder)  ####################
        print(save_folder)

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            os.makedirs(result_location) ##############################
            
        print(save_folder)
        files = os.listdir(os.path.join(root, folder)) 
        print('\nSegmenting ' + str(folder) + ' folder...')
        for file in files:
            file_split = file.split('.')  ######################3
            file_name = file_split[-2]+'.txt'            #########################
            print(file_name)                   ####################################
            
            # img read
            input = os.path.join(root, folder, file)
            im = cv2.imread(input)  # args.input
            data = torch.from_numpy(np.array([im.transpose((2, 0, 1)).astype('float32') / 255.]))
            if use_cuda:
                data = data.cuda()
            data = Variable(data)

            # slic
            labels = segmentation.slic(im, compactness = args.compactness, n_segments = args.num_superpixels)
            labels = labels.reshape(im.shape[0] * im.shape[1])
            u_labels = np.unique(labels)
            l_inds = []
            for i in range(len(u_labels)):
                l_inds.append(np.where(labels == u_labels[i])[0])

            # train
            print('image ' + str(file))
            model = MyNet(data.size(1))
            if use_cuda:
                model.cuda()
            model.train()
            loss_fn = torch.nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum = 0.9)
            label_colours = np.random.randint(255, size=(100, 3))
            for batch_idx in range(args.maxIter):
                batch_idx += 1
                # forwarding
                optimizer.zero_grad()
                output = model(data)[0]
                output = output.permute(1, 2, 0).contiguous().view(-1, args.nChannel)
                ignore, target = torch.max(output, 1)
                im_target = target.data.cpu().numpy()
                nLabels = len(np.unique(im_target))
                if args.visualize:
                    im_target_rgb = np.array([label_colours[c % 100] for c in im_target])
                    im_target_rgb = im_target_rgb.reshape(im.shape).astype(np.uint8)
                    # cv2.imshow("output", im_target_rgb)
                    cv2.waitKey(10)

                # superpixel refinement
                # TODO: use Torch Variable instead of numpy for faster calculation
                for i in range(len(l_inds)):
                    labels_per_sp = im_target[l_inds[i]]
                    u_labels_per_sp = np.unique(labels_per_sp)
                    hist = np.zeros(len(u_labels_per_sp))
                    for j in range(len(hist)):
                        hist[j] = len(np.where(labels_per_sp == u_labels_per_sp[j])[0])
                    im_target[l_inds[i]] = u_labels_per_sp[np.argmax(hist)]
                target = torch.from_numpy(im_target)
                if use_cuda:
                    target = target.cuda()
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()

                # print (batch_idx, '/', args.maxIter, ':', nLabels, loss.data[0])
                print(batch_idx, '/', args.maxIter, ':', nLabels, loss.item())
                
                f_image_wise = open(os.path.join(result_location, file_name),"a+")               ###################3
                f_image_wise.write(str(batch_idx)+','+str(args.maxIter)+','+str(nLabels)+','+str(loss.item())+'\n')  #################3
                f_image_wise.close()                            ##########################################
                
                

                if nLabels <= args.minLabels:
                    print("nLabels", nLabels, "reached args.minLabels", args.minLabels, ".")
                    break

            if not args.visualize:
                output = model(data)[0]
                output = output.permute(1, 2, 0).contiguous().view(-1, args.nChannel)
                ignore, target = torch.max(output, 1)
                im_target = target.data.cpu().numpy()
                im_target_rgb = np.array([label_colours[c % 100] for c in im_target])
                im_target_rgb = im_target_rgb.reshape(im.shape).astype(np.uint8)

            #             image_save_file_name = file
            save_file_path = os.path.join(args.segmented_data, folder, file)
            cv2.imwrite(save_file_path, im_target_rgb)
            
            f_image_wise = open(os.path.join(result_location, folder+'.txt'),"a+") ###########################
            f_image_wise.write(file.split('.')[-2]+','+str(batch_idx)+','+str(loss.item())+'\n') ############################3
            f_image_wise.close() ###################################################
            
            
            print('file ' + str(file) + ' saved...')

        print('Total ' + str(len(files)) + ' images segmented in ' + str(folder) + ' folder')
print('\nsegmentation completed ...\n')
