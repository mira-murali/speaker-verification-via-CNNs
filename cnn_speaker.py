
import os
import sys
import time

import numpy as np

import torch
import torch.nn as nn
import torchvision
import torch.utils.data as data
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import argparse

from utils import train_load, dev_load, test_load
from utils import EER
# from preprocessing import cifar_10_preprocess

parser = argparse.ArgumentParser("Speaker Verification CNN")
parser.add_argument('--batch_size', default=16, type=int, help='Set batch size to load in data')
parser.add_argument('--num-frames', default=14000, type=int, help='Number of frames taken from each utterance')
parser.add_argument('--lr', default=0.1, type=float, help='Learning rate for model')
parser.add_argument('--epochs', default=20, type=int, help='Total number of epochs')
parser.add_argument('--print_freq', default=50, type=int, help='Printing frequency, i.e., print after every i iterations per epoch')
parser.add_argument('--data-path', default='./', type=str, help='Data path to load in the data')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for the optimizer')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay for optimizer')
parser.add_argument('--resume', default=0, type=int, help='To test on data by loading in previously trained model')
parser.add_argument('--end-range', default=2, type=int, help='Range of training chunks to load in \
(setting to 2 would load in first chunk, setting to 3 would load in first 2 chunks, etc.)')
parser.add_argument('--modelnum', default='1', type=str, help='Model name to store the trained model, will be stored in the same directory')
parser.add_argument('--finetune', default=0, type=int, help='To finetune model further on larger data by loading in previously trained model')
parser.add_argument('--toload', default='1', type=str, help='Name of model to load in for resuming/finetuning. must be in the same directory')
args = parser.parse_args()


class PrepDataset(data.Dataset):
    def __init__(self, data, labels, window):
     self.data = data
     self.window = window
     if labels is not None:
         self.labels = labels
     else:
         self.labels = None

    def __len__(self):
     return len(self.data)

    def __getitem__(self, idx):
     data = self.data[idx]
     data = data.astype('float')
     index_range = data.shape[0]
     start_index = np.random.randint(0, index_range)
     if start_index+self.window >= index_range:
         diff = abs(index_range - (start_index+self.window))
         padded_data = np.pad(data, ((0, diff), (0, 0)), 'wrap')
         data = padded_data[start_index:start_index+self.window]
     else:
         data = data[start_index:start_index+self.window]

     #data = np.transpose(data)
     data = data[None, :, :]

     if self.labels is not None:
         labels = self.labels[idx]
         return (data, labels)
     else:
         return (data, 0)

class residualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(residualBlock, self).__init__()
        self.resblock = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(out_channels),
                                       nn.ELU(inplace=True),
                                       nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(out_channels),
                                       nn.ELU(inplace=True)
                                       )

    def forward(self, x):
        y = self.resblock(x)
        y = y + x
        return y

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x


class speakCNN(nn.Module):
    def __init__(self, num_classes):
     super(speakCNN, self).__init__()
     self.features = nn.Sequential(nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=0, bias=False),
                                   #PrintLayer(),
                                   nn.ELU(),
                                   residualBlock(32, 32),
                                   nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=0, bias=False),
                                   #PrintLayer(),
                                   nn.ELU(),
                                   residualBlock(64, 64),
                                   nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=0, bias=False),
                                   #PrintLayer(),
                                   nn.ELU(),
                                   residualBlock(128, 128),
                                   nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=0, bias=False),
                                   #PrintLayer(),
                                   nn.ELU(),
                                   residualBlock(256, 256)
                                   # PrintLayer()
                                   #PrintLayer()
                                   )
     # self.avg_layer = nn.AvgPool2d(kernel_size=(32,1), stride=(32, 1))
     self.classifier = nn.Linear(256, num_classes)

    def forward(self, x, alpha=16):
     y = self.features(x)
     y = torch.mean(y, dim=2)
     y_flat = y.view(y.size(0), -1)
     y_norm = torch.norm(y_flat, p=2, dim=1)
     y_new = y_flat/y_norm[:, None]
     output = self.classifier(y_new)
     return output, y_new



def main():

    global args
    model = speakCNN(num_classes=3429)
    model = model.cuda()
    print(model)
    print("Please Note: The model was initially trained on chunk 1, and the best model was saved "
    "and loaded in while training on chunks 1-3 and the same for when training on all 6 chunks. "
    "In order to change the default configuration of the model (number of frames, "
    "learning rate, chunks loaded in, etc.) please take a look at the argparser "
    "in the very beginning of the code.")

    model.apply(weight_init)
    flat_shape = 256
    bs = args.batch_size

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    criterion = nn.CrossEntropyLoss()
    if args.resume:
        checkpoint = torch.load('model_'+args.toload+'.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])
        dev_trials, dev_labels, dev_enroll, dev_test = dev_load(args.data_path+'dev.preprocessed.npz')
        test_trials, test_enroll, test_test = test_load(args.data_path+'test.preprocessed.npz')

        val_enroll_dataset = PrepDataset(data=dev_enroll, labels=None, window=args.num_frames)
        val_enroll_loader = data.DataLoader(val_enroll_dataset, shuffle=False, batch_size=args.batch_size, num_workers=0)
        val_test_dataset = PrepDataset(data=dev_test, labels=None, window=args.num_frames)
        val_test_loader = data.DataLoader(val_test_dataset, shuffle=False, batch_size=args.batch_size, num_workers=0)
        test_enroll_dataset = PrepDataset(data=test_enroll, labels=None, window=args.num_frames)
        test_enroll_loader = data.DataLoader(test_enroll_dataset, shuffle=False, batch_size=args.batch_size, num_workers=0)
        test_test_dataset = PrepDataset(data=test_test, labels=None, window=args.num_frames)
        test_test_loader = data.DataLoader(test_test_dataset, shuffle=False, batch_size=args.batch_size, num_workers=0)
        val_enroll_embedding = torch.zeros([dev_enroll.shape[0], flat_shape])
        val_test_embedding = torch.zeros([dev_test.shape[0], flat_shape])
        test_enroll_embedding = torch.zeros([test_enroll.shape[0], flat_shape])
        test_test_embedding = torch.zeros([test_test.shape[0], flat_shape])
        if args.finetune:
            features, speakers, nspeakers = train_load(args.data_path, range(1,args.end_range))
            train_dataset = PrepDataset(data=features, labels=speakers, window=args.num_frames)
            train_loader = data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=0)
            train_accuracy = []
            train_loss = []

            min_loss = 10
            epochs = args.epochs
            best_model = None
            min_err = 0.5
            lr = args.lr
            max_train = 0
            for epoch in range(epochs):
                adjust_learning_rate(optimizer, epoch)
                acc_train, loss_train = train(model, train_loader, criterion, optimizer, epoch)
                if max_train < acc_train:
                    max_train = acc_train
                    best_model = model
                #
                # train_accuracy.append(acc_train)
                # train_loss.append(loss_train)
                print("Training accuracy per epoch: ", acc_train)

                val_scores = []
                model.eval()
                with torch.no_grad():
                    for i, (input, _) in enumerate(val_enroll_loader):
                        input = input.float().cuda()
                        _, out_feat = model(input)
                        val_enroll_embedding[i*bs:(i+1)*bs] = out_feat
                    for i, (input, _) in enumerate(val_test_loader):
                        input = input.float().cuda()
                        _, out_feat = model(input)
                        val_test_embedding[i*bs:(i+1)*bs] = out_feat
                    for dev_ind, test_ind in dev_trials:
                        val_feat = val_enroll_embedding[dev_ind]
                        test_feat = val_test_embedding[test_ind]
                        val_score = F.cosine_similarity(val_feat, test_feat, dim=0)
                        val_scores.append(val_score.item())
                    val_scores = np.array(val_scores)
                    eer, thresh = EER(dev_labels, val_scores)
                    print("EER: ", eer)
                    if eer < min_err:
                        min_err = eer
                        checkpoint = {'state_dict': model.state_dict()}
                        torch.save(checkpoint, 'model_'+args.modelnum+'.pth.tar')


            model.eval()
            test_scores = []
            with torch.no_grad():
                for i, (input, _) in enumerate(test_enroll_loader):
                    input = input.float().cuda()
                    _, out_feat = model(input)
                    test_enroll_embedding[i*bs:(i+1)*bs] = out_feat

                for i, (input, _) in enumerate(test_test_loader):
                    input = input.float().cuda()
                    _, out_feat = model(input)
                    test_test_embedding[i*bs:(i+1)*bs] = out_feat

                for enroll_ind, test_ind in test_trials:
                    val_feat = test_enroll_embedding[enroll_ind]
                    test_feat = test_test_embedding[test_ind]
                    test_score = F.cosine_similarity(val_feat, test_feat, dim=0)
                    test_scores.append(test_score.item())

            test_scores = np.array(test_scores)
            np.save('test_scores.npy', test_scores)
        else:

            model.eval()


            with torch.no_grad():
                val_scores = []
                for i, (input, _) in enumerate(val_enroll_loader):
                    input = input.float().cuda()
                    _, out_feat = model(input)
                    val_enroll_embedding[i*bs:(i+1)*bs] = out_feat
                for i, (input, _) in enumerate(val_test_loader):
                    input = input.float().cuda()
                    _, out_feat = model(input)
                    val_test_embedding[i*bs:(i+1)*bs] = out_feat
                for dev_ind, test_ind in dev_trials:
                    val_feat = val_enroll_embedding[dev_ind]
                    test_feat = val_test_embedding[test_ind]
                    val_score = F.cosine_similarity(val_feat, test_feat, dim=0)
                    val_scores.append(val_score.item())
                val_scores = np.array(val_scores)
                eer, thresh = EER(dev_labels, val_scores)
                print("EER: ", eer)

                test_scores = []
                for i, (input, _) in enumerate(test_enroll_loader):
                    #print(len(test_enroll_loader))
                    input = input.float().cuda()
                    _, out_feat = model(input)
                    test_enroll_embedding[i*bs:(i+1)*bs] = out_feat

                for i, (input, _) in enumerate(test_test_loader):
                    input = input.float().cuda()
                    _, out_feat = model(input)
                    test_test_embedding[i*bs:(i+1)*bs] = out_feat

                for enroll_ind, test_ind in test_trials:
                    val_feat = test_enroll_embedding[enroll_ind]
                    test_feat = test_test_embedding[test_ind]
                    test_score = F.cosine_similarity(val_feat, test_feat, dim=0)
                    test_scores.append(test_score.item())

            print("Saving test scores")
            test_scores = np.array(test_scores)
            np.save('test_scores.npy', test_scores)
            sys.exit(0)
    else:


        features, speakers, nspeakers = train_load(args.data_path, range(1,args.end_range))
        dev_trials, dev_labels, dev_enroll, dev_test = dev_load(args.data_path+'dev.preprocessed.npz')
        test_trials, test_enroll, test_test = test_load(args.data_path+'test.preprocessed.npz')


        train_dataset = PrepDataset(data=features, labels=speakers, window=args.num_frames)
        train_loader = data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=0)
        val_enroll_dataset = PrepDataset(data=dev_enroll, labels=None, window=args.num_frames)
        val_enroll_loader = data.DataLoader(val_enroll_dataset, shuffle=False, batch_size=args.batch_size, num_workers=0)
        val_test_dataset = PrepDataset(data=dev_test, labels=None, window=args.num_frames)
        val_test_loader = data.DataLoader(val_test_dataset, shuffle=False, batch_size=args.batch_size, num_workers=0)
        test_enroll_dataset = PrepDataset(data=test_enroll, labels=None, window=args.num_frames)
        test_enroll_loader = data.DataLoader(test_enroll_dataset, shuffle=False, batch_size=args.batch_size, num_workers=0)
        test_test_dataset = PrepDataset(data=test_test, labels=None, window=args.num_frames)
        test_test_loader = data.DataLoader(test_test_dataset, shuffle=False, batch_size=args.batch_size, num_workers=0)

        val_enroll_embedding = torch.zeros([dev_enroll.shape[0], flat_shape])
        val_test_embedding = torch.zeros([dev_test.shape[0], flat_shape])
        test_enroll_embedding = torch.zeros([test_enroll.shape[0], flat_shape])
        test_test_embedding = torch.zeros([test_test.shape[0], flat_shape])





        train_accuracy = []
        train_loss = []

        max_train = 0
        epochs = args.epochs
        best_model = None
        min_err = 0.5
        for epoch in range(epochs):
            adjust_learning_rate(optimizer, epoch)
            acc_train, loss_train = train(model, train_loader, criterion, optimizer, epoch)
            if max_train < acc_train:
                max_train = acc_train
                best_model = model

            train_accuracy.append(acc_train)
            train_loss.append(loss_train)
            print("Training accuracy per epoch: ", acc_train)

            #if epoch%5==0:
            val_scores = []
            model.eval()
            with torch.no_grad():
                for i, (input, _) in enumerate(val_enroll_loader):
                    input = input.float().cuda()
                    _, out_feat = model(input)
                    val_enroll_embedding[i*bs:(i+1)*bs] = out_feat
                for i, (input, _) in enumerate(val_test_loader):
                    input = input.float().cuda()
                    _, out_feat = model(input)
                    val_test_embedding[i*bs:(i+1)*bs] = out_feat
                for dev_ind, test_ind in dev_trials:
                    val_feat = val_enroll_embedding[dev_ind]
                    test_feat = val_test_embedding[test_ind]
                    val_score = F.cosine_similarity(val_feat, test_feat, dim=0)
                    val_scores.append(val_score.item())
                val_scores = np.array(val_scores)
                eer, thresh = EER(dev_labels, val_scores)
                print("EER: ", eer)
                if eer < min_err:
                    min_err = eer
                    checkpoint = {'state_dict': model.state_dict()}
                    torch.save(checkpoint, 'model_'+args.modelnum+'.pth.tar')

        model.eval()
        test_scores = []
        with torch.no_grad():
            for i, (input, _) in enumerate(test_enroll_loader):
                input = input.float().cuda()
                _, out_feat = model(input)
                test_enroll_embedding[i*bs:(i+1)*bs] = out_feat

            for i, (input, _) in enumerate(test_test_loader):
                input = input.float().cuda()
                _, out_feat = model(input)
                test_test_embedding[i*bs:(i+1)*bs] = out_feat

            for enroll_ind, test_ind in test_trials:
                val_feat = test_enroll_embedding[enroll_ind]
                test_feat = test_test_embedding[test_ind]
                test_score = F.cosine_similarity(val_feat, test_feat, dim=0)
                test_scores.append(test_score.item())

        test_scores = np.array(test_scores)
        np.save('test_scores.npy', test_scores)

def train(model, train_loader, criterion, optimizer, epoch):
    global args
    count = 0
    total = 0
    correct = 0
    avg_loss = 0
    start = time.time()
    model.train()
    for i, (data, speaker) in enumerate(train_loader):
     data, speaker = data.float().cuda(), speaker.long().cuda()
     data_time = time.time() - start
     output, feature = model(data)

     loss = criterion(output, speaker)

     optimizer.zero_grad()
     loss.backward()
     optimizer.step()
     total += speaker.size(0)
     _, pred = torch.max(output, 1)

     correct += (pred==speaker).sum().item()
     avg_loss += loss.item()
     correct_present = (pred==speaker).sum().item()
     batch_time = time.time() - start
     if i%args.print_freq==0:
         print('Epoch: [{0}]:[{1}/{2}]\t'
               'Loss: {3}\t'
               'Accuracy: {4}\t'
               'Data time: {5}\t'
               'Batch time: {6}\t'.format(epoch, i, len(train_loader), loss,
               (correct_present/speaker.size(0))*100, data_time, batch_time))


    avg_loss = avg_loss/len(train_loader)
    accuracy = (correct/total)*100
    return accuracy, avg_loss

def weight_init(layer):
    if type(layer) == nn.Conv2d:
        nn.init.kaiming_normal_(layer.weight)
    elif type(layer) == nn.BatchNorm2d:
        nn.init.constant_(layer.weight, 1)
        nn.init.constant_(layer.bias, 0)

def adjust_learning_rate(optimizer, epoch):
    global args
    lr = args.lr
    if epoch>=int(args.epochs/2):
     lr *= 0.1
    for param_group in optimizer.param_groups:
     param_group['lr'] = lr
if __name__=='__main__':
    main()
