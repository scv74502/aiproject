import torch
import torch.nn as nn
import cv2
import os
from torch.utils.data import DataLoader
from torch.autograd import Variable

class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()

        self.conv2d = nn.Conv2d(3, 128, (1,32))
        self.maxpool = nn.MaxPool2d((1,369))
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.lstm = nn.LSTM(128, 64, 2, batch_first=True)
        self.FC = nn.Linear(64, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.maxpool(x)
        x = self.leaky_relu(x)
        h0 = torch.zeros(2, 1000, 64).requires_grad_()
        c0 = torch.zeros(2, 1000, 64).requires_grad_()
        x = x.reshape(1000,-1,128)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.FC(out[-1, :, :])
        # out = self.softmax(out)

        return out

def getfile(path='./subway_mel_spectrogram/'):
    file_list = os.listdir(path)
    file_list_1 = os.listdir('./bus_mel_spectrogram/')
    file_list_py = [file for file in file_list if file.endswith(".png")]
    file_list_py_1 = [file for file in file_list_1 if file.endswith(".png")]
    # print("file_list_py: {}".format(file_list_py))
    ims = []
    for i, img in enumerate(file_list_py):
        im = cv2.imread(path + img)
        ims.append(im)
    ims = torch.FloatTensor(ims)
    ims = torch.transpose(ims, 1, 3)

    ims_1 = []
    for i, img in enumerate(file_list_py_1):
        im = cv2.imread('./bus_mel_spectrogram/' + img)
        ims_1.append(im)
    ims_1 = torch.FloatTensor(ims_1)
    ims_1 = torch.transpose(ims_1, 1, 3)
    ims = torch.cat([ims, ims_1], dim=0)
    label = torch.ones(len(ims), dtype=torch.long)
    label_1 = torch.zeros(len(ims_1), dtype=torch.long)
    labels = torch.cat([label, label_1], dim=0)
    # labels = torch.zeros(len(file_list_py),dtype=torch.long)
    # labels[11:17] = 1
    return ims, labels

def dataloader(ims, labels):
    dataset = list(zip(ims, labels))
    train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)
    return train_loader

if __name__ == '__main__':
    model = LSTMModel()
    # model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.1
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    ims, labels = getfile()
    train_loader = dataloader(ims, labels)
    i = 0
    for epoch in range(10):
        print('epoch : ',epoch)
        for i,data in enumerate(train_loader):
            print('batch number : ', i)
            img, label = data
            img, label = Variable(img), Variable(label)


            # img = img.cuda()


            output = model(img)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("Epoch no:", epoch + 1, "| Epoch progress(%):",
                  format(i / (len(train_loader) / 32) * 100, '0.2f'), "| Avg train loss:",
                  format(loss, '0.4f'))
    savePath = "./model/ver1.h5"
    torch.save(model.state_dict(), savePath)





