import torch
import torch.nn as nn
import cv2
import os
from torch.utils.data import DataLoader
from torch.autograd import Variable
import model


def getfile(path='./mel_spectrogram/'):
    file_list = os.listdir(path)
    file_list_py = [file for file in file_list if file.endswith(".png")]
    print("file_list_py: {}".format(file_list_py))
    ims = []
    for i, img in enumerate(file_list_py):
        im = cv2.imread(path + img)
        ims.append(im)
    ims = torch.FloatTensor(ims)
    ims = torch.transpose(ims, 1, 3)

    labels = torch.zeros(len(file_list_py),dtype=torch.long)
    labels[11:17] = 1
    return ims, labels

def dataloader(ims, labels):
    dataset = list(zip(ims, labels))
    train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=2)
    return train_loader


new_model = model.LSTMModel()
new_model.load_state_dict(torch.load("./model/ver1.h5"))
ims, labels = getfile()
test_loader = dataloader(ims, labels)
criterion = nn.CrossEntropyLoss()
for i, data in enumerate(test_loader):
        img, label = data
        output = new_model(img)
        loss = criterion(output, label)

        print(loss)