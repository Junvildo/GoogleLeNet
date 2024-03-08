import torch
from model import *

input = torch.rand(1,3,224,224)

model = GoogLeNet()
model.state_train = True
output = model(input)

if model.state_train == True:
    print(output[0].shape)
    print(output[1].shape)
    print(output[2].shape)
    print(torch.argmax(torch.softmax(output[0], dim=1), dim=1))
    print(torch.argmax(torch.softmax(output[1], dim=1), dim=1))
    print(torch.argmax(torch.softmax(output[2], dim=1), dim=1))
else:
    print(output.shape)
    print(torch.argmax(torch.softmax(output, dim=1), dim=1))