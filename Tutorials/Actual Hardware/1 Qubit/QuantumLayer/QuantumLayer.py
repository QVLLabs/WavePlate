from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets,transforms
import torch
import math
import cmath
import numpy as np
import serial
import struct
import os
import time

## -- ARDUINO Functions --

ser = serial.Serial('COM8', 9600)

def send_to_arduino(ser, values):
    ser.write(struct.pack('>BB',*values))

def receive():
    if ser.in_waiting > 0:
        data = ser.readline().decode('utf-8').rstrip()
        return data

def get_exp(value1,value2):
    expects = np.zeros(1)
    shots = value1 + value2
    states = [value1,value2]
    for idx,counts in enumerate(states):
      perc = counts/shots
      check = np.array([( idx - 1/2 )*2*perc])
      expects += check
    return expects

# -- Neural Network and the Quantum Layer -----

def quantum_layer(q_input_features, q_weights_flat):
    value1,value2 = None,None
    q_weights = q_weights_flat.reshape(q_depth, n_qubits)

    for idx, angle in enumerate(q_input_features):
        ang = angle
        while ang < 0:
          ang += math.pi/2
        an = int(((ang*2)*180/math.pi))
        send_to_arduino(ser,[an,idx])

    for layer in range(q_depth):
        for idx, angle in enumerate(q_weights[layer]):
          ang = angle
          while ang < 0:
            ang += math.pi/2
          an = int(((ang*2)*180/math.pi))
      
          send_to_arduino(ser,[an,idx+1])

    time.sleep(2)
    while value1 == None:
      value1 = receive()
    while value2 == None:
      value2 = receive()
    print(f'value 1 is {value1} and value2 is {value2}')
    return get_exp(value1,value2)

def run(input,q_weights):
    value1, value2 = None, None
    for idx, angle in enumerate(input):
        ang = angle
        while ang < 0:
          ang += math.pi/2
        an = int(((ang*2)*180/math.pi))
        send_to_arduino(ser,[an,idx])
    for idx, angle in enumerate(q_weights):
        ang = angle
        while ang < 0:
          ang += math.pi/2
        an = int(((ang*2)*180/math.pi))
        send_to_arduino(ser,[an,idx+1])
    time.sleep(2)
    while value1 == None:
      value1 = receive()
    while value2 == None:
      value2 = receive()
    print(f'value 1 is {value1} and value2 is {value2}')
    return get_exp(value1,value2)

def apply_gradient(input,params):
    input,params = input.tolist(), params.tolist()
    s = np.pi/2
    gradient = []
    for k in params:
        k_plus = k + s
        k_minus = k - s
        exp_plus = run(input,[k_plus])
        exp_minus = run(input,[k_minus])
        gr = (exp_plus - exp_minus) / 2
        gradient.append(gr)
    return torch.tensor(gradient,dtype=torch.float32)



# %%
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ])
}

# %%
data_dir = 'E:\\hymenoptera_data'
image_datasets = {'train': datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                                data_transforms['train'])}
dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=1,
                                                    shuffle=True)}
dataset_sizes = {'train': len(image_datasets['train'])}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

n_qubits = 1
q_depth = 1
q_delta = 0.01

# %%
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32 * 54 * 54, 1200)
        self.fc2 = nn.Linear(1200, 120)
        self.fc3 = nn.Linear(120, 84)
        self.fc4 = nn.Linear(84, 1)
        self.q_params = torch.nn.Parameter( q_delta * torch.randn(q_depth,dtype=torch.float32))

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 54 * 54)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        q_out = None
        for elem in x:
            # print(elem)
            q_out_elem = quantum_layer(elem, self.q_params)
            if q_out == None:
                q_out = torch.tensor([[q_out_elem]])
            else:
                q_out = torch.add(q_out, torch.tensor([[q_out_elem]]))

        q_out = (q_out+1)/2
        q_out = torch.cat((q_out, 1-q_out), -1)
        q_out = q_out.requires_grad_()
        return q_out


network = Net()
optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
epochs = 10
criterion = nn.CrossEntropyLoss()
loss_list = []
# for data,j in dataloaders[‘train’]: print(data)

# %%
for epoch in range(epochs):
    total_loss = []
    target_list = []
    for data, target in dataloaders['train']:
        data = data.to(device)
        target = target.to(device)
        target_list.append(target.item())
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # update quantum parameters
        gradient = apply_gradient(output,network.q_params)
        outlist = output.tolist()[0]
        out = (1-outlist[0]) if outlist[0] > outlist[1] else outlist[1]
        new_params = nn.Parameter(network.q_params - (0.001 * (out-target.item()) * gradient))
        network.q_params = new_params
        total_loss.append(loss.item())
    loss_list.append(sum(total_loss) / len(total_loss))
    print('Loss = {:.2f} after epoch #{:2d}'.format(loss_list[-1], epoch + 1))