import torch.nn as nn
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
n_qubits = 2
q_depth = 1
q_delta = 0.1

plus = 0
def send_to_arduino(ser, values):
    ser.write(struct.pack('>BB', *values))

def receive():
    time.sleep(1)
    if ser.in_waiting > 0:
        data = ser.readline().decode('utf-8').rstrip()
        return data

def get_exp(value1,value2,value3,value4):
    states = {'00':value1+value3,'01':value1+value4,'10':value2+value3,'11':value2+value4}
    expects = np.zeros(2)
    shots = value1 + value2 + value3 + value4
    for key in states.keys():
        perc = int(states[key])/int(shots)
        check = np.array([( float(key[i]) - 1/2 )*2*perc for i in range(2)])
        expects += check
    #print(value1,value2,value3,value4)
    return expects

# -- Neural Network and the Quantum Layer -----

def quantum_layer(q_input_features, q_weights_flat):
    value1, value2, value3,value4 = None, None,None,None
    q_weights = q_weights_flat.reshape(q_depth, n_qubits)
    for idx, angle in enumerate(q_input_features):
        ang = angle.item()
        while ang < 0:
          ang += math.pi/2
        an = int(math.degrees(ang))
        #print(an)
        send_to_arduino(ser,[an,idx])

    for layer in range(q_depth):
        for idx, angle in enumerate(q_weights[layer]):
          ang = angle.item()
          while ang < 0:
            ang += math.pi/2
          an = int(math.degrees(ang))
          #print(an)
          send_to_arduino(ser,[an,idx+2])

    #time.sleep(1)
    #print('Started layer')
    while value1 == None:
        try: value1 = int(receive())
        except: pass
    while value2 == None:        
        try: value2 = int(receive())
        except: pass
    while value3 == None:        
        try: value3 = int(receive())
        except: pass
    while value4 == None:
        try: value4 = int(receive())
        except: pass
    #print('Done')
    #print(f'in the layer value 1 is {value1} and value2 is {value2}, {value3} and {value4}')
    return get_exp(value1,value2,value3,value4)

def run(q_weights):
    #print(f'weights {q_weights}')
    try: 
        if len(q_weights.tolist()) < 2: q_weights = q_weights[0]
    except: pass
    value1, value2, value3,value4 = None, None,None,None
    for idx, angle in enumerate(q_weights):
        #print('angle',angle)
        try: ang = angle.item()
        except: ang = angle
        while ang < 0:
          #print('ang',ang)
          ang += math.pi/2
        an = int(math.degrees(ang)) * 2
        #print(an)
        send_to_arduino(ser,[an,idx+2])

    #time.sleep(2)
    #print('Started run')
    while value1 == None:
        try: value1 = int(receive())
        except: pass
    while value2 == None:        
        try: value2 = int(receive())
        except: pass
    while value3 == None:        
        try: value3 = int(receive())
        except: pass
    while value4 == None:      
        try: value4 = int(receive())
        except: pass
    #print('Done')
    #print(f'value 1 is {value1} and value2 is {value2}, {value3} and {value4}')
    #print(type(value2))
    #print(get_exp(value1,value2,value3,value4) ,'getcle')
    return get_exp(value1,value2,value3,value4)
def compute_gradient(func, params,y):
    eps = 0.01  # Small constant for numerical stability
    try: params = params.tolist()
    except: pass
    plus = [i+eps for i in params]
    minus = [i-eps for i in params]
    grad = []
    f1,f2 = func(plus),func(minus)
    for i in range(len(f1)):
      grad.append( (f1[i] - f2[i]) / (2*eps) )
    return grad

# def compute_gradient(func,theta_weights,y):
#     ypred = run(theta_weights)
#     ypred = torch.tensor([ypred.tolist()],requires_grad=True)
#     #print('y',ypred,y)
#     loss = F.nll_loss(ypred, y)

#     loss.backward()

#     return ypred.grad

def update(params, learning_rate,func,y):
    grad = compute_gradient(func, params,y)
    #print( grad )
    params = params.tolist()
    #print('jijij',params,np.multiply(learning_rate , grad))
    updated_params = params - np.multiply(learning_rate , grad)
    return updated_params


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


# %%
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32 * 54 * 54, 1200)
        self.fc2 = nn.Linear(1200, 120)
        self.fc3 = nn.Linear(120, 84)
        self.fc4 = nn.Linear(84, 2)
        self.q_params = torch.nn.Parameter( q_delta * torch.randn(q_depth*n_qubits,dtype=torch.float32))

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 54 * 54)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        #print('this is x',x)
        q_out = None
        for elem in x:
            # print(elem)
            q_out_elem = quantum_layer(elem, self.q_params)
            if q_out == None:
                q_out = torch.tensor([q_out_elem])
            else:
                q_out = torch.cat((q_out, torch.tensor([q_out_elem],dtype=torch.float64)),dim=0)

        #q_out = (q_out+1)/2
        #q_out = torch.cat((q_out, 1-q_out), -1)
        q_out = q_out.requires_grad_()
        #print(q_out,'qout')
        return q_out


network = Net()
lr = 0.1
optimizer = optim.SGD(network.parameters(), lr=lr, momentum=0.9)
epochs = 15
loss_func = torch.nn.L1Loss()
loss_list = []
# for data,j in dataloaders[‘train’]: print(data)

# %%
for epoch in range(25):
    total_loss = []
    target_list = []
    for data, target in dataloaders['train']:
        data = data.to(device)
        target = target.to(device)
        tr = None
        for i in target:
            if tr == None:
                if i.item() == 1:tr = torch.tensor([[1,0]],dtype=torch.float32)
                if i.item() == 0:tr = torch.tensor([[0,1]],dtype=torch.float32)
            else: 
                if i.item() == 1:tr = torch.cat((tr,torch.tensor([[1,0]],dtype=torch.float32)),dim=0)
                if i.item() == 0:tr = torch.cat((tr,torch.tensor([[0,1]],dtype=torch.float32)),dim=0)
        optimizer.zero_grad()
        output = network(data)
        #print(output,tr,'poo')
        loss = loss_func(output, tr)
        #print(loss)
        loss.backward()
        optimizer.step()

        # update quantum parameters
        new_params = update(network.q_params,lr,run,target)
        new_params = nn.Parameter(new_params)
        network.q_params = new_params
        
        total_loss.append(loss.item())
    loss_list.append(sum(total_loss) / len(total_loss))
    print('Loss = {:.4f} after epoch #{:2d}'.format(loss_list[epoch], epoch + 1))


data_dir = 'E:\\hymenoptera_data'
image_datasets = {'test': datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                                data_transforms['train'])}
testloaders = {'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=1,
                                                   shuffle=True)}
dataset_sizes = {'test': len(image_datasets['test'])}
class_names = image_datasets['test'].classes
def test():
  network.eval()
  correct = 0
  with torch.no_grad():
    for data, target in testloaders['test']:
      output = CaQNN(data)
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()

  print(f"Accuracy: {correct} / {len(testloaders['test'].dataset)} = {100 * correct/len(testloaders['test'].dataset)}%")
  
test()
