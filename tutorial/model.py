import json
import sys

import torch
import torch.nn as nn

torch.set_printoptions(threshold=sys.maxsize)


class OurEncoder(json.JSONEncoder):
    def default(self, obj):
        # print(type(obj))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if isinstance(obj, torch.cuda.FloatTensor):
            # print(obj)
            return obj.cpu().numpy().tolist()
        return super(OurEncoder, self).default(obj)


# 全连接网络
class CLCPPredictor(nn.Module):
    def __init__(self):
        super(CLCPPredictor, self).__init__()

        self.dropout = nn.Dropout(p=1 / 512)

        self.hidden1 = nn.Sequential(
            nn.Linear(in_features=18,
                      out_features=64,  # 30
                      bias=True),
            nn.Sigmoid())
        self.hidden2 = nn.Sequential(
            nn.Linear(in_features=64,  # 30
                      out_features=128,
                      bias=True),
            nn.Sigmoid())
        self.hidden3 = nn.Sequential(
            nn.Linear(in_features=128,
                      out_features=128,
                      bias=True),
            nn.Sigmoid())
        self.hidden4 = nn.Sequential(
            nn.Linear(in_features=128,
                      out_features=32,
                      bias=True),
            nn.Sigmoid())
        self.hidden5 = nn.Sequential(
            nn.Linear(in_features=32,
                      out_features=8,
                      bias=True),
            nn.Tanh())
        self.hidden6 = nn.Sequential(
            nn.Linear(in_features=8,
                      out_features=1,
                      bias=True))

    def forward(self, x):
        fc1 = self.hidden1(x)
        fc1 = self.dropout(fc1)

        fc2 = self.hidden2(fc1)
        fc2 = self.dropout(fc2)

        fc3 = self.hidden3(fc2)
        fc3 = self.dropout(fc3)

        fc4 = self.hidden4(fc3)
        fc4 = self.dropout(fc4)

        fc5 = self.hidden5(fc4)
        fc5 = self.dropout(fc5)

        output = self.hidden6(fc5)

        return output

    def save(self, filename, std_before, mean_before, std, mean):
        data = {'w1': self.state_dict()['hidden1.0.weight'],
                'w2': self.state_dict()['hidden2.0.weight'],
                'w3': self.state_dict()['hidden3.0.weight'],
                'w4': self.state_dict()['hidden4.0.weight'],
                'w5': self.state_dict()['hidden5.0.weight'],
                'w6': self.state_dict()['hidden6.0.weight'],
                'b1': self.state_dict()['hidden1.0.bias'],
                'b2': self.state_dict()['hidden2.0.bias'],
                'b3': self.state_dict()['hidden3.0.bias'],
                'b4': self.state_dict()['hidden4.0.bias'],
                'b5': self.state_dict()['hidden5.0.bias'],
                'b6': self.state_dict()['hidden6.0.bias'],
                'activation': ['Sigmoid', 'Sigmoid', 'Sigmoid', 'Sigmoid', 'Tanh'],
                'std_before': std_before,
                'mean_before': mean_before,
                'std': std,
                'mean': mean
                }

        with open(filename, 'w') as f:
            json.dump(data, f, cls=OurEncoder)
