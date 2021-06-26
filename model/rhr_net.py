import torch
import numpy as np
import yaml, logging, math

# import torchaudio

class RHRNet(torch.nn.Module):
    def __init__(self):
        super(RHRNet, self).__init__()
        # self.hp = hyper_parameters
        self.GRU_Size = [2, 128, 256, 512, 256, 128]
        self.Step_Ratio = [0.5, 0.5, 0.5, 2.0, 2.0, 2.0]
        self.Residual = [None, None, None, None, 2, 1]
        self.layer_Dict = torch.nn.ModuleDict()

        previous_Size = 1
        for index, (size, ratio, residual) in enumerate(zip(self.GRU_Size, self.Step_Ratio, self.Residual)):
            self.layer_Dict['GRU_{}'.format(index)] = GRU(
                input_size= previous_Size,
                hidden_size= size,
                num_layers= 1,
                batch_first= True,
                bidirectional= True
                )
            if not residual is None:
                self.layer_Dict['PReLU_{}'.format(index)] = torch.nn.PReLU()

            previous_Size = int(size * 2 / ratio)

        self.layer_Dict['Last'] = GRU(
            input_size= previous_Size,
            hidden_size= 1,
            num_layers= 1,
            batch_first= True,
            bidirectional= False
            )

    def forward(self, x):
        '''
        x: [Batch, Time]
        '''
        # [Batch, 1, Time]
        # print("==before unsqueeze==")
        # print(x.shape)
        x = x.squeeze(1) # [Batch, Time]
        # print("==before unsqueeze==")
        # print(x.shape)
        x = x.unsqueeze(2)
        # x = x.unsqueeze(2)   # [Batch, Time, 1]
        # print("==after unsqueeze==")
        # print(x.shape)

        stacks = []
        for index, (ratio, residual) in enumerate(zip(self.Step_Ratio, self.Residual)):
            # print("==before reshape==")
            # print(x.shape)
            if not residual is None:
                x = self.layer_Dict['PReLU_{}'.format(index)](x + stacks[residual])
            # print("==after residual==")
            # print(x.shape)
            x = self.layer_Dict['GRU_{}'.format(index)](x)[0]
            stacks.append(x)
            # print("==after stacks.append==")
            # print(x.shape)
            x = x.reshape(x.size(0), torch.tensor(x.size(1) * ratio).long(), torch.tensor(x.size(2) / ratio).long())    # I am not sure why sometime x.size() is tensor or just int when JIT.
            # print("==after reshape==")
            # print(x.shape)
        x = self.layer_Dict['Last'](x)[0]
        # print("==after self.layer_Dict==")
        # print(x.shape)
        return x.squeeze(2)

class GRU(torch.nn.GRU):
    def reset_parameters(self):
        for name, parameter in self.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(parameter)
            elif 'bias' in name:
                torch.nn.init.zeros_(parameter)

# https://github.com/tuantle/regression-losses-pytorch
class Log_Cosh_Loss(torch.nn.Module):
    def forward(self, logits, labels):
        return torch.mean(torch.log(torch.cosh(labels - logits)))

if __name__ == "__main__":
    import yaml
    from Arg_Parser import Recursive_Parse
    hp = Recursive_Parse(yaml.load(
        open('Hyper_Parameters.yaml', encoding='utf-8'),
        Loader=yaml.Loader
        ))  
    net = RHRNet()
    net(torch.randn(3, 1, 1024))