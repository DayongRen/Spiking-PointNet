import torch.nn as nn
from models.spike_layer_without_MPR import SpikeConv, LIFAct, SpikeLinear, SpikeModule, SpikeBatchNorm

class SpikeModel(SpikeModule):

    def __init__(self, model: nn.Module, step=2, temp=3.0):
        super().__init__()
        self.model = model
        self.step = step
        self.temp =temp
        self.spike_module_refactor(self.model, step=step, temp=temp)


    def spike_module_refactor(self, module: nn.Module, step=2, temp=3.0):
        """
        Recursively replace the normal conv1d and Linear layer to SpikeLayer
        """
        for name, child_module in module.named_children():

            if isinstance(child_module, nn.Conv1d):
                setattr(module, name, SpikeConv(child_module, step=step))

            elif isinstance(child_module, nn.Linear):
                setattr(module, name, SpikeLinear(child_module, step=step))

            elif isinstance(child_module, nn.ReLU):
                setattr(module, name, LIFAct(step=step, temp=temp))

            elif isinstance(child_module, nn.BatchNorm1d):
                setattr(module, name, SpikeBatchNorm(child_module, step=step))
            
            else:
                self.spike_module_refactor(child_module, step=step, temp=temp)

    def forward(self, input):
        if input.dim() == 3:
            input = input.repeat(self.step, 1, 1, 1)
        else:
            print('input is error')
        
        out, trans_feat = self.model(input)

        if len(out.shape) == 3:
            out = out.mean([0])
            trans_feat = trans_feat.mean([0])
        else:
            print('output is error')

        return out, trans_feat

    def set_spike_state(self, use_spike=True):
        self._spiking = use_spike
        for m in self.model.modules():
            if isinstance(m, SpikeModule):
                m.set_spike_state(use_spike)

    def set_spike_before(self, name):
        self.set_spike_state(False)
        for n, m in self.model.named_modules():
            if isinstance(m, SpikeModule):
                m.set_spike_state(True)
            if name == n:
                break
