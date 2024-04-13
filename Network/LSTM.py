from torch import nn, stack, float16, float64, float32
import torch
from torch.autograd import Variable
class LSTMModel(nn.Module) :
   def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bias=True) :
       super(LSTMModel, self).__init__()
       self.hidden_dim = hidden_dim
       self.input_dim = input_dim
       self.layer_dim = layer_dim
       self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=layer_dim)
       self.fc = nn.Linear(hidden_dim, output_dim)
       
   def forward(self, x) :
       if torch.cuda.is_available() :
           h0 = Variable(torch.zeros(self.layer_dim, self.hidden_dim, dtype=float64).cuda())
       else :
           h0 = Variable(torch.zeros(self.layer_dim, self.hidden_dim, dtype=float64))
           
       if torch.cuda.is_available() :
           c0 = Variable(torch.zeros(self.layer_dim, self.hidden_dim, dtype=float64).cuda())
       else :
           c0 = Variable(torch.zeros(self.layer_dim, self.hidden_dim, dtype=float64))
       
       output, (final_hidden_state, final_cell_state) = self.lstm(x, (h0, c0))
       out = self.fc(output)
       return out
