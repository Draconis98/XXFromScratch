import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']
        
        self.U_i = nn.Linear(self.input_size, self.hidden_size)
        self.V_i = nn.Linear(self.hidden_size, self.hidden_size)
        self.b_i = nn.Parameter(torch.zeros(self.hidden_size))
        
        self.U_f = nn.Linear(self.input_size, self.hidden_size)
        self.V_f = nn.Linear(self.hidden_size, self.hidden_size)
        self.b_f = nn.Parameter(torch.zeros(self.hidden_size))
        
        self.U_c = nn.Linear(self.input_size, self.hidden_size)
        self.V_c = nn.Linear(self.hidden_size, self.hidden_size)
        self.b_c = nn.Parameter(torch.zeros(self.hidden_size))
        
        self.U_o = nn.Linear(self.input_size, self.hidden_size)
        self.V_o = nn.Linear(self.hidden_size, self.hidden_size)
        self.b_o = nn.Parameter(torch.zeros(self.hidden_size))
        
        self.init_weights()
        
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0) 
            else:
                nn.init.xavier_uniform_(param)
        
    def forward(self, x: torch.Tensor, init_states: tuple = None) -> torch.Tensor:
        batch_size, seq_len = x.shape[0], x.shape[1]
        hidden_seq = []
        
        if init_states is None:
            h_t, c_t = torch.zeros(batch_size, self.hidden_size, device=x.device), torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h_t, c_t = init_states
            
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            i_t = torch.sigmoid(self.U_i(x_t) + self.V_i(h_t) + self.b_i)
            f_t = torch.sigmoid(self.U_f(x_t) + self.V_f(h_t) + self.b_f)
            c_t = torch.tanh(self.U_c(x_t) + self.V_c(h_t) + self.b_c)
            o_t = torch.sigmoid(self.U_o(x_t) + self.V_o(h_t) + self.b_o)
            
            c_t = f_t * c_t + i_t * c_t
            h_t = o_t * torch.tanh(c_t)
            
            hidden_seq.append(h_t.unsqueeze(0))
            
        hidden_seq = torch.cat(hidden_seq, dim=0)
        return hidden_seq, (h_t, c_t)