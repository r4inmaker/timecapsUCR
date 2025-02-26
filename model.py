import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Cell_A(nn.Module):
    
    def __init__(self, cp, ap, g2, cSA, aSA, g3):
        super(Cell_A, self).__init__()
        
        self.cp,  self.ap  = cp,  ap
        self.cSA, self.aSA = cSA, aSA
        
        self.psi_m = nn.Conv1d(
            in_channels = self.cp * self.ap,
            out_channels = self.cp * self.ap,
            kernel_size = g2,
            stride = 1,
            padding = 'same',
            bias = False
        ).double()
        
        assert g3 % 2 != 0, '[g3] needs to be odd'
        pad_A = int( g3/2 - 0.5 )
        
        self.psi_m_A = nn.Conv2d(
            in_channels = 1,
            out_channels = self.cSA * self.aSA,
            kernel_size = (self.ap, g3),
            stride = (self.ap, 1),
            padding = (0, pad_A),
            bias = True
        ).double()
        
    def forward(self, X):
        N, C, L = X.shape
        assert C == self.cp * self.ap, '[cp * ap] must equal [K]'
        
        # Φ_A_Conv
        X = self.psi_m(X) 
        
        # Ω_A_PTC
        sq_norm = (X**2).sum(-1).unsqueeze(-1)
        X = ((sq_norm / (1 + sq_norm)) * X / (sq_norm + 1e-8)).unsqueeze(1)
        
        # W_A_Conv
        X = self.psi_m_A(X)
        
        return X
    
    
class Cell_B(nn.Module):
    
    def __init__(self, cb, ab, ins, outs, Ln, n, cSB, aSB, gB):
        super(Cell_B, self).__init__()
        
        self.cb, self.ab = cb, ab
        self.Ln, self.n = Ln, n
        
        assert gB % 2 != 0, '[gB] needs to be odd'
        pad_B = int( 5/2 - 0.5 )
        
        self.Conv_B = nn.Conv1d(
            in_channels = ins,
            out_channels = outs,
            kernel_size = 1
        ).double()
        
        self.psi_k_B = nn.Conv2d(
            in_channels = 1,
            out_channels = cSB * aSB,
            kernel_size = (cb*ab, gB),
            stride = (cb*ab, 1),
            padding = (0, pad_B)
        ).double()
    
        
    def forward(self, X):
        
        # Φ_B_Conv
        X = self.Conv_B(X)
        
        # splicing accross spatial dimension
        X = X.reshape(-1, self.cb*self.ab, self.n, self.Ln).permute(0,2,1,3).reshape(-1, self.cb*self.ab*self.n, self.Ln).unsqueeze(1)
        
        # W_b_conv
        X = self.psi_k_B(X)
        
        # squash
        sq_norm = (X**2).sum(-1).unsqueeze(-1)
        X = ((sq_norm / (1 + sq_norm)) * X / (sq_norm + 1e-8))
        
        return X


class Routing(nn.Module):
    def __init__(self, device, n_iters=3):
        super(Routing, self).__init__()
        
        self.n_iters = n_iters
        self.device = device
        
    def forward(self, X):
        
        # einsum tryhard
        N, C, n_caps, L = X.shape
        b = torch.zeros(N, C, n_caps, 1).to(self.device)
      
        for _ in range(self.n_iters - 1):
            
            c = b.softmax(-2)
            
            sj = (c * X).sum(-2).unsqueeze(-2)
            
            # squash
            sq_norm = (sj**2).sum(-1).unsqueeze(-1)
            vj = (sq_norm / (sq_norm + 1)) * (sj / (sq_norm + 1e-8))
            
            # update b
            bj = torch.einsum('abij,abjk->abik', X, vj.permute(0,1,3,2))
            b += bj
        
        return vj.squeeze(-1)
    
    
    
class DigitCaps(nn.Module):
    def __init__(self, batch_size, input_dim, output_dim, n_classes, n_iters, device):
        super(DigitCaps, self).__init__()
        
        self.W = nn.Parameter(torch.randn(n_classes, input_dim, output_dim, dtype=torch.float64))
        self.n_iters = n_iters
        
        self.routing = Routing(device=device)
        
    def forward(self, X):
        
        u_jis = torch.stack([xi @ self.W for xi in torch.unbind(X)])
        v_js = self.routing(u_jis)
        
        return v_js.squeeze(-2)  
    

class TimeCaps(nn.Module):
    
    def __init__(self, L, K, g1, cp, ap, g2, n, cSA, aSA, g3, cb, ab,
                 cSB, aSB, gB, n_classes, batch_size, device):
        
        super(TimeCaps, self).__init__()

        assert L % n == 0, '[L] must be divisible by [n]'
        

        self.K, self.g1 = K, g1
        self.cp, self.ap, self.g2 = cp, ap, g2
        self.cSA, self.aSA, self.g3 = cSA, aSA, g3
        self.cb, self.ab = cb, ab
        
        self.L = L
        self.Ln = L // n 
        self.n = n
        self.cSB, self.aSB = cSB, aSB
        self.gB = gB
        
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        
        self.device = device
        
        self.conv1 = nn.Conv1d(
            in_channels = 1,
            out_channels = K,
            kernel_size = g1,
            stride = 1,
            padding = 'same',
            bias = False
        ).double()
        
        self.cell_A = Cell_A(
            cp=self.cp, ap=self.ap, g2=self.g2, cSA=self.cSA, aSA=self.aSA, g3=self.g3
        )
        
        self.cell_B = Cell_B(
            cb=self.cb, ab=self.ab, ins=K, outs=cb*ab, Ln=self.Ln, n=self.n, cSB=self.cSB, aSB=self.aSB, gB=self.gB
        )
        
        # torch.Size([64, 64, 8, 1]) torch.Size([64, 64, 8, 360])
        self.routing_A = Routing(device=self.device)
        
        # torch.Size([64, 8, 10, 1]) torch.Size([64, 8, 10, 36])
        self.routing_B = Routing(device=self.device)
        
        self.digit_caps = DigitCaps(batch_size=batch_size, input_dim=self.cSA*self.L+self.Ln, output_dim=32, n_classes=n_classes,
                                    n_iters=3, device=self.device)

        
        
    def swap_devices(self, swap_to):
        self.routing_A.device = swap_to
        self.routing_B.device = swap_to
        self.digit_caps.routing.device = swap_to
        
        
    def forward(self, X):
        
        # Φ_Conv1
        X = self.conv1(X)
    
        # Temporal Capsules [A]
        X_A = self.cell_A(X)
        
        # Routing [A]
        X_A = self.routing_A(X_A).reshape(-1, self.cSA, self.aSA * self.L)
        
        # Spacial Capsules [B]
        X_B = self.cell_B(X)
        
        # Routing [B]
        X_B = self.routing_B(X_B).squeeze(-2)
        
        # Concat
        X_A *= self.alpha
        X_B *= self.beta
        X_cat = torch.cat((X_A, X_B), -1)
        
        # Digit Caps
        X_digit = self.digit_caps(X_cat)
    
        return X_digit



class Decoder(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(Decoder, self).__init__()
        
        self.lin1 = nn.Linear(in_features=input_dim, out_features=2048, bias=True).double()
        self.lin2 = nn.Linear(in_features=2048, out_features=1024, bias=True).double()
        self.lin3 = nn.Linear(in_features=1024, out_features=512, bias=True).double()
        self.lin4 = nn.Linear(in_features=512, out_features=output_dim, bias=True).double()

    
    def forward(self, x):
        
        x = self.lin1(x)
        x = torch.tanh(x)
        x = self.lin2(x)
        x = torch.tanh(x)
        x = self.lin3(x)
        x = torch.tanh(x)
        x = self.lin4(x)
        
        return x
        

class MarginLoss(nn.Module):
    def __init__(self, m_plus=0.9, m_minus=0.1, _lambda=0.5, n_classes=5):
        
        super(MarginLoss, self).__init__()
        self.m_plus = m_plus
        self.m_minus = m_minus
        self._lambda = _lambda
        self.n_classes = n_classes
        
    def forward(self, X, y):
        
        y_hot = F.one_hot(y, num_classes=self.n_classes)
        X_lengths = torch.sqrt((X**2).sum(-1))
        Lk = y_hot * torch.relu(self.m_plus - X_lengths)**2  +  self._lambda * (1 - y_hot) * torch.relu(X_lengths - self.m_minus)**2
        Lk = Lk.sum(-1)
        return Lk.sum()