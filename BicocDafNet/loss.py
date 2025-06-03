import torch
import torch.nn as nn
import torch.nn.functional as F
    
class OCSoftmax(nn.Module):
    def __init__(self, r_real=0.5, r_fake=0.2, alpha=20.0):
        super(OCSoftmax, self).__init__()
        self.r_real = r_real
        self.r_fake = r_fake
        self.alpha = alpha
        self.softplus = nn.Softplus()

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        x = x.squeeze()        
        x[labels == 0] = self.r_real - x[labels == 0]
        x[labels == 1] = x[labels == 1] - self.r_fake   
        loss = self.softplus(self.alpha * x).mean()
        return loss

class TOCSoftmax(nn.Module):
    def __init__(self, r_real=0.5, r_fake=0.2, alpha=20.0):
        super(TOCSoftmax, self).__init__()
        self.r_real = r_real
        self.r_fake = r_fake
        self.alpha = alpha
        self.softplus = nn.Softplus()

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        x = x.squeeze()
        weight = torch.logical_not((labels == 1) & (x < self.r_fake)).float()
        x[labels == 0] = self.r_real - x[labels == 0]
        x[labels == 1] = x[labels == 1] - self.r_fake
        loss = (weight*self.softplus(self.alpha * x)).mean()
        return loss
    
class DC_OCSoftmax(nn.Module):
    def __init__(self,d_args, r_real=0.5, r_fake=0.2, alpha=20.0,a=0.1):
        super(DC_OCSoftmax, self).__init__()
        self.r_real = r_real
        self.r_fake = r_fake
        self.alpha = alpha
        self.softplus = nn.Softplus()
        self.a = a
        self.lamuda1=d_args['lamuda1']
        self.lamuda2=d_args['lamuda2']

    def forward(self, x, labels,xi):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        x = x.squeeze()        
        x[labels == 0] = self.r_real - x[labels == 0]
        x[labels == 1] = x[labels == 1] - self.r_fake 
        Ex1 = xi[labels == 1] #jia
        Ex0 = xi[labels == 0] #zhen
        
        if len(Ex1)>1:
            C2=Ex1.mean(dim=0)
            LC2 = 1/(torch.sum(torch.norm((xi[labels == 1] - C2), p=2) ** 2, dim=0)/2+self.a)
        else:
            LC2 = 0
            
        
        
        C1=Ex0.mean(dim=0)
        #LC2 = 1/(torch.mean(torch.sum(torch.norm((xi[labels == 1] - C2), p=2) ** 2, dim=0))/2+self.a)
        LC1 = -(torch.mean(F.cosine_similarity(xi[labels == 0], C1.unsqueeze(0), dim=1)))

        loss = self.softplus(self.alpha * x).mean() + self.lamuda1*LC1 +self.lamuda2*LC2
        return loss    