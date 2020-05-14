import torch

def bone_constraint_loss(ypred, ytrue, device):
    """ 
    bone constraint loss function 
     
    input:
        ypred = predicted 3D hand skeletonization - torch.Size([batch_size x 21 x 3])
        ytrue = true 3D hand skeletonization - torch.Size([batch_size x 21 x 3])
         
    output:
        L_len = bone length loss - torch.Size([1])
        L_dir = bone direction loss  - torch.Size([1])
    """
    
    batch_size, num_joint, num_dim = ypred.shape    
    
    num_bone = num_joint-1
        
    i = torch.tensor([0 if i%4==0 else i for i in range(num_bone)])
    j = torch.arange(1,num_bone+1)
        
    b_pred = ypred[:,i,:] - ypred[:,j,:]      # [batch_size x 20 x 3]
    b_true = ytrue[:,i,:] - ytrue[:,j,:]      # [batch_size x 20 x 3]
    
    L_len = torch.abs(torch.norm(b_true, dim=2) - torch.norm(b_pred, dim=2))     # [batch_size x 20]
    
    norm_b_true = b_true.view(num_dim, batch_size, num_bone)/torch.norm(b_true, dim=2)  # [3 x batch_size x 20]
    norm_b_pred = b_pred.view(num_dim, batch_size, num_bone)/torch.norm(b_pred, dim=2)  # [3 x batch_size x 20]
    L_dir = torch.norm(norm_b_true - norm_b_pred, dim=0)
    
    return torch.sum(L_len), torch.sum(L_dir)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    