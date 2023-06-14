import torch

def pixel_loss(y_true, y_pred, eps):
    device = y_pred.device
    loss = torch.FloatTensor(torch.sum(torch.sqrt((y_true - y_pred).pow(2)+ eps**2))).to(device)
    return loss
