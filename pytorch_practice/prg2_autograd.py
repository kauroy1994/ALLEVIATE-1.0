import torch

#x = torch.tensor([2],dtype=torch.float32,requires_grad=True)
#y = x*x
#z = y*x
#z = z.mean() #without this it is d(vector)/d(vector)
#z.backward() #dz/dx
#print (x.grad)
#set x.requires_grad_(False) after training complete
#y = x.detach() new tensor with the same values with no grad
#if you want to do some compute with x
'''
with torch.no_grad():
    y = x + 2
    print (y)
'''
#x.grad.zero_() reset gradient to 0 after each epoch, or it will sum
'''if we use optimizer
optimizer = torch.optim.SGD(x,lr=0.01)
optimizer.step() ,one optimization step
optimizer.zero_grad(), reset to 0 after each optimization step
