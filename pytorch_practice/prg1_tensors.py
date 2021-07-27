import torch
#x = torch.rand(1)
#x = torch.zeros(1)
#x = torch.rand(2,3) [i][j] access is possible in tensors
#x = torch.tensor([2.5,0.1])
#x = torch.rand(1,dtype=torch.float32)
#print (x.size())
#z = x + y element wise addition
#z = torch.add(x,y) same as above
#y.add_(x) add all elements of x to y, underscore is inplace operation
#c = x - y, torch.sub(x,y), y.mul/div_(x) [element wise]
#print (x[:,0]) all rows but only column 0
#print (x[1,:]) row number 1 but all columns
#print (x[1,1]) also a way to index
#x[1,1].item() will get actual value not create a tensor
#y = x.view(4) if x = 2,2 (2*2) tensor reshapes to a 1D tensor of 4 elements
#y = x.view(-1,1) would determine the row dimension based on column and total
#y = x.numpy() transforms tensor to a numpy array, y and x same pointer
#y = torch.from_numpy(x) transformers numpy array to torch tensor,same pointers
#cuda stuff time 15 onwards
#x = torch.ones(5, requires_grad = True) to enable taking gradient later
