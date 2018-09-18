require 'nn'

-- Implementation of sparsemax loss criterion from:
-- "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification"
-- André F. T. Martins, Ramón Fernandez Astudillo (http://arxiv.org/abs/1602.02068)

local SparseMaxLoss,parent = torch.class('nn.SparseMaxLoss', 'nn.Module')

function SparseMaxLoss:__init(sizeAverage)
  parent:__init(self)
  self.taus = torch.Tensor()
  self.sparsemax = torch.Tensor()
  if sizeAverage ~= nil then
    self.sizeAverage = sizeAverage
  else
    self.sizeAverage = true
  end
  self.zs = torch.Tensor()
  self.bound = torch.Tensor()
  self.is_gt = torch.Tensor()
  self.zs_sparse = torch.Tensor()
  self.zs_squared = torch.Tensor()
  self.taus_squared = torch.Tensor()
  self.regTerm = torch.Tensor()
  self.range = torch.Tensor()
  self.cumsum_zs = torch.Tensor()
--  self.index = torch.CudaLongTensor()
end

function SparseMaxLoss:updateOutput(input, target)
  local dim = 1
  -- local target = target
  if input:nDimension() == 2 then
    dim = 2
    -- target = target:view(-1,1)
  elseif input:nDimension() ~= 1 then
    error('vector or matrix epected')
  end
  local sizeDim = input:size()[dim]
self.index = torch.CudaLongTensor()
  -- Sort input in descending order
  torch.sort(self.zs, self.index, input, dim, true)

  self.range = torch.range(1, sizeDim):typeAs(input)
  if dim == 2 then
    self.range = self.range:view(1, sizeDim)
  end
  self.range = self.range:expandAs(self.zs)
  -- Determine sparsity of projection
  torch.cmul(self.bound, self.range, self.zs)
  torch.add(self.bound, self.bound, 1)
  torch.cumsum(self.cumsum_zs, self.zs, dim)
  self.is_gt = torch.gt(self.bound, self.cumsum_zs):typeAs(self.range)
  local k = torch.max(torch.cmul(self.range, self.is_gt), dim)
  -- Compute threshold function
  torch.cmul(self.zs_sparse, self.is_gt, self.zs)
  torch.cdiv(self.taus, torch.sum(self.zs_sparse, dim) - 1, k)
  -- Sparsemax loss
  torch.cmul(self.zs_squared, self.zs_sparse, self.zs_sparse)
  torch.cmul(self.taus_squared, self.is_gt, torch.cmul(self.taus, self.taus):expandAs(self.zs_sparse))
  torch.sum(self.regTerm, self.zs_squared - self.taus_squared, dim)
  self.regTerm = self.regTerm:expandAs(input)
  torch.mul(self.regTerm, self.regTerm, 0.5)
  torch.add(self.regTerm, self.regTerm, 0.5)
  self.output = input -  self.regTerm
  return self.output
end

function SparseMaxLoss:updateGradInput(input, gradOutput)
  local dim = 1
  -- local target = target
  if input:nDimension() == 2 then
    dim = 2
    -- target = target:view(-1,1)
  elseif input:nDimension() ~= 1 then
    error('vector or matrix epected')
  end
  torch.cmax(self.sparsemax,
    torch.zeros(input:size()):typeAs(input),
    input - self.taus:expandAs(input)
  )

  self.gradInput = gradOutput - torch.cmul(self.sparsemax,gradOutput:sum(dim):expandAs(self.sparsemax))
  return self.gradInput
end
