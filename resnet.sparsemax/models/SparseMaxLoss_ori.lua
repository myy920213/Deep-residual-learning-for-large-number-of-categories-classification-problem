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
  --self.output:resizeAs(input)

  -- Sort input in descending order
  local zs = torch.sort(input, dim, true)

  local range = torch.range(1, sizeDim):typeAs(input)
  if dim == 2 then
    range = range:view(1, sizeDim)
  end
  range = range:expandAs(zs)

  -- Determine sparsity of projection
  local bound = 1 + torch.cmul(range, zs)
  local cumsum_zs = torch.cumsum(zs, dim)
  local is_gt = torch.gt(bound, cumsum_zs):typeAs(range)
  local k = torch.max(torch.cmul(range, is_gt), dim)

  -- Compute threshold function
  local zs_sparse = torch.cmul(is_gt, zs)
  self.taus = torch.cdiv(torch.sum(zs_sparse, dim) - 1, k)

  -- Sparsemax loss
  local zs_squared = torch.cmul(zs_sparse, zs_sparse)
  local taus_squared = torch.cmul(self.taus, self.taus):expandAs(zs_sparse)
  taus_squared = torch.cmul(is_gt, taus_squared)
  local regTerm = (torch.sum(zs_squared - taus_squared, dim) * 0.5 + 0.5):expandAs(input)
  self.output = input - regTerm
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
  self.sparsemax = torch.cmax(
    torch.zeros(input:size()):typeAs(input),
    input - self.taus:expandAs(input)
  )

  self.gradInput = gradOutput - torch.cmul(self.sparsemax,gradOutput:sum(dim):expandAs(self.sparsemax))
  return self.gradInput
end
