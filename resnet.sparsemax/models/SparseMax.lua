require 'nn'

-- Implementation of Sparsemax function from:
-- "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification"
-- André F. T. Martins, Ramón Fernandez Astudillo (http://arxiv.org/abs/1602.02068)

local SparseMax = torch.class('nn.SparseMax', 'nn.Module')

function SparseMax:updateOutput(input)
  local dim = 1
  local inputDim = input:nDimension()
  if inputDim == 2 or inputDim == 4 then -- match functionality of nn.SoftMax
    dim = 2
  elseif input:nDimension() > 4 then
    error('1D, 2D, 3D or 4D tensor expected')
  end
  local sizeDim = input:size()[dim]

  -- Translate input by max for numerical stability
  self.input = input - torch.max(input, dim):expandAs(input)

  -- Sort input in descending order.
  -- (NOTE: Can be replaced with linear time selection method described here:
  --  http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
  self.zs = torch.CudaTensor()
  self.index = torch.CudaLongTensor()
  self.bound = torch.CudaTensor()
  self.is_gt = torch.CudaTensor()
  self.zs_sparse = torch.CudaTensor()
  self.zs_squared = torch.CudaTensor()
  self.taus_squared = torch.CudaTensor()
  self.regTerm = torch.CudaTensor()
  self.range = torch.CudaTensor()
  self.cumsum_zs = torch.CudaTensor()
  self.taus = torch.CudaTensor()

  torch.sort(self.zs, self.index, self.input, dim, true)

  self.range = torch.range(1, sizeDim):typeAs(self.input)
  self.rangeViewMask = self.zs:size():fill(1)
  self.rangeViewMask[dim] = -1
  self.range = self.range:view(self.rangeViewMask):expandAs(self.zs)

  -- Determine sparsity of projection
  torch.cmul(self.bound, self.range, self.zs)
  torch.add(self.bound, self.bound, 1)
  torch.cumsum(self.cumsum_zs, self.zs, dim)
  self.is_gt = torch.gt(self.bound, self.cumsum_zs):typeAs(self.range)
  local k = torch.max(torch.cmul(self.range, self.is_gt), dim)

  -- Compute threshold function
  torch.cmul(self.zs_sparse, self.is_gt, self.zs)
  torch.cdiv(self.taus, torch.sum(self.zs_sparse, dim) - 1, k)

  -- Sparsemax
  torch.cmax(self.output,
    torch.zeros(self.input:size()):typeAs(self.input),
    self.input - self.taus:expandAs(self.input)
  )
  return self.output
end

function SparseMax:updateGradInput(input, gradOutput)
  local dim = 1
  local inputDim = input:nDimension()
  if inputDim == 2 or inputDim == 4 then
    dim = 2
  elseif input:nDimension() > 4 then
    error('1D, 2D, 3D or 4D tensor expected')
  end

  self.nonzeros = torch.ne(self.output, 0):typeAs(self.output)
  local sum = torch.sum(torch.cmul(gradOutput, self.nonzeros), dim) / torch.sum(self.nonzeros)
  self.gradInput = torch.cmul(self.nonzeros, gradOutput - sum:expandAs(gradOutput))
  return self.gradInput
end
