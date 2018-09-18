require 'nn'
-- Combination of SparseMaxLoss and ClassNLLCriterion

local SparseMaxCriterion, Criterion = torch.class('nn.SparseMaxCriterion', 'nn.Criterion')

function SparseMaxCriterion:__init(weights)
  Criterion.__init(self)
  self.sml = nn.SparseMaxLoss()
  self.nll = nn.ClassNLLCriterion(weights)
end

function SparseMaxCriterion:updateOutput(input, target)
  input = input:squeeze()
  target = type(target) == 'number' and target or target:squeeze()
  self.sml:updateOutput(input)
  self.nll:updateOutput(self.sml.output, target)
  self.output = self.nll.output
  return self.output
end

function SparseMaxCriterion:updateGradInput(input, target)
  local size = input:size()
  input = input:squeeze()
  target = type(target) == 'number' and target or target:squeeze()
  self.nll:updateGradInput(self.sml.output, target)
  self.sml:updateGradInput(input, self.nll.gradInput)
  self.gradInput:view(self.sml.gradInput, size)
  return self.gradInput
end

return nn.SparseMaxCriterion
