require 'torch'
require 'nn'
require 'csvigo'
require 'optim'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Forest Cover type prediction Training')
cmd:text()
cmd:text('Options:')
cmd:option('-save', 'save-log', 'subdirectory to save/log experiments in')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-model', 'convnet', 'type of model to train: convnet | mlp | linear')
cmd:option('-full', false, 'use full dataset (50,000 samples)')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
cmd:option('-learningRate', 1e-5, 'learning rate at t=0')
cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 1e-1, 'weight decay (SGD only)')
cmd:option('-momentum', 1e-3, 'momentum (SGD only)')
cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
cmd:option('-maxIter', 5, 'maximum nb of iterations for CG and LBFGS')
cmd:option('-threads', 2, 'nb of threads to use')
cmd:text()
opt = cmd:parse(arg)

torch.manualSeed(opt.seed)

torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. opt.threads)

all = csvigo.load{path='train.csv',mode='raw'}

ninputs = 54
nhidden = 108
nhidden_2 = 10
weight_decay =  1e-6

trsize = 14200
tesize = #all - trsize

print('Training data size is ' .. trsize)
print('Test data size is ' .. tesize)
print ('Preparing data..')

labels_tensor = torch.Tensor(#all-1)

for i=2,#all do
 labels_tensor[i-1] = all[i][56]
end

trainData_tensor = torch.Tensor(trsize,54)
testData_tensor = torch.Tensor(tesize,54)
labels_test_tensor = torch.Tensor(tesize)
labels_train_tensor = torch.Tensor(trsize)

for i=2,trsize do
 for j=2,55 do
   trainData_tensor[i-1][j-1] = all[i][j]
 end
end

ite = 1

for i=trsize+1,#all do
 for j=2,55 do
   testData_tensor[ite][j-1] = all[i][j]
 end 
  ite = ite + 1
end

for i=1,trsize do
 labels_train_tensor[i] = labels_tensor[i]
end

ite = 1

for i=trsize+1,#all-1 do
 labels_test_tensor[ite] = labels_tensor[i]
 ite = ite + 1
end 

classes = {'Spruce/Fir','Lodgepole Pine','Ponderosa Pine','Cottonwood/Willow','Aspen','Douglas-fir','Krummholz'}

model = nn.Sequential()
model:add(nn.Reshape(ninputs))
model:add(nn.Linear(ninputs, nhidden))
model:add(nn.Tanh())
model:add(nn.Linear(nhidden, nhidden_2))
model:add(nn.Tanh())
model:add(nn.Linear(nhidden_2, #classes))
model:add(nn.LogSoftMax())
parameters,gradParameters = model:getParameters()


criterion = nn.ClassNLLCriterion()

--number = #all-1

print (model)

trainDataStruct = {
   data = trainData_tensor,
   labels = labels_train_tensor,
   size = function() return trsize end
}

testDataStruct = {
   data = testData_tensor,
   labels = labels_test_tensor,
   size = function() return tesize end
}

--trainData = nil
--testData = nil
--labels = nil
all = nil

print ('Datasets created. Normalising data for columns 1-10 (i.e. 2-11) ...')

-- calculating for training data

mean = {}
std = {}

for j=1,10 do
 mean[j] = 0
 for i=1,trainDataStruct.size() do
   mean[j] = mean[j] + trainDataStruct.data[i][j]
 end
 mean[j] = mean[j]/trainDataStruct.size()
 --print("mean value of column number " .. j .. " is " .. mean[j])
end

sum = {}

for j=1,10 do
 sum[j] = 0
 for i=1,trainDataStruct.size() do
   sum[j] = sum[j] + ((trainDataStruct.data[i][j] - mean[j]) * (trainDataStruct.data[i][j] - mean[j]))
 end
 sum[j] = sum[j]/trainDataStruct.size()
 std[j] = math.sqrt(sum[j])
  --print("std value of column number " .. j .. " is " .. std[j])
end

for j=1,10 do
 for i=1,trainDataStruct.size() do
  trainDataStruct.data[i][j] = (trainDataStruct.data[i][j]-mean[j])/std[j]
 end
end

--calculating for test data


mean = {}
std = {}

for j=1,10 do
 mean[j] = 0
 for i=1,testDataStruct.size() do
   mean[j] = mean[j] + testDataStruct.data[i][j]
 end
 mean[j] = mean[j]/testDataStruct.size()
 --print("mean value of column number" .. j .. " is " .. mean[j])
end

sum = {}

for j=1,10 do
 sum[j] = 0
 for i=1,testDataStruct.size() do
   sum[j] = sum[j] + ((testDataStruct.data[i][j] - mean[j]) * (testDataStruct.data[i][j] - mean[j]))
 end
 sum[j] = sum[j]/testDataStruct.size()
 std[j] = math.sqrt(sum[j])
  --print("std value of column number" .. j .. " is " .. std[j])
end

for j=1,10 do
 for i=1,testDataStruct.size() do
  testDataStruct.data[i][j] = (testDataStruct.data[i][j]-mean[j])/std[j]
 end
end

print ("finished normalising. data prepared")

confusion = optim.ConfusionMatrix(classes)
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

reg = {}
reg[1] = model:get(2).weight
reg[2] = model:get(2).bias

function train(dataset)
   -- epoch tracker
   epoch = epoch or 1
  -- ctr = ctr or 1
   -- local vars
   
   --if (ctr+2000>15000) then
	--ctr = 1
   --end
   
   local time = sys.clock()

   -- do one epoch
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,dataset:size(),opt.batchSize do --,opt.batchSize do
      -- disp progress
      xlua.progress(t, dataset:size())

      -- create mini batch
      local inputs = {}--torch.Tensor((math.min(t+opt.batchSize-1,dataset:size())-t),55)
      local targets = {}--torch.Tensor((math.min(t+opt.batchSize-1,dataset:size())-t),1)
      --ctr = 1
      for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
         -- load new sample
         local input = dataset.data[i]
         local target = dataset.labels[i]
         --inputs[ctr] = input
         --targets[ctr] = target
         --ctr = ctr + 1
	 table.insert(inputs, input)
         table.insert(targets, target)
      end

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
                       -- get new parameters
                       if x ~= parameters then
                          parameters:copy(x)
                       end

                       -- reset gradients
                       gradParameters:zero()

                       -- f is the average of all criterions
                       local f = 0

                       -- evaluate function for complete mini batch
                       for i = 1,#inputs do
                          -- estimate f
                          local output = model:forward(inputs[i])
                          local err = criterion:forward(output, targets[i])
                          f = f + err

                          -- estimate df/dW
                          local df_do = criterion:backward(output, targets[i])
                          model:backward(inputs[i], df_do)

                          -- update confusion
                          confusion:add(output, targets[i])

                       end

                       -- normalize gradients and f(X)
                       gradParameters:div(#inputs)
                       f = f/#inputs

                       -- return f and df/dX
                       return f,gradParameters
                    end

      -- optimize on current mini-batch
      if opt.optimization == 'CG' then
         config = config or {maxIter = opt.maxIter}
         optim.cg(feval, parameters, config)

      elseif opt.optimization == 'LBFGS' then
         config = config or {learningRate = opt.learningRate,
                             maxIter = opt.maxIter,
                             nCorrection = 10}
         optim.lbfgs(feval, parameters, config)

      elseif opt.optimization == 'SGD' then
         config = config or {learningRate = opt.learningRate,
                             weightDecay = opt.weightDecay,
                             momentum = opt.momentum,
                             learningRateDecay = 5e-7}
         optim.sgd(feval, parameters, config)

      elseif opt.optimization == 'ASGD' then
         config = config or {eta0 = opt.learningRate,
                             t0 = nbTrainingPatches * opt.t0}
         _,_,average = optim.asgd(feval, parameters, config)

      else
         error('unknown optimization method')
      end
      --regularize
      for _,w in ipairs(reg) do
          w:add(weight_decay,w) 	
      end
	
   end

   -- time taken
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   confusion:zero()

   -- save/log current net
   local filename = paths.concat(opt.save, 'model.net')
   os.execute('mkdir -p ' .. paths.dirname(filename))
   if paths.filep(filename) then
      --os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
   end
   print('<trainer> saving network to '..filename)
   torch.save(filename, model)

   -- next epoch
   epoch = epoch + 1
   --ctr = ctr + 2000
   inputs = nil
   targets = nil
end


-- test function
function test(dataset)
   -- local vars
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- test over given dataset
   print('<trainer> on testing Set:')
   chtng = 0
   for t = 1,tesize do
      -- disp progress
      xlua.progress(t, dataset:size())

      -- get new sample
      local input = dataset.data[t]
      local target = dataset.labels[t]

      -- test sample
      local pred = model:forward(input)
 -------------------------------------------------
           
      --if(pred == target) then 
	confusion:add(pred, target)
            
      --else 
	--testing = math.random(2)
        --if(testing == 1) then 
        --	confusion:add(target, target)
          --      chtng = chtng+1 
        
        --else 
	--	confusion:add(pred, target)
	--end
      --end
-------------------------------------------------------
   end

   -- timing
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   print(chtng)
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   confusion:zero()

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
end


while true do
   -- train/test
   train(trainDataStruct)
   test(testDataStruct)

   trainLogger:style{['% mean class accuracy (train set)'] = '-'}
   testLogger:style{['% mean class accuracy (test set)'] = '-'}
   --trainLogger:plot()
   --testLogger:plot()
end
