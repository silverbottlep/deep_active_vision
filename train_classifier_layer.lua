require 'torch'
require 'nn'
require 'optim'
require 'cunn'
require 'cudnn'
local t = require './fb.resnet.torch/datasets/transforms'
torch.setdefaulttensortype('torch.FloatTensor')

opt = lapp[[
  --batch_size				(default 50)
  --img_scale          (default 160)
	--iter_per_epoch		(default 500)
  --lr								(default 0.0001)
  --max_epoch					(default 5)
  --dropout						(default 0.7)
  --optim_layer				(default 3)
  --n_phi							(default 5)
  --n_theta						(default 24)
  --n_object					(default 32)
  -g, --gpu           (default 1)
	-d, --debug					(default 0)
]]

if opt.debug>0 then
	debugger = require('fb.debugger')
	debugger.enter()
end

local model_name = string.format('resnet18_do0%d_imgscale%d',opt.dropout*10, opt.img_scale)

local timer = torch.Timer()
local data_timer = torch.Timer()

-- initialize model
print("loading snapshot...")
local loader = torch.load('snapshots/' .. model_name .. '_fctuned.t7')
local cnn = loader.cnn
local state = loader.state
local loss_list = loader.loss_list
local model = cnn:clone()

local criterion = nn.ClassNLLCriterion()

-- mean subtraction, data augmentation
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}
local pca = {
   eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
   eigvec = torch.Tensor{
      { -0.5675,  0.7192,  0.4009 },
      { -0.5808, -0.0045, -0.8140 },
      { -0.5836, -0.6948,  0.4203 },
   },
}
local transform = t.Compose{
	t.RandomCrop(opt.img_scale),
	 t.ColorJitter({
		 brightness = 0.4,
		 contrast = 0.4,
		 saturation = 0.4,
	 }),
	t.Lighting(0.1, pca.eigval, pca.eigvec),
	t.ColorNormalize(meanstd),
}

local input_batch = torch.Tensor(opt.batch_size, 3, opt.img_scale, opt.img_scale)
local target_batch = torch.Tensor(opt.batch_size)

if opt.gpu > 0 then
	cnn = cnn:cuda()
	criterion = criterion:cuda()
	input_batch = input_batch:cuda()
	target_batch = target_batch:cuda()
end

local params, grad_params = cnn:parameters()

-- setting learning rate for every layer
local optim_layers = {0, 4, 20, 38, 56, 74}
local optim_upto = optim_layers[opt.optim_layer]
local config = {}
for i=1, #params do
	local lr = opt.lr
	if i<=optim_upto then
		lr = 0
	end
	table.insert(config,{ 
		learningRate = lr, 
		beta1 = 0.9,
		beta2 = 0.999
	})
end

-- loading background
print('loading background images...')
local backgrounds = torch.load('data/background.t7')
local n_bg = backgrounds:size(1)
backgrounds = backgrounds:float():div(255)
print('loading training images...')
local train_images = torch.load('data/bigbird_crop.t7')
train_images = train_images:float():div(255)

cnn:training()
for e=1,opt.max_epoch do
	for i=1,opt.iter_per_epoch do
		-- prepare the image batch
		data_timer:reset(); data_timer:resume()
		for k=1,opt.batch_size do
			local is_bg = torch.random(10)
			local phi_id = torch.random(opt.n_phi)
			local theta_id = torch.random(opt.n_theta*2)
			local bg_id = torch.random(n_bg)
			local bg = backgrounds[bg_id]:squeeze()
			local im_bg, o_id
			if is_bg > 8 then
				o_id = opt.n_object+1
				im_bg = bg
			else
				-- combine background with alpha channels
				o_id = torch.random(opt.n_object)
				if theta_id > opt.n_theta then
					im_bg = train_images[o_id][phi_id][theta_id][{{1,3},{},{}}]:clone()
				else
					local im4 = train_images[o_id][phi_id][theta_id]
					local im = im4[{{1,3},{},{}}]
					local alpha = im4[{{4},{},{}}]
					alpha = alpha:repeatTensor(3,1,1)
					im_bg = im:cmul(alpha) + bg:cmul(1-alpha)	
				end
			end
			-- data augmentation
			im_bg = transform(im_bg)
			-- data augmentation(30% change, rotation 90, -90 degree)
			local rotate = torch.random(10)
			if rotate > 7 then
				if torch.random(2) == 1 then
					im_bg = image.rotate(im_bg,90*math.pi/180)
				else
					im_bg = image.rotate(im_bg,-90*math.pi/180)
				end
			end
			-- data augmentation(scale 0.1 ~ 1)
			local sc_idx = torch.random(10)
			local scales = {0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1}
			local sc = scales[sc_idx]
			if sc < 1 then
				im_bg = image.scale(im_bg,opt.img_scale*sc,opt.img_scale*sc)
				im_bg = image.scale(im_bg,opt.img_scale,opt.img_scale)
			end
			--image.save('temp/' .. string.format('%d.jpg',k), im_bg)
			input_batch[k]:copy(im_bg)
			target_batch[k]=o_id
		end
		data_timer:stop()

		timer:reset(); timer:resume()
	
		-- layer by layer optimization with different learning rate
		collectgarbage()
		cnn:zeroGradParameters()
		local outputs = cnn:forward(input_batch)
		err = criterion:forward(outputs, target_batch)
		local df_do = criterion:backward(outputs, target_batch)
		cnn:backward(input_batch,df_do)
		for l=1, #params do	
			local feval = function(x)
				return err, grad_params[l]
			end
			_, loss = optim.adam(feval, params[l], config[l])
			--_, loss = optim.sgd(feval, params[l], config[l])
		end

		timer:stop()
		loss_list = torch.cat(loss_list,torch.Tensor(1,1):fill(err),1)
		print("epoch: " .. e .. " iter: " .. i .. '/' .. opt.iter_per_epoch .. 
				string.format(' lr: %f training loss: %04f time: %04f, datatime: %04f',config[#params].learningRate, err, timer:time().real, data_timer:time().real))

	end
	local snapshot = {}
	snapshot.cnn = cnn:clearState()
	snapshot.state = state
	snapshot.config = config
	snapshot.loss_list = torch.Tensor(loss_list)
	torch.save('./snapshots/' .. model_name .. string.format('_upto%d_epoch%d',opt.optim_layer,e) .. '.t7', snapshot)
end
