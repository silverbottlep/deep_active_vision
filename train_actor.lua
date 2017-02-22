require 'torch'
require 'nn'
require 'optim'
require 'cunn'
require 'cudnn'
require 'gnuplot'
local model_utils = require 'util.model_utils'
local t = require './fb.resnet.torch/datasets/transforms'
local create_model = require 'create_model'
torch.setdefaulttensortype('torch.FloatTensor')

opt = lapp[[
  --batch_size				(default 20)
  --img_scale         (default 160)
	--iter_per_epoch		(default 1000)
  --lr								(default 0.00005)
  --max_epoch					(default 30)
  --dropout						(default 0)
  --optim_layer				(default 3)
  --scale							(default 0.2)
  --threshold					(default 0.9)
	--T									(default 5)
	--split							(default 1)
  -g, --gpu           (default 1)
	-d, --debug					(default 0)
]]

if opt.debug>0 then
  debugger = require('fb.debugger')
	debugger.enter()
end

local model_name = string.format('actor_T%d_do0%d_lr%.6f_imgscale%d_split%d', 
			opt.T, opt.dropout*10, opt.lr, opt.img_scale, opt.split)

local timer = torch.Timer()
print("creating model")
local cnn_path = './snapshots/resnet-18.t7'
local cnn = torch.load(cnn_path)
cnn:remove(#cnn.modules)
cnn:remove(#cnn.modules)
cnn:remove(#cnn.modules)
cnn:remove(#cnn.modules)
cnn:remove(#cnn.modules)
local actor = create_model.actor(opt)

config = {
	learningRate = opt.lr,
	beta1 = 0.9,
	beta2 = 0.999,
}
state = {}
reward_list = torch.Tensor(1,1):fill(0)

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
--	 t.ColorJitter({
--		 brightness = 0.4,
--		 contrast = 0.4,
--		 saturation = 0.4,
--	 }),
--	t.Lighting(0.1, pca.eigval, pca.eigvec),
	t.ColorNormalize(meanstd),
}

local height = 1080*opt.scale
local width = 1920*opt.scale
local im_global_batch = torch.Tensor(opt.T, 1, 3, height, width)
local move_batch = torch.Tensor(opt.T, 1, 6)
local bb_batch = torch.Tensor(opt.T, 1, 4)

if opt.gpu > 0 then
	cnn = cnn:cuda()
	actor = actor:cuda()
	im_global_batch = im_global_batch:cuda()
	move_batch = move_batch:cuda()
	bb_batch = bb_batch:cuda()
end

local params, grad_params = actor:getParameters()
grad_params:zero()

print('cloning actor.... ')
actor_clones = model_utils.clone_many_times(actor, opt.T)

-- loading training images
scene_names = {}
--table.insert(scene_names, 'Bedroom_01_1')
table.insert(scene_names, 'Kitchen_Living_01_1')
--table.insert(scene_names, 'Kitchen_Living_02_1')
table.insert(scene_names, 'Kitchen_Living_03_1')
table.insert(scene_names, 'Kitchen_Living_03_2')
table.insert(scene_names, 'Kitchen_Living_04_2')
table.insert(scene_names, 'Kitchen_05_1')
table.insert(scene_names, 'Kitchen_Living_06')
table.insert(scene_names, 'Kitchen_Living_08_1')
table.insert(scene_names, 'Office_01_1')
datasets = {}
n_trains = torch.Tensor(#scene_names)
for i=1,#scene_names do
	print('loading training images... ' .. scene_names[i])
	datasets[i] = torch.load(string.format('data/rohit_%s_resize.t7',scene_names[i]))
	datasets[i].images = datasets[i].images:float():div(255)
	n_trains[i] = datasets[i].indices:size(1)
end
n_scene = #scene_names

for e=1,opt.max_epoch do
	for i=1,opt.iter_per_epoch do
		-- prepare the image batch
		local batch_reward = 0
		timer:reset(); timer:resume()
		for b=1,opt.batch_size do

		-- initial correct and score
		local init_correct = 0
		local init_score = 0
		local correct = 0
		local score = 0

		local scene_id = torch.random(n_scene)
		local train_idx = datasets[scene_id].indices[torch.random(n_trains[scene_id])]
		local image_id = datasets[scene_id].candidates[train_idx][1]
		local object_id = datasets[scene_id].candidates[train_idx][2]
		local input_im = datasets[scene_id].images[image_id]
		input_im = transform(input_im)
		local bb = datasets[scene_id].annotations[image_id][object_id][{{1,4}}]
		local w = bb[3]-bb[1]
		local h = bb[4]-bb[2]
		bb_batch[1][1][1] = (bb[1]+w/2)/width
		bb_batch[1][1][2] = (bb[2]+h/2)/height
		bb_batch[1][1][3] = w/width
		bb_batch[1][1][4] = h/height

		correct = datasets[scene_id].annotations[image_id][object_id][5]
		score = datasets[scene_id].annotations[image_id][object_id][6]
		init_correct = correct
		init_score = score
		im_global_batch[1][1]:copy(input_im)
		local move_avail = datasets[scene_id].moves[image_id]  
		move_batch[1][1]:copy(move_avail:gt(0))
		if b==1 then
			print(string.format('Episodes %d: ', i))
			print(string.format('image_id: %d, correct: %d, score: %.4f', image_id, correct, score))
		end

		local next_im = input_im
		local next_bb = bb
		local next_move_avail = move_avail

		local actions = torch.Tensor(opt.T)
		local probs = torch.Tensor(opt.T, 6):fill(0):cuda()
		local conv_feat = torch.Tensor(opt.T,1,128,27,48):cuda()
		local last_t 
		for t=1,opt.T do
			last_t = t
			cnn:evaluate()
			actor_clones[t]:training()
			-- forward propagation
			conv_feat[t]  = cnn:forward(im_global_batch[t]):clone()
			probs[t] = actor_clones[t]:forward({conv_feat[t],move_batch[t],bb_batch[t]})
			actions[t] = torch.multinomial(probs[t], 1):squeeze()
			-- take the action
			local next_image_id = next_move_avail[actions[t]]
			-- next_move is available
			if next_image_id > 0 then
				next_im = datasets[scene_id].images[next_image_id]
				next_im = transform(next_im)
				next_bb = datasets[scene_id].annotations[next_image_id][object_id][{{1,4}}]
				next_move_avail = datasets[scene_id].moves[next_image_id]  
				correct = datasets[scene_id].annotations[next_image_id][object_id][5]
				score = datasets[scene_id].annotations[next_image_id][object_id][6]
				if b==1 then
					print(string.format('action: %d, image_id: %d, correct: %d, score: %.4f, bb(%d,%d,%d,%d)', actions[t], next_image_id, correct, score, next_bb[1],next_bb[2],next_bb[3],next_bb[4]))
				end
			else
				if b==1 then
					print(string.format('action: %d(unavailable!), image_id: %d, correct: %d, score: %.4f, bb(%d,%d,%d,%d)', actions[t], next_image_id, correct, score, next_bb[1],next_bb[2],next_bb[3],next_bb[4]))
				end
			end
			-- we find the object!
			if score >= opt.threshold then
				if b==1 then
					print('FOUND!')
				end
				break
			end
			-- prepare data for next time step
			if t < opt.T then
				im_global_batch[t+1][1]:copy(next_im)
				move_batch[t+1][1]:copy(next_move_avail:gt(0))
				local w = next_bb[3]-next_bb[1]
				local h = next_bb[4]-next_bb[2]
				bb_batch[t+1][1][1] = (next_bb[1]+w/2)/width
				bb_batch[t+1][1][2] = (next_bb[2]+h/2)/height
				bb_batch[t+1][1][3] = w/width
				bb_batch[t+1][1][4] = h/height
			end
		end

--		gnuplot.figure(1)
--		gnuplot.raw('set multiplot layout 5,1')
--		for t=1,opt.T do
--			gnuplot.raw('unset xtics')
--			gnuplot.raw('unset ytics')
--			gnuplot.imagesc(im_global_batch[t][1][1])
--		end	
--		gnuplot.raw('unset multiplot')

		-- compute dlogp
		local dlogp = torch.Tensor(last_t, 6):fill(0)
		dlogp = dlogp:scatter(2,actions[{{1,last_t}}]:reshape(last_t,1):long(),1):cuda()
		dlogp:cdiv(probs[{{1,last_t},{}}]:add(0.00000001))
		-- multiply by reward(score we've got)
		if score < init_score then score = 0 end
		batch_reward = batch_reward + score
		dlogp:cmul(torch.Tensor(1):fill(score):repeatTensor(last_t,6):cuda())
		-- gradient descent
		dlogp:mul(-1)
		-- compute gradiets
		for t=1,last_t do
			actor_clones[t]:backward({conv_feat[t],move_batch[t],bb_batch[t]},dlogp[t])
		end

		-- for debugging
		if b==1 then
			for t=1,last_t do
				print(string.format('%.3f,%.3f,%.3f,%.3f,%.3f,%.3f', 
				probs[t][1],probs[t][2],probs[t][3],probs[t][4],probs[t][5],probs[t][6]))
			end
			--io.stdin:read('*l')
		end

		end --for b=1,opt.batch_size do

		-- update gradients
		local feval = function(x)
			collectgarbage()
			return err, grad_params
		end
		optim.adam(feval, params, config)
		grad_params:zero()
		timer:stop()

		batch_reward = batch_reward / opt.batch_size
		reward_list = torch.cat(reward_list,torch.Tensor(1,1):fill(batch_reward),1)
		print("epoch: " .. e .. " iter: " .. i .. '/' .. opt.iter_per_epoch .. 
				string.format(' lr: %f training reward: %04f time: %04f',
				config.learningRate, batch_reward, timer:time().real))
	end
	if e%10 == 0 then
		snapshot = {}
		snapshot.actor = actor
		snapshot.state = state
		snapshot.config = config
		snapshot.reward_list = torch.Tensor(reward_list)
		torch.save(string.format('./snapshots/%s_epoch%d.t7', model_name, e),snapshot)
	end
end

