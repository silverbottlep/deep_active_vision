require 'torch'
require 'nn'
require 'optim'
require 'cunn'
require 'cudnn'
require 'nngraph'
require 'gnuplot'
require 'paths'
local t = require './fb.resnet.torch/datasets/transforms'
local imagenetLabel = require './fb.resnet.torch/pretrained/imagenet'
local w_init = require 'weight-init'
torch.setdefaulttensortype('torch.FloatTensor')

opt = lapp[[
  --batch_size				(default 50)
  --img_scale         (default 160)
  --epoch							(default 5)
  --dropout						(default 0.7)
  --optim_layer				(default 3)
  --scale							(default 0.5)
  --plot							(default 1)
	--save							(default 1)
  -g, --gpu           (default 1)
	-d, --debug					(default 0)
]]

if opt.debug>0 then
  debugger = require('fb.debugger')
	debugger.enter()
end

scene_names = {}
table.insert(scene_names, 'Bedroom_01_1')
table.insert(scene_names, 'Kitchen_Living_01_1')
table.insert(scene_names, 'Kitchen_Living_02_1')
table.insert(scene_names, 'Kitchen_Living_03_1')
table.insert(scene_names, 'Kitchen_Living_03_2')
table.insert(scene_names, 'Kitchen_Living_04_2')
table.insert(scene_names, 'Kitchen_05_1')
table.insert(scene_names, 'Kitchen_Living_06')
table.insert(scene_names, 'Kitchen_Living_08_1')
table.insert(scene_names, 'Office_01_1')

obj_names_fname = 'object_names.txt'
file = io.open(obj_names_fname)
obj_names= {}
if file then
	for line in file:lines() do
		table.insert(obj_names,line)
	end
end
n_object = #obj_names

local model_name = string.format('resnet18_do0%d_imgscale%d', opt.dropout*10, opt.img_scale)
local loader = torch.load('snapshots/' .. model_name .. string.format('_upto%d_epoch%d.t7',opt.optim_layer,opt.epoch))
local cnn = loader.cnn

local criterion = nn.ClassNLLCriterion()

-- mean subtraction, data augmentation
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}
local transform = t.Compose{
	 t.ColorNormalize(meanstd),
}

local input_batch = torch.Tensor(1, 3, opt.img_scale, opt.img_scale)
local target_batch = torch.Tensor(1)

if opt.gpu > 0 then
	cnn = cnn:cuda()
	criterion = criterion:cuda()
	input_batch = input_batch:cuda()
	target_batch = target_batch:cuda()
end

local areas={0,30,40,50,60,70,80,90,100,115,120,130,140,150}
local n_correct_area = torch.Tensor(#areas):fill(0)
local n_tot_area = torch.Tensor(#areas):fill(0)
local n_correct = torch.Tensor(n_object):fill(0)
local n_tot = torch.Tensor(n_object):fill(0)
local n_tot_scene = torch.Tensor(n_object):fill(0)
local n_correct_comb = torch.Tensor(n_object,#areas):fill(0)
local n_tot_comb = torch.Tensor(n_object,#areas):fill(0)

local height = 1080*opt.scale
local width = 1920*opt.scale

-- making results directory
if opt.plot>0 then
	if not paths.dirp('./results/') then
		paths.mkdir('./results/')
	end
	for i=1,n_object do
		name = './results/'.. i
		if not paths.dirp(name) then
			paths.mkdir(name)
		end
		name = './results/'.. string.format('%d/bad/',i)
		if not paths.dirp(name) then
			paths.mkdir(name)
		end
		name = './results/'.. string.format('%d/good/',i)
		if not paths.dirp(name) then
			paths.mkdir(name)
		end
	end
end

for scene=1,#scene_names do
	-- loading background
	print('loading testing images...')
	local dataset = torch.load(string.format('data/rohit_%s.t7',scene_names[scene]))
	local images = dataset.images
	local annotations = dataset.annotations
	local moves = dataset.moves
	local n_image = images:size(1)
	local candidates = torch.Tensor(images:size(1)):fill(0)
	n_tot_scene:fill(0)

	for i=1,n_image do
		local whole_im = images[i]
		for j=1,n_object do
			-- check if this object is on the scene
			if annotations[i][j][2] > 0 then
				local obj_id = j
				local obj_name = obj_names[obj_id]
				local x1 = torch.round(annotations[i][j][1]*opt.scale)
				local y1 = torch.round(annotations[i][j][2]*opt.scale)
				local x2 = torch.round(annotations[i][j][3]*opt.scale)
				local y2 = torch.round(annotations[i][j][4]*opt.scale)
				local w = x2-x1+1
				local h = y2-y1+1
				local area = torch.sqrt(w*h)
				local crop_im = image.crop(whole_im,x1,y1,x2,y2):float()
				crop_im:div(255)

				local plot_im = crop_im:clone()
				crop_im = image.scale(crop_im, opt.img_scale, opt.img_scale)
				bg = transform(crop_im)

				input_batch[1]:copy(bg)
				target_batch[1] = obj_id
				cnn:evaluate()
				local output = cnn:forward(input_batch):clone()
				local prob = torch.exp(output:squeeze())
				local top5_prob, top5_idx = prob:topk(5,1,true,true) 

				correct = 0
				if obj_id == top5_idx[1] then
					correct = 1
				end

				-- counting per area
				local area_idx=1
				while 1 do
					if area < areas[area_idx] then
						n_tot_area[area_idx] = n_tot_area[area_idx] + 1	
						n_tot_comb[obj_id][area_idx] = n_tot_comb[obj_id][area_idx] + 1
						if correct == 1 then
							n_correct_area[area_idx] = n_correct_area[area_idx] + 1	
							n_correct_comb[obj_id][area_idx] = n_correct_comb[obj_id][area_idx] + 1
						end
						break
					end
					if area_idx == #areas then
						n_tot_area[area_idx] = n_tot_area[area_idx] + 1	
						n_tot_comb[obj_id][area_idx] = n_tot_comb[obj_id][area_idx] + 1
						if correct == 1 then
							n_correct_area[area_idx] = n_correct_area[area_idx] + 1	
							n_correct_comb[obj_id][area_idx] = n_correct_comb[obj_id][area_idx] + 1
						end
						break
					end
					area_idx = area_idx+1
				end

				-- counting per category
				n_tot[obj_id] = n_tot[obj_id] + 1
				n_tot_scene[obj_id] = n_tot_scene[obj_id] + 1
				if correct == 1 then
					n_correct[obj_id] = n_correct[obj_id] + 1
				end
				print(string.format('%02d_%05d, %d/%d(%.03f) area: %.03f, gt: %s, answer: %s, prob: %.04f', obj_id, n_tot[obj_id], n_correct:sum(), n_tot:sum(), n_correct:sum()/n_tot:sum(), area, obj_id, top5_idx[1], top5_prob[1]))

				annotations[i][j][5] = correct
				annotations[i][j][6] = prob[obj_id]

				if opt.plot > 0 then
					if correct == 1 then
						fname = string.format('results/%d/good/%d_%d_%05d_%.04f.jpg',obj_id,scene,area_idx,n_tot_scene[obj_id], prob[obj_id])
					else
						fname = string.format('results/%d/bad/%d_%d_%05d_%.04f.jpg',obj_id,scene,area_idx,n_tot_scene[obj_id], prob[obj_id])
					end
					image.save(fname, plot_im)
				end
			end
		end -- for j=1,n_object do
	end -- for i=1,n_image do
	if opt.save > 0 then
		print('saving dataset with results.....')
		dataset.annotations = annotations
		torch.save(string.format('data/rohit_%s.t7',scene_names[scene]),dataset)
	end
end	-- for scene=1,#scene_names do

print('category statistics------------------------------')
for i=1,n_object do
	print(string.format('category: %d, %d/%d(%.03f)',i, n_correct[i],n_tot[i],n_correct[i]/n_tot[i]))
end
print('area statistics------------------------------')
for i=1,#areas-1 do
	print(string.format('area: %d - %d, %d/%d(%.03f)',areas[i], areas[i+1], n_correct_area[i],n_tot_area[i],n_correct_area[i]/n_tot_area[i]))
end
print(string.format('area: %d above, %d/%d(%.03f)',areas[#areas], n_correct_area[#areas],n_tot_area[#areas],n_correct_area[#areas]/n_tot_area[#areas]))

print('combine statistics------------------------------')
print(n_correct_comb)
print(n_tot_comb)
print(n_correct_comb:cdiv(n_tot_comb))
