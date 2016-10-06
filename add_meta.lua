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

opt = {}
opt.debug = 0
opt.threshold = 0.9

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

if opt.debug>0 then
  debugger = require('fb.debugger')
	debugger.enter()
end

obj_names_fname = 'object_names.txt'
file = io.open(obj_names_fname)
obj_names= {}
if file then
	for line in file:lines() do
		table.insert(obj_names,line)
	end
end
local n_object = #obj_names
local n_scene = #scene_names

local categories = torch.Tensor({1,1,0,1,0,1,0,0,0,1,1,1,0,1,1,1,1,0,1,1,0,0,1,1,0,1,1,1,1,1,1,1})

local is_good_box = torch.Tensor(n_scene,n_object):fill(0)
local is_bad_box = torch.Tensor(n_scene,n_object):fill(0)
local candidate_instances = torch.Tensor(n_scene,n_object):fill(0)
for scene=1,n_scene do
	print('loading testing images...' .. scene_names[scene])
	local dataset = torch.load(string.format('data/rohit_%s_resize.t7',scene_names[scene]))
	local images = dataset.images
	local annotations = dataset.annotations
	local n_image = images:size(1)
	for i=1,n_image do
		for j=1,n_object do
			if annotations[i][j][2] > 0 then
				local obj_id = j
				local correct = annotations[i][j][5]
				if correct > 0 then
					is_good_box[scene][j] = 1
				else
					is_bad_box[scene][j] = 1
				end
			end
		end
	end
	for j=1,n_object do
		if is_good_box[scene][j]==1 and is_bad_box[scene][j]==1 then
			candidate_instances[scene][j] = 1
		end
	end

	local object_list = {}
	local n_train_objects = 0
	for i=1,n_image do
		local whole_im = images[i]
		for j=1,n_object do
			-- check if this object is on the scene
			if annotations[i][j][2] > 0 then
				local obj_id = j
				local correct = annotations[i][j][5]
				local score = annotations[i][j][6]
				if categories[obj_id] == 1 and score < opt.threshold and candidate_instances[scene][j] == 1 then
					n_train_objects = n_train_objects + 1
					local temp = torch.Tensor(2)
					temp[1] = i
					temp[2] = j 
					object_list[n_train_objects] = temp
				end
			end 
		end 
	end 

	local candidates = torch.Tensor(n_train_objects,2)
	for t=1,n_train_objects do
		candidates[t] = object_list[t]
	end
	indices = torch.randperm(n_train_objects)
	train_indices = indices[{{1,n_train_objects*0.7-1}}]
	val_indices = indices[{{n_train_objects*0.7,n_train_objects*0.8-1}}]
	test_indices = indices[{{n_train_objects*0.8,n_train_objects}}]

	print('saving dataset with results.....' .. n_train_objects)
	dataset.candidates = candidates
	dataset.indices = indices
	dataset.train_indices = train_indices
	dataset.val_indices = val_indices
	dataset.test_indices = test_indices
	torch.save(string.format('data/rohit_%s_resize.t7',scene_names[scene]),dataset)
end
