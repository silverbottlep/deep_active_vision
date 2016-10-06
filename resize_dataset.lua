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
opt.scale = 0.2

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
n_object = #obj_names

local categories = torch.Tensor({1,1,0,1,0,1,0,0,0,1,1,1,0,1,1,1,1,0,1,1,0,0,1,1,0,1,1,1,1,1,1,1})

local height = 1080*opt.scale
local width = 1920*opt.scale

for scene=1,#scene_names do
	-- loading background
	print('loading testing images...')
	local dataset = torch.load(string.format('data/rohit_%s.t7',scene_names[scene]))
	local images_orig = dataset.images
	local annotations = dataset.annotations
	local n_image = images_orig:size(1)
	local images = torch.ByteTensor(n_image,3,height,width)

	for i=1,n_image do
		local im = images_orig[i]
		images[i]:copy(image.scale(im,width,height))
		for j=1,n_object do
			annotations[i][j][{{1,4}}] = annotations[i][j][{{1,4}}]*opt.scale
		end
	end

	print('saving dataset with results.....')
	dataset.images = images
	dataset.annotations = annotations
	torch.save(string.format('data/rohit_%s_resize.t7',scene_names[scene]),dataset)
end	-- for scene=1,#scene_names do
