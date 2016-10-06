require 'torch'
require 'image'
require 'gnuplot'

local opt = {}
opt.debug = 0
if opt.debug>0 then
	debugger = require('fb.debugger')
	debugger.enter()
end

scene_name = 'Bedroom_01_1'
--scene_name = 'Kitchen_Living_01_1'
--scene_name = 'Kitchen_Living_02_1'
--scene_name = 'Kitchen_Living_03_1'
--scene_name = 'Kitchen_Living_03_2'
--scene_name = 'Kitchen_Living_04_2'
--scene_name = 'Kitchen_05_1'
--scene_name = 'Kitchen_Living_06'
--scene_name = 'Kitchen_Living_08_1'
--scene_name = 'Office_01_1'

print('loading dataset...')
dataset = torch.load(string.format('./data/rohit_%s_resize.t7',scene_name))
images = dataset.images
moves = dataset.moves

n_image = images:size(1)
n_move = 6


-- 1 forward
-- 2 backward
-- 3 left
-- 4 right
-- 5 rotate clockwise
-- 6 rotate counter clockwise

image_id = torch.random(n_image)
while 1 do
	gnuplot.imagesc(images[image_id][1])
	print(moves[image_id])
	print('enter a command ( 1 - 6 ): ')
	command = io.read("*n") -- read a number
	key = moves[image_id][command]
	if key > 0 then
		image_id = key
	else
		print('can not perform the action: ' .. key)
	end
end
