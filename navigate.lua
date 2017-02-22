require 'torch'
require 'image'
require 'gnuplot'

--local scene_names = {'Home_01_1','Home_01_2','Home_02_1','Home_03_1','Home_03_2','Home_04_1','Home_04_2','Home_05_1','Home_05_2','Home_06_1','Home_08_1','Home_14_1','Home_14_2','Office_01_1'}

opt = lapp[[
  --scene_name				(default 'Home_02_1')
	-d, --debug					(default 0)
]]

if opt.debug>0 then
	debugger = require('fb.debugger')
	debugger.enter()
end

print('loading dataset...')
dataset = torch.load(string.format('./data/rohit_%s.t7',opt.scene_name))
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
	-- show grey-scale image
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
