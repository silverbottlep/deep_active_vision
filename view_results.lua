require 'torch'
require 'gnuplot'
require 'image'

opt = {}
opt.debug = 0
if opt.debug>0 then
  debugger = require('fb.debugger')
	debugger.enter()
end
-- loading training images
scene_names = {}
--table.insert(scene_names, 'Bedroom_01_1')
--table.insert(scene_names, 'Kitchen_Living_01_1')
--table.insert(scene_names, 'Kitchen_Living_02_1')
--table.insert(scene_names, 'Kitchen_Living_03_1')
--table.insert(scene_names, 'Kitchen_Living_03_2')
--table.insert(scene_names, 'Kitchen_Living_04_2')
--table.insert(scene_names, 'Kitchen_05_1')
--table.insert(scene_names, 'Kitchen_Living_06')
--table.insert(scene_names, 'Kitchen_Living_08_1')
table.insert(scene_names, 'Office_01_1')
scene_id = 1
results = torch.load(string.format('actor_results_%s.t7',scene_names[scene_id]))
datasets = torch.load(string.format('data/rohit_%s.t7',scene_names[scene_id]))
images = datasets.images
annotations = datasets.annotations

N = results:size(1)
T = results:size(2)-1

local width = images[1]:size(3)
local height = images[1]:size(2)

while 1 do
	print(string.format('enter a episode number ( 1 - %d ): ',N))
	n = io.read("*n") -- read a number
	result = results[n]

	local object_id = result[1][1]
	local image_id = result[1][2]
	local action = result[1][3]
	local correct = result[1][4]
	local score = result[1][5]
	--local box = result[1][{{6,9}}]*2.5
	local box = annotations[image_id][object_id]*0.5
	if box[1] < 1 then box[1] = 1 end
	if box[2] > width then box[1] = width end
	if box[3] < 1 then box[3] = 1 end
	if box[4] > height then box[1] = height end
	local im = images[image_id]
	local crop_im = image.crop(im,box[1],box[2],box[3],box[4])
	print(string.format('object id: %d, action: %d, image_id: %d, correct: %d, score: %.4f', 
			object_id, action, image_id, correct, score))
	crop_im = image.crop(im,box[1],box[2],box[3],box[4])
	image.save(string.format('./temp/scene%d_episode%d_im_t0.jpg',scene_id,n),im)
	image.save(string.format('./temp/scene%d_episode%d_bb_t0.jpg',scene_id,n),crop_im)
--	gnuplot.figure(1)
--	gnuplot.imagesc(im[1])
--	gnuplot.figure(1*(T+1))
--	gnuplot.imagesc(crop_im[1])

	for t=1,T do
		action = result[t+1][3]
		if action > 0 then
			object_id = result[t+1][1]
			image_id = result[t+1][2]
			correct = result[t+1][4]
			score = result[t+1][5]
			--box = result[t+1][{{6,9}}]*2.5
			box = annotations[image_id][object_id]*0.5
			if box[1] < 1 then box[1] = 1 end
			if box[2] > width then box[2] = width end
			if box[3] < 1 then box[3] = 1 end
			if box[4] > height then box[4] = height end

			im = images[image_id]
			image.save(string.format('./temp/scene%d_episode%d_im_t%d.jpg',scene_id,n,t),im)
--			gnuplot.figure(t+1)
--			gnuplot.imagesc(im[1])
			print(string.format('object id: %d, action: %d, image_id: %d, correct: %d, score: %.4f', 
					object_id, action, image_id, correct, score))

			if box[2] > 0 then
				crop_im = image.crop(im,box[1],box[2],box[3],box[4])
				image.save(string.format('./temp/scene%d_episode%d_bb_t%d.jpg',scene_id,n,t),crop_im)
--				gnuplot.figure((t+1)*(T+1))
--				gnuplot.imagesc(crop_im[1])
			end
		end
	end
end
