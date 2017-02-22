require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'image'
require 'paths'
local t = require './fb.resnet.torch/datasets/transforms'
torch.setdefaulttensortype('torch.FloatTensor')

opt = lapp[[
  --data_dir					(default '/home/eunbyung/Works/data/rohit/')
  --output_dir				(default './data')
	--scale							(default 0.2)
	-g, --gpu           (default 1) 
	-d, --debug					(default 0)
]]

if opt.debug>0 then
	debugger = require('fb.debugger')
	debugger.enter()
end

function string:splitAtSpaces()
  local sep, values = " ", {}
  local pattern = string.format("([^%s]+)", sep)
  self:gsub(pattern, function(c) values[#values+1] = c end)
  return values
end

function loadData(dataFile)
  local data = {}
	local i=1
  for line in io.lines(dataFile) do
    local values = line:splitAtSpaces()
    data[i] = torch.FloatTensor(values)
    i = i + 1
  end
  return data
end

-- scene names and ids
local scene_names = {'Home_01_1','Home_01_2','Home_02_1','Home_03_1','Home_03_2','Home_04_1','Home_04_2','Home_05_1','Home_05_2','Home_06_1','Home_08_1','Home_14_1','Home_14_2','Office_01_1'}
local scene_ids = {00011,00012,00021,00031,00032,00041,00042,00051,00052,00061,00081,00141,00142,10011}
-- object categories used for training(not all object categories are used)
--local old_object_filter = torch.Tensor({1,1,0,1,0,1,0,0,0,1,1,1,0,1,1,1,1,0,1,1,0,0,1,1,0,1,1,1,1,1,1,1})
local object_filter = torch.Tensor({1,1,0,1,0,1,0,0,0,1,1,1,0,1,1,1,0,0,1,1,0,1,1,1,1,1,1,1,1,0,1,0,1,})

-- image size(make it smaller)
local height = 1080*opt.scale
local width = 1920*opt.scale
-- the number of object categories
local n_object = 33
-- the number of actions(moves)
local n_move = 6

-- loading classfier for measuring score of bounding boxes
local loader = torch.load('snapshots/resnet18_classifier.t7')
local cnn = loader.cnn
local input_batch = torch.Tensor(1, 3, 160, 160)
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}
local transform = t.Compose{
	 t.ColorNormalize(meanstd),
}

-- pretrained classifier has different category orders than released dataset
-- object_names_new.txt vs object_names_old.txt
local class_idx = torch.Tensor({1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,22,33,23,24,25,27,28,29,30,31,32,16,17,18,19,21,26})
function class_old2new(old_prob)
	local new_prob = torch.zeros(n_object)
	for i=1,n_object do
		new_prob[i] = old_prob[class_idx[i]]
	end
  return new_prob
end

-- getting annotations(each entry - x1,y1,x2,y2,correct,prob)
function get_annotations(scene_id, N)
	local annon_dir = paths.concat(opt.data_dir,scene_names[scene_id],'annotations')
	local annotations = torch.Tensor(N,n_object,6):fill(0)
	for i=1,N do
		local fname = paths.concat(annon_dir,
			string.format('%05d%06d0101_boxes.txt',scene_ids[scene_id],i))
		local annon = loadData(fname)
		if annon ~= nil then
			for j=1,#annon do
				-- i:image_idx, j:object_id
				annotations[i][annon[j][1]][{{1,4}}] = annon[j][{{2,5}}]
			end
		end
	end
	return annotations
end

-- getting moves (forward,backward,left,right,rotate clockwise,rotatecounter clockwise)
function get_moves(scene_id, N)
	local move_dir = paths.concat(opt.data_dir,scene_names[scene_id],'moves')
	local moves = torch.Tensor(N,n_move)
	for i=1,N do
		local fname = paths.concat(move_dir,
			string.format('%05d%06d0101_moves.txt',scene_ids[scene_id],i))
		local move = loadData(fname)
		for j=1,n_move do
			moves[i][j] = move[j][2]
		end
	end
	return moves
end

-- getting classifier scores (evaluated by pretrained classifier)
function get_score(im, annon)
	-- iterate over the object categories in the image
	-- only one instance of each object categories can show up in an image
	for j=1,n_object do
		-- check if this object is in the image, and crop the bounding box
		-- classify it, and save if top prediction is correct and its score
		if annon[j][2] > 0 then
			local obj_id = j
			local x1 = torch.round(annon[obj_id][1])
			local y1 = torch.round(annon[obj_id][2])
			local x2 = torch.round(annon[obj_id][3])
			local y2 = torch.round(annon[obj_id][4])
			local w = x2-x1+1
			local h = y2-y1+1
			local crop_im = image.crop(im,x1,y1,x2,y2):float():div(255)
			crop_im = image.scale(crop_im, 160, 160)
			input_batch[1]:copy(transform(crop_im))
			local output = cnn:forward(input_batch):clone()
			local prob = torch.exp(output:squeeze())
			local new_prob = class_old2new(prob)
			local top5_prob, top5_idx = new_prob:topk(5,1,true,true) 
			local pred_id = top5_idx[1]
			local correct = 0
			if obj_id == pred_id then
				correct = 1
			end
			annon[obj_id][5] = correct
			annon[obj_id][6] = new_prob[obj_id]
		end
	end
	return annon
end

-- When training action network, we only picked up the object categories
-- that have both good and bad views in the scene.
-- a good view means it was classified correctly and vice versa.
-- So, agent can start with the bad view, and will find good view for the object
function get_candidate_objects(annotations,N)
	local candidate_objects = torch.zeros(n_object)
	local is_good_box = torch.zeros(n_object)
	local is_bad_box = torch.zeros(n_object)
	for i=1,N do
		for j=1,n_object do
			if annotations[i][j][2] > 0 then
				local obj_id = j
				local correct = annotations[i][j][5]
				if correct > 0 then
					is_good_box[j] = 1
				else
					is_bad_box[j] = 1
				end
			end
		end
	end
	for j=1,n_object do
		if is_good_box[j]==1 and is_bad_box[j]==1 and object_filter[j]==1 then
			candidate_objects[j] = 1
		end
	end
	return candidate_objects
end

function get_candidates(annotations,candidate_objects,N)
	-- we don't start with too confident views
	local score_threshold = 0.9
	local object_list = {}
	local n_train_objects = 0
	for i=1,N do
		for j=1,n_object do
			-- check if this object is on the scene
			if annotations[i][j][2] > 0 then
				local obj_id = j
				local correct = annotations[i][j][5]
				local score = annotations[i][j][6]
				if score < score_threshold and candidate_objects[j] == 1 then
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
	return candidates
end

if opt.gpu > 0 then
	cnn = cnn:cuda()
	input_batch = input_batch:cuda()
end
cnn:evaluate()

for k=1,#scene_names do
	print(string.format('processing scene:%d, %s',k, scene_names[k]))
	local rgb_dir = paths.concat(opt.data_dir,scene_names[k],'rgb')
	local image_list = paths.dir(rgb_dir)
	-- remove '.' '..'
	table.sort(image_list)
	table.remove(image_list,1)
	table.remove(image_list,1)
	local n_image = #image_list

	-- getting annotations
	local annotations = get_annotations(k, n_image)
	-- getting moves (forward,backward,left,right,rotate clockwise,rotatecounter clockwise)
	local moves = get_moves(k, n_image)

	-- getting images and scores for each bounding boxes by pretrained classifier
	local images = torch.ByteTensor(n_image, 3, height, width)
	for i=1,n_image do
		-- image scale is [0,255]
		print(string.format('processing scan:%d, image:%d',k,i))
		local fname = paths.concat(rgb_dir,string.format('%05d%06d0101.jpg',scene_ids[k],i))
		local im = image.load(fname,3,'byte')
		annotations[i] = get_score(im, annotations[i])	

		-- resize whole image smaller
		im = image.scale(im, width, height)
		images[i]:copy(im)
	end

	-- get candidate object categories for training
	local candidate_objects = get_candidate_objects(annotations, n_image)
	-- get candidate object views(each entry - image_id, object_id)
	local candidates = get_candidates(annotations, candidate_objects, n_image)
	
	-- save dataset
	dataset={}
	dataset.images = images
	dataset.annotations = annotations
	dataset.moves = moves
	dataset.candidates = candidates
	torch.save(paths.concat(opt.output_dir,string.format('rohit_%s.t7',scene_names[k])),dataset)
end
