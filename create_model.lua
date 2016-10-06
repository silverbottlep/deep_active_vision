require 'nn'
require 'nngraph'

create_model={}

function create_model.actor(opt)
	-- 216 x 384
	local input = nn.Identity()()
	local conv1 = nn.ReLU()(nn.SpatialConvolution(128,128,3,3,2,2,1,1)(input))
	local conv2 = nn.ReLU()(nn.SpatialConvolution(128,128,3,3,2,2,1,1)(conv1))
	-- 7 x 12
	local cnn_feat = nn.Reshape(128*7*12)(conv2)

	-- input move availability
	local input_move = nn.Identity()()
	local move = nn.ReLU()(nn.Linear(6,512)(input_move))
	-- input bb
	local input_bb = nn.Identity()()
	local bb = nn.ReLU()(nn.Linear(4,512)(input_bb))

	local concat = nn.JoinTable(2)({cnn_feat,move,bb})
	local fc1 = nn.Dropout(opt.dropout)(nn.ReLU()(nn.Linear(128*7*12+512+512,512)(concat)))
	local fc2 = nn.SoftMax()(nn.Linear(512,6)(fc1))

	return nn.gModule({input,input_move,input_bb},{fc2})
end

return create_model
