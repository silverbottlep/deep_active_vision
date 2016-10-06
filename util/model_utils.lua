
-- adapted from https://github.com/wojciechz/learning_to_execute
-- utilities for combining/flattening parameters in a model
-- the code in this script is more general than it needs to be, which is 
-- why it is kind of a large

require 'torch'
local model_utils = {}
function model_utils.combine_all_parameters(...)
    --[[ like module:getParameters, but operates on many modules ]]--

    -- get parameters
    local networks = {...}
    local parameters = {}
    local gradParameters = {}
    for i = 1, #networks do
        local net_params, net_grads = networks[i]:parameters()

        if net_params then
            for _, p in pairs(net_params) do
                parameters[#parameters + 1] = p
            end
            for _, g in pairs(net_grads) do
                gradParameters[#gradParameters + 1] = g
            end
        end
    end

    local function storageInSet(set, storage)
        local storageAndOffset = set[torch.pointer(storage)]
        if storageAndOffset == nil then
            return nil
        end
        local _, offset = unpack(storageAndOffset)
        return offset
    end

    -- this function flattens arbitrary lists of parameters,
    -- even complex shared ones
    local function flatten(parameters)
        if not parameters or #parameters == 0 then
            return torch.Tensor()
        end
        local Tensor = parameters[1].new

        local storages = {}
        local nParameters = 0
        for k = 1,#parameters do
            local storage = parameters[k]:storage()
            if not storageInSet(storages, storage) then
                storages[torch.pointer(storage)] = {storage, nParameters}
                nParameters = nParameters + storage:size()
            end
        end

        local flatParameters = Tensor(nParameters):fill(1)
        local flatStorage = flatParameters:storage()

        for k = 1,#parameters do
            local storageOffset = storageInSet(storages, parameters[k]:storage())
            parameters[k]:set(flatStorage,
                storageOffset + parameters[k]:storageOffset(),
                parameters[k]:size(),
                parameters[k]:stride())
            parameters[k]:zero()
        end

        local maskParameters=  flatParameters:float():clone()
        local cumSumOfHoles = flatParameters:float():cumsum(1)
        local nUsedParameters = nParameters - cumSumOfHoles[#cumSumOfHoles]
        local flatUsedParameters = Tensor(nUsedParameters)
        local flatUsedStorage = flatUsedParameters:storage()

        for k = 1,#parameters do
            local offset = cumSumOfHoles[parameters[k]:storageOffset()]
            parameters[k]:set(flatUsedStorage,
                parameters[k]:storageOffset() - offset,
                parameters[k]:size(),
                parameters[k]:stride())
        end

        for _, storageAndOffset in pairs(storages) do
            local k, v = unpack(storageAndOffset)
            flatParameters[{{v+1,v+k:size()}}]:copy(Tensor():set(k))
        end

        if cumSumOfHoles:sum() == 0 then
            flatUsedParameters:copy(flatParameters)
        else
            local counter = 0
            for k = 1,flatParameters:nElement() do
                if maskParameters[k] == 0 then
                    counter = counter + 1
                    flatUsedParameters[counter] = flatParameters[counter+cumSumOfHoles[k]]
                end
            end
            assert (counter == nUsedParameters)
        end
        return flatUsedParameters
    end

    -- flatten parameters and gradients
    local flatParameters = flatten(parameters)
    local flatGradParameters = flatten(gradParameters)

    -- return new flat vector that contains all discrete parameters
    return flatParameters, flatGradParameters
end




function model_utils.clone_many_times(net, T)
    local clones = {}

    local params, gradParams
    if net.parameters then
        params, gradParams = net:parameters()
        if params == nil then
            params = {}
        end
    end

    local paramsNoGrad
    if net.parametersNoGrad then
        paramsNoGrad = net:parametersNoGrad()
    end

    local mem = torch.MemoryFile("w"):binary()
    mem:writeObject(net)

    for t = 1, T do
        -- We need to use a new reader for each clone.
        -- We don't want to use the pointers to already read objects.
        local reader = torch.MemoryFile(mem:storage(), "r"):binary()
        local clone = reader:readObject()
        reader:close()

        if net.parameters then
            local cloneParams, cloneGradParams = clone:parameters()
            local cloneParamsNoGrad
            for i = 1, #params do
                cloneParams[i]:set(params[i])
                cloneGradParams[i]:set(gradParams[i])
            end
            if paramsNoGrad then
                cloneParamsNoGrad = clone:parametersNoGrad()
                for i =1,#paramsNoGrad do
                    cloneParamsNoGrad[i]:set(paramsNoGrad[i])
                end
            end
        end

        clones[t] = clone
        collectgarbage()
    end

    mem:close()
    return clones
end

function model_utils.init_with_pretrained(model, pretrained_model)
	model.modules[1].weight = pretrained_model.modules[1].weight
	model.modules[1].bias = pretrained_model.modules[1].bias
              
	model.modules[3].weight = pretrained_model.modules[3].weight
	model.modules[3].bias = pretrained_model.modules[3].bias
              
	model.modules[5].weight = pretrained_model.modules[5].weight
	model.modules[5].bias = pretrained_model.modules[5].bias
              
	model.modules[7].weight = pretrained_model.modules[7].weight
	model.modules[7].bias = pretrained_model.modules[7].bias

	model.modules[9].weight = pretrained_model.modules[9].weight
	model.modules[9].bias = pretrained_model.modules[9].bias
              
	model.modules[11].weight = pretrained_model.modules[11].weight
	model.modules[11].bias = pretrained_model.modules[11].bias
              
	model.modules[13].weight = pretrained_model.modules[13].weight
	model.modules[13].bias = pretrained_model.modules[13].bias
	            
	model.modules[15].weight = pretrained_model.modules[15].weight
	model.modules[15].bias = pretrained_model.modules[15].bias
	return model
end

function model_utils.remove_outputs(model)
	 for i,node in ipairs(model.forwardnodes) do
		 if node.data.module then
			 node.data.module.output = torch.Tensor()
		 end
	 end
end

function model_utils.cleanup_resnet_50(net)
	out_layer_idx={1,2,3,4,9,10,11}
	layer_idx={5,6,7,8}
	bottleneck_idx={3,4,6,3}
	for i=1,#out_layer_idx do
		m = net.modules[out_layer_idx[i]]
		if m.output ~= nil then
			m.output = torch.Tensor()
		end
--		if m.gradInput ~= nil then
--			m.gradInput = torch.Tensor()
--		end
		if m.finput ~= nil then
			m.finput = torch.Tensor()
		end
	end
	for i=1,#layer_idx do
		layer = net.modules[layer_idx[i]]
		if layer.output ~= nil then
			layer.output = torch.Tensor()
		end
--		if layer.gradInput ~= nil then
--			layer.gradInput = torch.Tensor()
--		end
		if layer.finput ~= nil then
			layer.finput = torch.Tensor()
		end
		for j=1,bottleneck_idx[i] do
			bottleneck = layer.modules[j]
			if bottleneck.output ~= nil then
				bottleneck.output = torch.Tensor()
			end
--			if bottleneck.gradInput ~= nil then
--				bottleneck.gradInput = torch.Tensor()
--			end
			if bottleneck.finput ~= nil then
				bottleneck.finput = torch.Tensor()
			end
			basic = bottleneck.modules[1]
			if basic.output ~= nil then
				basic.output = torch.Tensor()
			end
--			if basic.gradInput ~= nil then
--				basic.gradInput = torch.Tensor()
--			end
			if basic.finput ~= nil then
				basic.finput = torch.Tensor()
			end
			bb = basic.modules[1]
			if bb.output ~= nil then
				bb.output = torch.Tensor()
			end
--			if bb.gradInput ~= nil then
--				bb.gradInput = torch.Tensor()
--			end
			if bb.finput ~= nil then
				bb.finput = torch.Tensor()
			end
			shortcut = basic.modules[2]
			if shortcut.output ~= nil then
				shortcut.output = torch.Tensor()
			end
--			if shortcut.gradInput ~= nil then
--				shortcut.gradInput = torch.Tensor()
--			end
			if shortcut.finput ~= nil then
				shortcut.finput = torch.Tensor()
			end
			for k=1,7 do
				if bb.modules[k].output ~= nil then
					bb.modules[k].output=torch.Tensor()
				end
--				if bb.modules[k].gradInput ~= nil then
--					bb.modules[k].gradInput =torch.Tensor()
--				end
				if bb.modules[k].finput ~= nil then
					bb.modules[k].finput =torch.Tensor()
				end
			end
			if bottleneck.modules[2].output ~= nil then
				bottleneck.modules[2].output=torch.Tensor()
			end
--			if bottleneck.modules[2].gradInput ~= nil then
--				bottleneck.modules[2].gradInput=torch.Tensor()
--			end
			if bottleneck.modules[2].finput ~= nil then
				bottleneck.modules[2].finput =torch.Tensor()
			end
			if bottleneck.modules[3].output ~= nil then
				bottleneck.modules[3].output=torch.Tensor()
			end
--			if bottleneck.modules[3].gradInput ~= nil then
--				bottleneck.modules[3].gradInput=torch.Tensor()
--			end
			if bottleneck.modules[3].finput ~= nil then
				bottleneck.modules[3].finput =torch.Tensor()
			end
			if bottleneck.modules[4].output ~= nil then
				bottleneck.modules[4].output=torch.Tensor()
			end
--			if bottleneck.modules[4].gradInput ~= nil then
--				bottleneck.modules[4].gradInput=torch.Tensor()
--			end
			if bottleneck.modules[4].finput ~= nil then
				bottleneck.modules[4].finput =torch.Tensor()
			end
		end
	end
end

--function model_utils.cleanup_model(node)
--  if node.output ~= nil then
--    node.output = torch.Tensor()
--  end
--  if node.gradInput ~= nil then
--    node.gradInput = torch.Tensor()
--  end
--  if node.finput ~= nil then
--    node.finput = torch.Tensor()
--  end
--  -- Recurse on nodes with 'modules'
--  if (node.modules ~= nil) then
--    if (type(node.modules) == 'table') then
--      for i = 1, #node.modules do
--        local child = node.modules[i]
--        model_utils.cleanup_model(child)
--      end
--    end
--  end
--  collectgarbage()
--end

return model_utils
