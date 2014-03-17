
recog = struct2cell(load('network_v2')){1};

function retval = h(theta, x)
	% get output of first two neurons
	for i = 1:size(theta)(2)
		x = g(theta{i}, cat(2, [1], x));
	end
	retval = x;
end

function retval = g(theta, x)
	retval =  (1 ./ (1 + (e .^ -(x * theta'))));
end


function retval = J(theta, training)
	tx = training{1};
	ty = reshape(cell2mat(training{2}), columns(training{2}{1}), length(training{2}))';
	y1 = (ty .* log(reshape(cell2mat(cellfun(@(x) h(theta, x), tx, 'UniformOutput', false)), columns(ty), rows(ty))'));
	y0 = (1 - ty) .* log(1 - reshape(cell2mat(cellfun(@(x) h(theta, x), tx, 'UniformOutput', false)), columns(ty), rows(ty))');
	retval = -(1 / length(training{2})) * (sum(sum(y0') + sum(y1'))); % sum ks before is (fold y0 and y1)
	% no regularization for now, but that would be on the end of the summation above...
end


function retval = train(theta, training, learningRate=0.01, iterations=10)
	%learningRate = 0.001;
	tx = training{1};
	ty = training{2};
	for p = 1:iterations % gradient decent event loop
		thetas = getThetas(theta); % thetas as an array of models, we can use it to calculate delta
		capDelta = {0};
		J(theta, training)
		for i = 1:length(training{1}) % iterate through training set
			y = ty{i};
			x = tx{i};
			% split network into layers
		
			delta = [];
			deltas = {};
			activations = {x};
			for l = length(thetas):-1:1 % iterate through layers
				% get delta
				if (l == length(thetas))
					delta = (h(thetas{l}, x) - y)';
					activations{l + 1} = h(thetas{l}, x);
				else
					% the long way
					% we have delta(l + 1), theta(l) and a(l)
					thetaj = theta{l + 1}(:, 2:(columns(theta{l + 1}))); % theta of layer l + 1
					delta = (thetaj' * delta) .* (h(thetas{l}, x) .* (1 - h(thetas{l}, x)))';
					activations{l + 1} = h(thetas{l}, x);
					% delta be a vector the with length K, K being the number of nurons in layer l
		
		
				end
				deltas{l} = delta;
			end
			for l = 1:length(deltas)
				
				deltas{l} = deltas{l} * cat(2, [1], activations{l});
			end
			% update capdelta
			if (capDelta{1} == 0)
				capDelta = deltas;
			else
				for l = 1:length(capDelta)
					capDelta{l} += deltas{l};
				end
			end
		end
		temp = {}; % temp is used to check if J(x) is diverging
		for l = 1:length(capDelta)
			temp{l} = theta{l} - ((1 / length(training)) * capDelta{l}) * learningRate;
		end
		if (J(theta, training) > J(temp, training))
			theta = temp; % update network
		else
			disp('cost function divergant, did not update');
			break;
		end
	end
	J(theta, training)
	retval = theta;
end


function retval = getThetas(theta)
	thetas = {};
	for l = 1:length(theta)
		current = {};
		for m = 1:l
			current{m} = theta{m};
		end
		thetas{l} = current;
	end
	retval = thetas;
	return;
end


function retval =  buildNet(structure)
	net = {};
	inputs = structure(1);
	for i = 2:columns(structure)
		net{i - 1} = randn(structure(:, i), structure(:, i - 1) + 1);
	end
	retval = net;
	return;
end

function retval =  buildTrainingSet(in, out)
	trainingX = {};
	trainingY = {};
	for i = 1:rows(in)
		trainingX{i} = in(i, :);
		trainingY{i} = out(i, :);
	end
	retval = {trainingX, trainingY};
	return;
end


function retval = predict(network, input)
	prediction = h(network, input);
	retval = (find(prediction == max(prediction))) - 1;
end



in = load('imagedata_v2.dat');
out = load('output_v2.dat');
%net = buildNet([20 20 10]);
training = buildTrainingSet(in, out);