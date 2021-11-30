classdef DAGNetwork < nnet.internal.cnn.TrainableNetwork
    % DAGNetwork   Class for a directed acyclic graph network
    
    %   Copyright 2017-2018 The MathWorks, Inc.
    
    properties
        % Layers of the optimized LayerGraph
        Layers
        
        % Connections of the optimized LayerGraph
        Connections
    end
    
    properties(SetAccess = private)
        % NumInputLayers   The number of input layers for this network
        %   The number of input layers for this network. This property is
        %   public because it is needed by the other DAGNetwork object.
        NumInputLayers
        
        % NumOutputLayers   The number of output layers for this network
        %   The number of output layers for this network. This property is
        %   public because it is needed by the other DAGNetwork object.
        NumOutputLayers
        
        % InputLayerIndices   The indices of the input layers
        %   The indices of the input layers.
        InputLayerIndices
        
        % OutputLayerIndices   The indices of the output layers
        %   The indices of the output layers.
        OutputLayerIndices
        
        % InputSizes   Sizes of the network inputs
        InputSizes
        
        % Outputsizes   Sizes of the network outputs
        OutputSizes
        
        % TopologicalOrder  Topological order of layers in the
        % OriginalLayers array
        TopologicalOrder
        
        % NetworkOptimizer  Object defining how layers have been optimized
        % on construction of the network
        NetworkOptimizer
        
        % SortedConnections  Connections matrix for the externally
        % visible layer graph (sorted but unoptimized). This is stored
        % rather than recomputed on access for performance reasons.
        SortedConnections
    end
    
    properties(Access = private)
        
        % LayerGraphExecutionInfo  Stores relationships between layers and
        % inputs and outputs stored in an activations buffer during
        % propagation
        LayerGraphExecutionInfo
        
        % Sizes   The output sizes for each activation
        Sizes
        
        % LayerOutputSizes  The output sizes for each layer
        LayerOutputSizes
        
        % UseGpu  Records if execution is taking place on the GPU, used to
        % ensure data is in the right place before propagation
        UseGpu = false
    end
    
    properties( Dependent, Access = private )
        
        % NumLayers
        NumLayers
        
        % NumActivations   Number of activations
        %   The number of unique output activations in the network.
        NumActivations
        
        % EdgeTable  Used for efficient graph execution
        EdgeTable
        
        % ListOfBufferInputIndices   List of buffer input indices
        ListOfBufferInputIndices
        
        % ListOfBufferOutputIndices   List of buffer output indices
        ListOfBufferOutputIndices
        
        % ListOfBufferIndicesForClearingForward   List of buffer entries
        % that can be cleared as we move forward through the network
        ListOfBufferIndicesForClearingForward
        
        % ListOfBufferIndicesForClearingBackward   List of buffer entries
        % that can be cleared as we move backward through the network
        ListOfBufferIndicesForClearingBackward
    end
    
    properties (Dependent, SetAccess = private)
        % LearnableParameters    Learnable parameters of the networks
        %                        (vector of nnet.internal.cnn.layer.LearnableParameter)
        LearnableParameters
        
        % LayerGraph    The optimized layer graph
        %   This contains an internal layer graph with the most recent
        %   learnable parameters and is created using the Layers and
        %   Connections properties.
        LayerGraph
        
        % OriginalLayers  Layers in the original order, unoptimized
        OriginalLayers
        
        % OriginalConnections  Connections in the original order,
        % unoptimized
        OriginalConnections
        
        % SortedLayers   Unoptimized Layers in a topologically sorted order
        SortedLayers
        
        % NumInputs   The number of input layers in the network
        NumInputs
        
        % NumOutputs   The number of output layers in the network
        NumOutputs
    end
    
    properties (Dependent)
        % InputLayers   The input layers for this network
        InputLayers
        
        % OutputLayers   The output layers for this network
        OutputLayers
    end
    
    methods
        function learnableParameters = get.LearnableParameters(this)
            learnableParameters = [];
            for el = 1:this.NumLayers
                thisParam = this.Layers{el}.LearnableParameters;
                if ~isempty( thisParam )
                    learnableParameters = [learnableParameters thisParam]; %#ok<AGROW>
                end
            end
        end
        
        function inputLayers = get.InputLayers(this)
            inputLayers = this.Layers(this.InputLayerIndices);
        end
        
        function outputLayers = get.OutputLayers(this)
            outputLayers = this.Layers(this.OutputLayerIndices);
        end
        
        function this = set.OutputLayers(this, val)
            this.Layers(this.OutputLayerIndices) = val;
        end
        
        function layerGraph = get.LayerGraph(this)
            layerGraph = makeTrainedLayerGraph(this);
        end
        
        function originalLayers = get.OriginalLayers(this)
            originalLayers = nnet.internal.cnn.LayerGraph.sortedToOriginalLayers(this.SortedLayers, this.TopologicalOrder);
        end
        
        function originalConnections = get.OriginalConnections(this)
            originalConnections = nnet.internal.cnn.LayerGraph.sortedToOriginalConnections(this.SortedConnections, this.TopologicalOrder);
            originalConnections = sortrows(originalConnections);
        end
        
        function sortedLayers = get.SortedLayers(this)
            numOriginalLayers = numel(this.TopologicalOrder);
            sortedLayers = cell(numOriginalLayers, 1);
            for l = 1:numel(this.Layers)
                thisLayer = this.Layers{l};
                originalLayerIndices = this.NetworkOptimizer.mapToOriginal(l);
                if isa(thisLayer, 'nnet.internal.cnn.layer.FusedLayer')
                    sortedLayers(originalLayerIndices) = thisLayer.OriginalLayers(:);
                else
                    sortedLayers{originalLayerIndices} = thisLayer;
                end
            end
        end
        
        function val = get.NumInputs(this)
            val = numel(this.InputLayers);
        end
        
        function val = get.NumOutputs(this)
            val = numel(this.OutputLayers);
        end
        
        function numActivations = get.NumActivations(this)
            numActivations = this.LayerGraphExecutionInfo.NumActivations;
        end
        
        function edgeTable = get.EdgeTable(this)
            edgeTable = this.LayerGraphExecutionInfo.EdgeTable;
        end
        
        function listOfBufferInputIndices = get.ListOfBufferInputIndices(this)
            listOfBufferInputIndices = this.LayerGraphExecutionInfo.ListOfBufferInputIndices;
        end
        
        function listOfBufferOutputIndices = get.ListOfBufferOutputIndices(this)
            listOfBufferOutputIndices = this.LayerGraphExecutionInfo.ListOfBufferOutputIndices;
        end
        
        function listOfBufferIndicesForClearingForward = get.ListOfBufferIndicesForClearingForward(this)
            listOfBufferIndicesForClearingForward = this.LayerGraphExecutionInfo.ListOfBufferIndicesForClearingForward;
        end
        
        function listOfBufferIndicesForClearingBackward = get.ListOfBufferIndicesForClearingBackward(this)
            listOfBufferIndicesForClearingBackward = this.LayerGraphExecutionInfo.ListOfBufferIndicesForClearingBackward;
        end
        
        function val = get.NumLayers(this)
            val = numel(this.Layers);
        end
        
        function [index, offset] = getInternalForExternalLayerIndex(this, index)
            index = this.TopologicalOrder(index);
            [index, offset] = this.NetworkOptimizer.mapFromOriginal(index);
        end
    end
    
    methods
        function this = DAGNetwork(sortedLayerGraph, topologicalOrder, networkOptimizer)
            %DAGNetwork - Create an internal DAGNetwork.
            %   this = DAGNetwork(sortedLayerGraph, topologicalOrder)
            %   creates an internal DAGNetwork. Input sortedLayerGraph is
            %   an internal LayerGraph containing a topologically sorted
            %   array of internal layers and input topologicalOrder is a
            %   vector representing the indices of the sorted internal
            %   layers in the original (unsorted) array of internal layers.
            
            % Save original connections. Rest of layer graph will be
            % optimized
            this.SortedConnections = sortedLayerGraph.Connections;
            this.TopologicalOrder = topologicalOrder;
            
            % Optimize layer graph
            if nargin < 3
                networkOptimizer = nnet.internal.cnn.optimizer.DefaultNetworkOptimizer();
            end
            if isempty( networkOptimizer )
                networkOptimizer = nnet.internal.cnn.optimizer.NoOpNetworkOptimizer();
            end
            sortedLayerGraph = networkOptimizer.optimize(sortedLayerGraph);
            this.NetworkOptimizer = networkOptimizer;
            
            % Get sorted internal layers based on topological order
            sortedInternalLayers = sortedLayerGraph.Layers;
            
            this.Layers = sortedInternalLayers;
            this.LayerGraphExecutionInfo = nnet.internal.cnn.util.LayerGraphExecutionInfo(sortedLayerGraph);
            
            this.NumInputLayers = iCountInputLayers(sortedInternalLayers);
            this.NumOutputLayers = iCountOutputLayers(sortedInternalLayers);
            this.InputLayerIndices = iGetInputLayerIndices(sortedInternalLayers);
            this.OutputLayerIndices = iGetOutputLayerIndices(sortedInternalLayers);
            
            this = inferSizes(this);
            this.InputSizes = iGetInputSizes(this.LayerOutputSizes, ...
                this.InputLayerIndices);
            this.OutputSizes = iGetOutputSizes(this.LayerOutputSizes, ...
                this.OutputLayerIndices);
            
            % Save the internal connections. A layer graph with the most
            % recent values of learnable parameters can be accessed using
            % the LayerGraph property.
            this.Connections = iExternalToInternalConnections(this.EdgeTable);
        end
        
        function this = optimizeNetwork( this, networkOptimizer )
            % Take an existing network and apply a new optimizer to its
            % layer graph. If networkOptimizer is empty, revert to
            % original, unoptimized layer graph. If missing, use the
            % default optimizer.
            if nargin < 2
                networkOptimizer = nnet.internal.cnn.optimizer.DefaultNetworkOptimizer;
            end
            sortedUnoptimizedLayerGraph = iMakeInternalLayerGraph( this.SortedLayers, this.SortedConnections );
            this = nnet.internal.cnn.DAGNetwork( sortedUnoptimizedLayerGraph, this.TopologicalOrder, networkOptimizer );
        end
        
        function [activationsBuffer, memoryBuffer, layerIsLearning] = forwardPropagationWithMemory(this, X, Y, fuzzy_flag)
            % Forward propagation used by training. Note, this version
            % retains activations and memory, but deletes any that won't be
            % needed for backpropagation.
            
            listOfBufferOutputIndices = this.ListOfBufferOutputIndices;
            listOfBufferInputIndices = this.ListOfBufferInputIndices;
            listOfBufferIndicesForClearingForward = this.ListOfBufferIndicesForClearingForward;
            inputLayerIndices = this.InputLayerIndices;
            
            % Wrap X in cell if needed
            X = iWrapInCell(X);
            
            % Allocate space for the activations.
            activationsBuffer = cell(this.NumActivations,1);
            memoryBuffer = cell(this.NumActivations,1);
            
            % We can recover GPU memory by gathering the current
            % intermediate activations and memory cell arrays back to the
            % host.
            function gatherLayerOutputsAndMemory()
                activationsBuffer = iGatherGPUCell(activationsBuffer);
                memoryBuffer = iGatherGPUCell(memoryBuffer);
            end
            recoveryStrategies = {@gatherLayerOutputsAndMemory};
            
            layerIsLearning = false(this.NumLayers, 1);
            for i = 1:this.NumLayers
                % Mark whether this layer can learn, for backpropagation
                % optimisation
                thisLayer = this.Layers{i};
                learnablesThisLayer = thisLayer.LearnableParameters;
                layerIsLearning(i) = ~isempty(learnablesThisLayer) && any([ learnablesThisLayer.LearnRateFactor ]);
                
                if any(i == inputLayerIndices)
                    [~, currentInputLayer] = find(this.InputLayerIndices == i);
                    XForThisLayer = X{currentInputLayer};
                else
                    XForThisLayer = iGetTheseActivationsFromBuffer( ...
                        activationsBuffer, ...
                        listOfBufferInputIndices{i});
                end
                XForThisLayer = this.moveToEnvironment( XForThisLayer );
                [outputActivations, memory] = iExecuteWithStagedGPUOOMRecovery( ...
                    @() this.Layers{i}.forward( XForThisLayer ), ...
                    2, recoveryStrategies, i );
                
                bufferOutputIndices = listOfBufferOutputIndices{i};
                activationsBuffer = iAssignActivationsToBuffer( ...
                    activationsBuffer, ...
                    bufferOutputIndices, ...
                    outputActivations);
                
                memoryBuffer = iAssignMemoryToBuffer( ...
                    memoryBuffer, ...
                    bufferOutputIndices, ...
                    memory);
                
                % Throw away data from layers that aren't going to be
                % visited on the backward pass
                if ~any(layerIsLearning) && i > 1
                    indicesToClear = listOfBufferIndicesForClearingForward{i};
                    activationsBuffer = iClearActivationsFromBuffer( ...
                        activationsBuffer, indicesToClear );
                    memoryBuffer = iClearActivationsFromBuffer( ...
                        memoryBuffer, indicesToClear );
                end
                
                
                %--------------------------------------------------------------------------
                % change the Probability in Fully Connected Layer
                
                %  Six-classification:
                %        **Normal**
                % 0- incomplete
                % 1-negative
                %        **Benign**
                % 2-benign findings
                % 3-probably benign
                %        **Malignant**
                % 4-suspicious abnormality
                % 5-highly suspicious of malignancy
                %        **Non**
                % 6-known biopsy with proven malignancy
                
                % So, 0 and 1
                
%                 threshhold = 1/4;
%                 if fuzzy_flag >=8 && i == this.NumLayers - 1
%                     Vp = [];  % the sum of score*probability over the sum of score
%                     Pvp = [];  % the probability of Vp
%                     Prob_fully_recify = [];  % the recified Probability
%                     Prob_fully_recify = []; % recify the Pribability in FullyConnected Layer
%                     
%                     Prob_fully = squeeze(activationsBuffer{end-1});
%                     [catorgary_num batch_size]= size(Prob_fully);
%                     
%                     Catergary = [1:catorgary_num]'; % the score of each item
%                     
%                     for batch_num = 1:batch_size
%                         sum_score = [];
%                         Vp = [];
%                         Pvp = [];
%                         
%                         % if there is any possibility that two adjacent
%                         % probabilities are greater than 1/3, which means
%                         % this image is something between these two
%                         % degrees.
%                         Prob_fully_recify = Prob_fully;
%                         if (Prob_fully(1,batch_num) > 1/3 && Prob_fully(1,batch_num) < 2/3) && (Prob_fully(3,batch_num) > 1/3 && Prob_fully(3,batch_num) < 2/3)
%                             sum_score = (Prob_fully(1,batch_num) * 2  + Prob_fully(3,batch_num) * 3);
%                             Vp = sum_score / (Prob_fully(1,batch_num) + Prob_fully(3,batch_num));
%                             Pvp = sum_score / (1 + 3);
%                             
%                             Prob_fully_recify(3,batch_num) = abs(Vp - 1)*sum_score;
%                             Prob_fully_recify(1,batch_num) = abs(Vp - 3)*sum_score;
%                         end
%                         if (Prob_fully(4,batch_num) > 1/3 && Prob_fully(4,batch_num) < 2/3) && (Prob_fully(3,batch_num) > 1/3 && Prob_fully(3,batch_num) < 2/3)
%                             sum_score = (Prob_fully(3,batch_num) * 3  + Prob_fully(4,batch_num) * 4);
%                             Vp = sum_score / (Prob_fully(3,batch_num) + Prob_fully(4,batch_num));
%                             Pvp = sum_score / (3 + 4);
%                             Prob_fully_recify(3,batch_num) = abs(Vp - 4)*sum_score;
%                             Prob_fully_recify(4,batch_num) = abs(Vp - 3)*sum_score;
%                         end
%                         if (Prob_fully(1,batch_num) > 1/3 && Prob_fully(1,batch_num) < 2/3) && (Prob_fully(3,batch_num) > 1/3 && Prob_fully(3,batch_num) < 2/3)
%                             % maintain the same condition
%                         end
%                     end
%                     
%                     % revise the Probability
%                     Prob_fully_recify = reshape(Prob_fully_recify,[1, 1, 4, batch_size]);
%                     activationsBuffer(i) = {Prob_fully_recify};
%                 end
                
                
                
                threshhold = 1/5;
                if fuzzy_flag >= 90 && i == this.NumLayers - 1
                    Vp = [];  % the sum of score*probability over the sum of score
                    Pvp = [];  % the probability of Vp
                    Prob_fully_recify = [];  % the recified Probability, recify the Pribability in FullyConnected Layer
                    
                    Prob_fully = squeeze(activationsBuffer{end-1});
                    Y_lable = squeeze(Y{1});
                    [catorgary_num batch_size]= size(Prob_fully);
                    
                    Catergary = [1:catorgary_num]'; % the score of each item
                    
                    Prob_fully_recify = Prob_fully;
                    
                    for batch_num = 1:batch_size
                        sum_score = 0;
                        Vp = [];
                        Pvp = [];
                        
                        % if there is any possibility that two adjacent
                        % probabilities are greater than 1/3, which means
                        % this image is something between these two
                        % degrees.
                        
                        for cate_num =2:5
                            
                            if ( max(Prob_fully_recify(:,batch_num)) == Prob_fully_recify(cate_num-1,batch_num)) || (max(Prob_fully_recify(:,batch_num))==Prob_fully_recify(cate_num,batch_num)) || (max(Prob_fully_recify(:,batch_num)) == Prob_fully_recify(cate_num+1,batch_num))
                                
                                if cate_num ==5
                                    
                                    if (Prob_fully_recify(cate_num-1,batch_num) > threshhold) && (Prob_fully_recify(cate_num,batch_num) > threshhold) && (Prob_fully_recify(cate_num+1,batch_num) < threshhold) % Cate(N-1), Cate(N), Cate(N+1)
                                        sum_score = sum(Prob_fully_recify(cate_num-2:cate_num+1,batch_num) .* Catergary(cate_num-2:cate_num+1));%cate_num  + Prob_fully(cate_num+1,batch_num) * (cate_num+1));
                                        if sum_score ~= 0
                                            Vp = sum_score;
                                            Pvp = sum_score / (cate_num + cate_num-1);
                                            
                                            if ( Vp < 0.95*(cate_num) && Vp > 1.05*(cate_num-1))
                                                if ( Vp < 0.99*((cate_num-1)+cate_num)/2 || Vp > 1.01*((cate_num-1)+cate_num)/2 )
                                                    
                                                    Prob_fully_recify(cate_num,batch_num) = (Vp - (cate_num-1))*(Prob_fully(cate_num-1,batch_num) + Prob_fully(cate_num,batch_num));
                                                    Prob_fully_recify(cate_num-1,batch_num) = ((cate_num) - Vp)*(Prob_fully(cate_num-1,batch_num) + Prob_fully(cate_num,batch_num));
                                                    break;
                                                end
                                            end
                                        end
                                    else if (Prob_fully_recify(cate_num-1,batch_num) < threshhold) && (Prob_fully_recify(cate_num,batch_num) > threshhold) && (Prob_fully_recify(cate_num+1,batch_num) > threshhold) % Cate(N), Cate(N+1), Cate(N+2)
                                            sum_score = sum(Prob_fully_recify(cate_num:cate_num+1,batch_num) .* Catergary(cate_num:cate_num+1));
                                            if sum_score ~= 0
                                                Vp = sum_score;
                                                Pvp = sum_score / (cate_num + cate_num+1);
                                                
                                                if ( Vp < 0.99*(cate_num+1) && Vp > 1.01*(cate_num))
                                                    if ( Vp < 0.99*((cate_num+1)+cate_num)/2 || Vp > 1.01*((cate_num+1)+cate_num)/2 )
                                                        
                                                        Prob_fully_recify(cate_num+1,batch_num) = (Vp - cate_num)*(Prob_fully(cate_num,batch_num) + Prob_fully(cate_num+1,batch_num));
                                                        Prob_fully_recify(cate_num,batch_num) = ((cate_num+1) - Vp)*(Prob_fully(cate_num,batch_num) + Prob_fully(cate_num+1,batch_num));
                                                        break;
                                                    end
                                                end
                                            else if (Prob_fully_recify(cate_num-1,batch_num) > threshhold) && (Prob_fully_recify(cate_num,batch_num) > threshhold) && (Prob_fully_recify(cate_num+1,batch_num) > threshhold) % Cate(N-1), Cate(N), Cate(N+1)
                                                    
                                                end
                                            end
                                        end
                                    end
                                end
                                
                                if cate_num == 3
                                    if (Prob_fully_recify(cate_num-1,batch_num) > threshhold) && (Prob_fully_recify(cate_num,batch_num) > threshhold) && (Prob_fully_recify(cate_num+1,batch_num) < threshhold) % Cate(N-1), Cate(N), Cate(N+1)
                                        sum_score = sum(Prob_fully_recify(cate_num-1:cate_num,batch_num) .* Catergary(cate_num-1:cate_num));%cate_num  + Prob_fully(cate_num+1,batch_num) * (cate_num+1));
                                        if sum_score ~= 0
                                            Vp = sum_score;
                                            Pvp = sum_score / (cate_num + cate_num-1);
                                            
                                            if ( Vp < 0.95*(cate_num) && Vp > 1.05*(cate_num-1))
                                                if ( Vp < 0.99*((cate_num-1)+cate_num)/2 || Vp > 1.01*((cate_num-1)+cate_num)/2 )
                                                    
                                                    Prob_fully_recify(cate_num,batch_num) = (Vp - (cate_num-1))*(Prob_fully(cate_num-1,batch_num) + Prob_fully(cate_num,batch_num));
                                                    Prob_fully_recify(cate_num-1,batch_num) = ((cate_num) - Vp)*(Prob_fully(cate_num-1,batch_num) + Prob_fully(cate_num,batch_num));
                                                    break;
                                                end
                                            end
                                        end
                                    else if (Prob_fully_recify(cate_num-1,batch_num) < threshhold) && (Prob_fully_recify(cate_num,batch_num) > threshhold) && (Prob_fully_recify(cate_num+1,batch_num) > threshhold) % Cate(N), Cate(N+1), Cate(N+2)
                                            sum_score = sum(Prob_fully_recify(cate_num-1:cate_num+2,batch_num) .* Catergary(cate_num-1:cate_num+2));
                                            if sum_score ~= 0
                                                Vp = sum_score;
                                                Pvp = sum_score / (cate_num + cate_num+1);
                                                
                                                if ( Vp < 0.95*(cate_num+1) && Vp > 1.05*(cate_num))
                                                    if ( Vp < 0.99*((cate_num+1)+cate_num)/2 || Vp > 1.01*((cate_num+1)+cate_num)/2 )
                                                        
                                                        Prob_fully_recify(cate_num+1,batch_num) = (Vp - cate_num)*(Prob_fully(cate_num,batch_num) + Prob_fully(cate_num+1,batch_num));
                                                        Prob_fully_recify(cate_num,batch_num) = ((cate_num+1) - Vp)*(Prob_fully(cate_num,batch_num) + Prob_fully(cate_num+1,batch_num));
                                                        break;
                                                    end
                                                end
                                            else if (Prob_fully_recify(cate_num-1,batch_num) > threshhold) && (Prob_fully_recify(cate_num,batch_num) > threshhold) && (Prob_fully_recify(cate_num+1,batch_num) > threshhold) % Cate(N-1), Cate(N), Cate(N+1)
                                                    
                                                end
                                            end
                                        end
                                    end
                                    if cate_num == 4
                                        if (Prob_fully_recify(cate_num-1,batch_num) > threshhold) && (Prob_fully_recify(cate_num,batch_num) > threshhold) && (Prob_fully_recify(cate_num+1,batch_num) < threshhold) % Cate(N-1), Cate(N), Cate(N+1)
                                            sum_score = sum(Prob_fully_recify(cate_num-2:cate_num+1,batch_num) .* Catergary(cate_num-2:cate_num+1));%cate_num  + Prob_fully(cate_num+1,batch_num) * (cate_num+1));
                                            if sum_score ~= 0
                                                Vp = sum_score;
                                                Pvp = sum_score / (cate_num + cate_num-1);
                                                
                                                if ( Vp < 0.95*(cate_num) && Vp > 1.05*(cate_num-1))
                                                    if ( Vp < 0.99*((cate_num-1)+cate_num)/2 || Vp > 1.01*((cate_num-1)+cate_num)/2 )
                                                        
                                                        Prob_fully_recify(cate_num,batch_num) = (Vp - (cate_num-1))*(Prob_fully(cate_num-1,batch_num) + Prob_fully(cate_num,batch_num));
                                                        Prob_fully_recify(cate_num-1,batch_num) = ((cate_num) - Vp)*(Prob_fully(cate_num-1,batch_num) + Prob_fully(cate_num,batch_num));
                                                        break;
                                                    end
                                                end
                                            end
                                        else if (Prob_fully_recify(cate_num-1,batch_num) < threshhold) && (Prob_fully_recify(cate_num,batch_num) > threshhold) && (Prob_fully_recify(cate_num+1,batch_num) > threshhold) % Cate(N), Cate(N+1), Cate(N+2)
                                                sum_score = sum(Prob_fully_recify(cate_num-1:cate_num+2,batch_num) .* Catergary(cate_num-1:cate_num+2));
                                                if sum_score ~= 0
                                                    Vp = sum_score;
                                                    Pvp = sum_score / (cate_num + cate_num+1);
                                                    
                                                    if ( Vp < 0.95*(cate_num+1) && Vp > 1.05*(cate_num))
                                                        if ( Vp < 0.99*((cate_num+1)+cate_num)/2 || Vp > 1.01*((cate_num+1)+cate_num)/2 )
                                                            
                                                            Prob_fully_recify(cate_num+1,batch_num) = (Vp - cate_num)*(Prob_fully(cate_num,batch_num) + Prob_fully(cate_num+1,batch_num));
                                                            Prob_fully_recify(cate_num,batch_num) = ((cate_num+1) - Vp)*(Prob_fully(cate_num,batch_num) + Prob_fully(cate_num+1,batch_num));
                                                            break;
                                                        end
                                                    end
                                                end
                                            else if (Prob_fully_recify(cate_num-1,batch_num) > threshhold) && (Prob_fully_recify(cate_num,batch_num) > threshhold) && (Prob_fully_recify(cate_num+1,batch_num) > threshhold) % Cate(N-1), Cate(N), Cate(N+1)
                                                    
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                    
                    
                    
                    % revise the Probability
                    Prob_fully_recify = reshape(Prob_fully_recify,[1, 1, 6, batch_size]);
                    activationsBuffer(i) = {Prob_fully_recify};
                end
                
                %--------------------------------------------------------------------------
            end
        end
        
        function Y = predict(this, X)
            
            listOfBufferOutputIndices = this.ListOfBufferOutputIndices;
            listOfBufferInputIndices = this.ListOfBufferInputIndices;
            listOfBufferIndicesForClearingForward = this.ListOfBufferIndicesForClearingForward;
            inputLayerIndices = this.InputLayerIndices;
            
            % Wrap X in cell if needed
            X = iWrapInCell(X);
            
            % Allocate space for the activations.
            activationsBuffer = cell(this.NumActivations,1);
            
            % Loop over topologically sorted layers to perform forward
            % propagation. Clear memory when activations are no longer
            % needed.
            for i = 1:this.NumLayers
                thisLayer = this.Layers{i};
                if any(i == inputLayerIndices)
                    [~, currentInputLayer] = find(this.InputLayerIndices == i);
                    
                    outputActivations = thisLayer.predict(X{currentInputLayer});
                else
                    XForThisLayer = iGetTheseActivationsFromBuffer( ...
                        activationsBuffer, ...
                        listOfBufferInputIndices{i});
                    
                    outputActivations = thisLayer.predict(XForThisLayer);
                end
                
                activationsBuffer = iAssignActivationsToBuffer( ...
                    activationsBuffer, ...
                    listOfBufferOutputIndices{i}, ...
                    outputActivations);
                
                activationsBuffer = iClearActivationsFromBuffer( ...
                    activationsBuffer, ...
                    listOfBufferIndicesForClearingForward{i});
            end
            
            % Return activations corresponding to output layers.
            Y = activationsBuffer( ...
                [listOfBufferOutputIndices{this.OutputLayerIndices}] );
        end
        
        
        
        function Z = activations(this, X, layerIndices, layerOutputIndices)
            % activations   Support Fused Layers by calling into
            % activations method if requested output is internal to a
            % FusedLayer
            
            % Convert layerIndices into indices into the optimized layers
            % plus offsets
            [layerIndices, layerOffsets] = this.NetworkOptimizer.mapFromOriginal(layerIndices);
            
            % For fused layers where the user has requested an internal
            % activation, replace that with a request for the inputs to the
            % fused layer. Later we will compute the internal activations
            % for each of those requests.
            %
            % Get the 'true' indices and offsets for layerIndices that are
            % FusedLayers
            internalFusedLayers = cellfun(@iIsAFusedLayer, this.Layers);
            whichInputsAreFusedLayers = ismember(layerIndices, find(internalFusedLayers));
            fusedLayerIndices = layerIndices(whichInputsAreFusedLayers);
            fusedLayerOffsets = layerOffsets(whichInputsAreFusedLayers);
            fusedLayerOutputIndices = layerOutputIndices(whichInputsAreFusedLayers)';
            numFusedLayerOutputs = numel(fusedLayerIndices);
            %
            % Get the 'true' indices and offsets for the non-FusedLayers
            normalLayerIndices = layerIndices(~whichInputsAreFusedLayers);
            normalLayerOutputIndices = layerOutputIndices(~whichInputsAreFusedLayers);
            numNormalLayerIndices = numel(normalLayerIndices);
            %
            % Create a new list of layer indices and output port indices
            % for non-FusedLayers that includes the inputs to FusedLayers
            layerInputConnections = this.LayerGraphExecutionInfo.LayerInputConnections;
            for i = 1:numFusedLayerOutputs
                inputsToFusedLayers = layerInputConnections{fusedLayerIndices(i)};
                inputsToFusedLayers = cat(1, inputsToFusedLayers{:});
                normalLayerIndices = [normalLayerIndices; inputsToFusedLayers(:,1)]; %#ok<AGROW>
                normalLayerOutputIndices = [normalLayerOutputIndices(:); inputsToFusedLayers(:,2)];
            end
            
            % Wrap X in cell if needed
            X = iWrapInCell(X);
            
            % Preparation
            numActivations = this.NumActivations;
            listOfBufferOutputIndices = this.ListOfBufferOutputIndices;
            listOfBufferInputIndices = this.ListOfBufferInputIndices;
            listOfBufferIndicesForClearingForward = this.ListOfBufferIndicesForClearingForward;
            
            % Allocate space for the activations.
            activationsBuffer = cell(numActivations,1);
            
            % Convert layer indices and layer output indices into indices
            % for the activations buffer.
            normalLayerActivationIndices = cellfun( ...
                @(i,o)i(o), listOfBufferOutputIndices(normalLayerIndices), ...
                num2cell(normalLayerOutputIndices) );
            
            % Loop over topologically sorted layers to perform forward
            % propagation. Clear memory when activations are no longer
            % needed.
            maxLayerIndex = max(normalLayerIndices);
            for i = 1:maxLayerIndex
                thisLayer = this.Layers{i};
                if any(i == this.InputLayerIndices)
                    [~, currentInputLayer] = find(this.InputLayerIndices == i);
                    
                    outputActivations = thisLayer.predict(X{currentInputLayer});
                else
                    XForThisLayer = iGetTheseActivationsFromBuffer( ...
                        activationsBuffer, ...
                        listOfBufferInputIndices{i});
                    
                    outputActivations = thisLayer.predict(XForThisLayer);
                end
                
                activationsBuffer = iAssignActivationsToBuffer( ...
                    activationsBuffer, ...
                    listOfBufferOutputIndices{i}, ...
                    outputActivations);
                
                indicesToClear = setdiff( ...
                    listOfBufferIndicesForClearingForward{i}, ...
                    normalLayerActivationIndices);
                
                activationsBuffer = iClearActivationsFromBuffer( ...
                    activationsBuffer, ...
                    indicesToClear);
            end
            
            % Now compute the activations for internal fused layers
            fusedLayerActivations = cell(numFusedLayerOutputs,1);
            for j = 1:numFusedLayerOutputs
                % Get location of inputs and outputs
                layerIndex = fusedLayerIndices(j);
                layerOffset = fusedLayerOffsets(j);
                layerOutputIndex = fusedLayerOutputIndices(j);
                inputsToFusedLayers = layerInputConnections{layerIndex};
                inputsToFusedLayers = cat(1, inputsToFusedLayers{:});
                
                % Get input activations for this FusedLayer
                bufferIndicesForAllOutputsFromInputs = listOfBufferOutputIndices(inputsToFusedLayers(:,1));
                bufferIndicesForInputs = cellfun( @(x,indices)x(indices), ...
                    bufferIndicesForAllOutputsFromInputs, num2cell(inputsToFusedLayers(:,2)) );
                XForThisLayer = iGetTheseActivationsFromBuffer( ...
                    activationsBuffer, bufferIndicesForInputs );
                
                % Call into
                fusedLayerActivations{j} = activations( this.Layers{layerIndex}, ...
                    XForThisLayer, layerOffset, layerOutputIndex );
            end
            
            % Reassemble the output activations from the normal layers and
            % the fused layers, in the order requested
            normalLayerActivationIndices = normalLayerActivationIndices(1:numNormalLayerIndices);
            Z(~whichInputsAreFusedLayers) = activationsBuffer(normalLayerActivationIndices);
            Z(whichInputsAreFusedLayers) = fusedLayerActivations;
        end
        
        function [gradients, predictions, states] = computeGradientsForTraining( ...
                this, X, Y, needsStatefulTraining, propagateState, fuzzy_flag)
            % computeGradientsForTraining    Computes the gradients of the
            % loss with respect to the learnable parameters, from the
            % network input and response. This is used during training to
            % avoid the need to store intermediate activations and
            % derivatives any longer than is necessary.
            %
            % Inputs
            %   X                      - an array containing the data
            %   Y                      - expected responses
            %   needsStatefulTraining  - logical scalar for each layer
            %                            marking whether the layer needs
            %                            stateful training or not. Note
            %                            that DAG does not support stateful
            %                            training, so each element of this
            %                            vector should be false.
            %   propagateState         - logical scalar marking whether
            %                            state needs to be propagated or
            %                            not. Note that DAG does not
            %                            support stateful training, so this
            %                            value should be false.
            %
            % Output
            %   gradients   - cell array of gradients with one element for
            %                 each learnable parameter array
            %   predictions - the output from the last layer, needs to be
            %                 preserved during training to report progress
            %   states      - cell array of state information needed to
            %                 update layer states after gradient
            %                 computation. Note that DAG does not support
            %                 stateful training, so this is always empty.
            
            % DAG currently does not support stateful training. Assert that
            % the user has not requested stateful training.
            assert(all(~needsStatefulTraining));
            assert(~propagateState);
            states = {};
            
            % Wrap X and Y in cell if needed
            X = iWrapInCell(X);
            Y = iWrapInCell(Y);
            
            % Do forward and get all activations
            [activationsBuffer, memoryBuffer, layerIsLearning] = this.forwardPropagationWithMemory(X, Y, fuzzy_flag);
            
            % Set up the backpropagation function, which calls backwards on
            % each layer and then discards the activations and memory when
            % they are no longer needed
            dLossdXBuffer = cell(this.NumActivations,1);
            function dLossdW = efficientBackProp(currentLayer)
                
                % Preparation
                thisLayer = this.Layers{currentLayer};
                bufferInputIndices = this.ListOfBufferInputIndices{currentLayer};
                bufferOutputIndices = this.ListOfBufferOutputIndices{currentLayer};
                learnablesThisLayer = thisLayer.LearnableParameters;
                dLossdW = cell(size(learnablesThisLayer));
                
                % Output layers
                if any(currentLayer == this.OutputLayerIndices)
                    % Perform backpropagation for an output layer
                    ZForThisLayer = this.moveToEnvironment( iGetTheseActivationsFromBuffer( ...
                        activationsBuffer, bufferOutputIndices) );
                    [~, currentInputLayer] = find(this.OutputLayerIndices == currentLayer);
                    TForThisLayer = Y{currentInputLayer};
                    
                    dLossdX = thisLayer.backwardLoss( ...
                        ZForThisLayer, TForThisLayer);
                    
                    dLossdXBuffer = iIncrementActivationsInBuffer( ...
                        dLossdXBuffer, bufferInputIndices, dLossdX);
                    
                    % Input layers
                elseif any(i == this.InputLayerIndices)
                    % Do nothing
                    
                    % Other layers
                else
                    % Perform backpropagation for some other kind of
                    % layer
                    XForThisLayer = iGetTheseActivationsFromBuffer( ...
                        activationsBuffer, bufferInputIndices);
                    ZForThisLayer = iGetTheseActivationsFromBuffer( ...
                        activationsBuffer, bufferOutputIndices);
                    dLossdZ = iGetTheseActivationsFromBuffer( ...
                        dLossdXBuffer, bufferOutputIndices);
                    memory = iGetTheseActivationsFromBuffer( ...
                        memoryBuffer, bufferOutputIndices);
                    
                    % Compute either all gradients or only the activations
                    % gradients depending on whether this layer is learning
                    backwardArgs = this.moveToEnvironment( ...
                        { XForThisLayer, ZForThisLayer, dLossdZ, memory } );
                    if layerIsLearning(currentLayer)
                        [dLossdX, dLossdW] = thisLayer.backward( backwardArgs{:} );
                    else
                        dLossdX = thisLayer.backward( backwardArgs{:} );
                    end
                    
                    dLossdXBuffer = iIncrementActivationsInBuffer( ...
                        dLossdXBuffer, bufferInputIndices, dLossdX );
                end
                
                % Delete data that is no longer needed
                indicesToClear = this.ListOfBufferIndicesForClearingBackward{currentLayer};
                activationsBuffer = iClearActivationsFromBuffer( ...
                    activationsBuffer, indicesToClear );
                memoryBuffer = iClearActivationsFromBuffer( ...
                    memoryBuffer, indicesToClear );
                dLossdXBuffer = iClearActivationsFromBuffer( ...
                    dLossdXBuffer, indicesToClear );
            end
            
            % We can recover GPU memory by gathering the current
            % intermediate activations back to the host.
            function gatherActivations()
                activationsBuffer = iGatherGPUCell(activationsBuffer);
            end
            recoveryStrategies = {@gatherActivations};
            %
            % We could also recover the memory and backward loss buffers
            function gatherBuffers()
                memoryBuffer = iGatherGPUCell(memoryBuffer);
                dLossdXBuffer = iGatherGPUCell(dLossdXBuffer);
            end
            recoveryStrategies = [ recoveryStrategies {@gatherBuffers} ];
            %
            % We could also return gradients on the host instead of the GPU
            gradients = {};
            function gatherGradients()
                gradients = iGatherGPUCell(gradients);
            end
            recoveryStrategies = [ recoveryStrategies {@gatherGradients} ];
            
            % To optimize away unnecessary backpropagation, determine
            % the earliest layer that needs its weight gradients computed
            earliestLearningLayer = find( layerIsLearning, 1, 'first' );
            
            % Propagate loss and gradient back through the network
            for i = this.NumLayers:-1:1
                if i >= earliestLearningLayer
                    theseGradients = iExecuteWithStagedGPUOOMRecovery( ...
                        @() efficientBackProp(i), ...
                        1, recoveryStrategies, i );
                else
                    % Pad output even if propagation has stopped
                    theseGradients = cell(1, numel(this.Layers{i}.LearnableParameters) );
                end
                gradients = [theseGradients gradients]; %#ok<AGROW>
            end
            
            % Predict
            predictions = cell(1, this.NumOutputLayers);
            for i = 1:this.NumOutputLayers
                outputLayerBufferIndex = this.ListOfBufferOutputIndices{this.OutputLayerIndices(i)};
                predictions{i} = activationsBuffer{outputLayerBufferIndex};
            end
            if this.NumOutputLayers == 1
                predictions = predictions{1};
            end
        end
        
        function loss = loss(this, Y, T)
            % Wrap Y and T in cell if needed
            Y = iWrapInCell(Y);
            T = iWrapInCell(T);
            
            % loss   Calculate the network loss
            loss = [];
            for i = 1:this.NumOutputLayers
                loss = [loss this.Layers{this.OutputLayerIndices(i)}.forwardLoss(Y{i}, T{i})]; %#ok<AGROW>
            end
            loss = sum(loss);
        end
        
        function this = updateLearnableParameters(this, deltas)
            % updateLearnableParameters   Update each learnable parameter
            % by subtracting a delta from it
            currentDelta = 1;
            for el = 1:this.NumLayers
                thisLayer = this.Layers{el};
                learnableParameters = thisLayer.LearnableParameters;
                numLearnables = numel(learnableParameters);
                if numLearnables > 0
                    this.Layers{el} = thisLayer.updateLearnableParameters( deltas(currentDelta:currentDelta+numLearnables-1) );
                    currentDelta = currentDelta + numLearnables;
                end
            end
        end
        
        function this = updateNetworkState(this, ~, ~)
        end
        
        function this = initializeLearnableParameters(this, precision)
            % initializeLearnableParameters   Initialize the learnable
            % parameters of the network
            for el = 1:this.NumLayers
                this.Layers{el} = this.Layers{el}.initializeLearnableParameters(precision);
            end
        end
        
        function this = prepareNetworkForTraining(this, executionSettings)
            % prepareNetworkForTraining   Convert the network into a format
            % suitable for training
            for el = 1:this.NumLayers
                this.Layers{el} = this.Layers{el}.prepareForTraining();
            end
            
            % Determine whether training should occur on host or GPU
            if ismember( executionSettings.executionEnvironment, {'gpu'} )
                % Don't move data if training in parallel, allow this to
                % happen as training progresses. This ensures we can
                % support clients without GPUs when the cluster has GPUs.
                delayMove = executionSettings.useParallel;
                this = this.setupNetworkForGPUTraining(delayMove);
            else
                this = this.setupNetworkForHostTraining();
            end
        end
        
        function this = prepareNetworkForPrediction(this)
            % prepareNetworkForPrediction   Convert the network into a
            % format suitable for prediction
            for el = 1:this.NumLayers
                this.Layers{el} = this.Layers{el}.prepareForPrediction();
            end
        end
        
        function this = setupNetworkForHostPrediction(this)
            % setupNetworkForHostPrediction   Setup the network to perform
            % prediction on the host
            for el = 1:this.NumLayers
                this.Layers{el} = this.Layers{el}.setupForHostPrediction();
            end
            this.UseGpu = false;
        end
        
        function this = setupNetworkForGPUPrediction(this)
            % setupNetworkForGPUPrediction   Setup the network to perform
            % prediction on the GPU
            for el = 1:this.NumLayers
                this.Layers{el} = this.Layers{el}.setupForGPUPrediction();
            end
            this.UseGpu = true;
        end
        
        function this = setupNetworkForHostTraining(this)
            % setupNetworkForHostTraining   Setup the network to train on
            % the host
            for el = 1:this.NumLayers
                this.Layers{el} = this.Layers{el}.setupForHostTraining();
                this.Layers{el} = this.Layers{el}.moveToHost();
            end
            this.UseGpu = false;
        end
        
        function this = setupNetworkForGPUTraining(this, deferMove)
            % setupNetworkForGPUTraining   Setup the network to train on
            % the GPU. deferMove allows the actual move of data to the GPU
            % to be deferred to happen as training progresses instead of in
            % advance.
            for el = 1:this.NumLayers
                this.Layers{el} = this.Layers{el}.setupForGPUTraining();
                if ~deferMove
                    this.Layers{el} = this.Layers{el}.moveToGPU();
                end
            end
            this.UseGpu = true;
        end
        
        function indices = namesToIndices(this, stringArray)
            % namesToIndices   Convert a string array of layer names into
            % layer indices
            numLayersToMatch = numel(stringArray);
            indices = zeros(numLayersToMatch,1);
            layerNames = nnet.internal.cnn.layer.Layer.getLayerNames(this.Layers);
            for i = 1:numLayersToMatch
                indices(i) = find(strcmp(stringArray(i), layerNames));
            end
        end
        
        function this = finalizeNetwork(this, X)
            % Wrap X in cell if needed
            X = iWrapInCell(X);
            
            % finalizeNetwork
            
            activationsBuffer = cell(this.NumActivations,1);
            
            % Allocate space for the activations.
            
            for i = 1:this.NumLayers
                thisLayer = this.Layers{i};
                if ismember(i, this.InputLayerIndices)
                    [~, currentInputLayer] = find(this.InputLayerIndices == i);
                    
                    [Z, memory] = thisLayer.forward(X{currentInputLayer});
                else
                    XForThisLayer = iGetTheseActivationsFromBuffer( ...
                        activationsBuffer, ...
                        this.ListOfBufferInputIndices{i});
                    
                    [Z, memory] = thisLayer.forward(XForThisLayer);
                end
                
                activationsBuffer = iAssignActivationsToBuffer( ...
                    activationsBuffer, ...
                    this.ListOfBufferOutputIndices{i}, ...
                    Z);
                
                activationsBuffer = iClearActivationsFromBuffer( ...
                    activationsBuffer, ...
                    this.ListOfBufferIndicesForClearingForward{i});
                
                if isa( thisLayer, 'nnet.internal.cnn.layer.Finalizable' ) && ...
                        thisLayer.NeedsFinalize
                    thisLayer = finalize(thisLayer, XForThisLayer, Z, memory);
                end
                this.Layers{i} = thisLayer;
                
            end
            
        end
        
        function this = inferSizes(this)
            % inferSizes   Infer layer output sizes
            
            sortedInternalLayers = this.Layers;
            numActivations = this.NumActivations;
            listOfBufferOutputIndices = this.ListOfBufferOutputIndices;
            listOfBufferInputIndices = this.ListOfBufferInputIndices;
            
            this.Sizes = cell(numActivations,1);
            numLayers = numel(sortedInternalLayers);
            this.LayerOutputSizes = cell(numLayers,1);
            
            for i = 1:numLayers
                if isa(sortedInternalLayers{i}, 'nnet.internal.cnn.layer.InputLayer')
                    inputSizesForThisLayer = sortedInternalLayers{i}.InputSize;
                else
                    inputSizesForThisLayer = iGetInputsFromBuffer( ...
                        this.Sizes, listOfBufferInputIndices{i});
                end
                
                sortedInternalLayers{i} = iInferSize( ...
                    sortedInternalLayers{i}, ...
                    inputSizesForThisLayer, ...
                    i);
                
                outputSizesForThisLayer = sortedInternalLayers{i}.forwardPropagateSize( ...
                    inputSizesForThisLayer);
                this.Sizes = iAssignOutputsToBuffer( ...
                    this.Sizes, listOfBufferOutputIndices{i}, outputSizesForThisLayer);
                this.LayerOutputSizes{i} = outputSizesForThisLayer;
            end
        end
        
        function layerOutputSizes = inferOutputSizesGivenInputSizes(this, inputSizes, layerIndices)
            % inferOutputSizesGivenInputSizes   Infer output size from all
            % or a given layer given new input sizes for input layers.
            %
            % Suppose this internal DAG network has N layers which have
            % been topologically sorted and numbered from 1 to N. Suppose
            % the network has M input layers and they appear in positions
            % i_1, i_2, ..., i_M in the topologically sorted list.
            %
            % inputSizes       - is a length M cell array specifying the
            %                    input sizes for layers i_1, i_2, ..., i_M
            %                    in that order.
            %
            % layerIndices    -  the indices into the topologically
            %                    sorted, non-optimized list of layers
            %                    originally used to construct this
            %                    network.
            %
            % layerOutputSizes - a cell array of length
            %                    numel(layerIndices), where
            %                    layerOutputSizes{i} gives the output size
            %                    for layer layerIndices(i). If that layer
            %                    has multiple outputs then
            %                    layerOutputSizes{i} is a cell array of
            %                    output sizes for this layer.
            
            % Return all sizes if no layers specified
            numLayers = numel(this.Layers);
            if nargin < 3
                layerIndices = 1:numLayers;
            end
            numOutputs = numel(layerIndices);
            
            listOfBufferOutputIndices = this.ListOfBufferOutputIndices;
            listOfBufferInputIndices = this.ListOfBufferInputIndices;
            
            % Convert layerIndices into a indices into the optimized layers
            % plus offsets
            [layerIndices, layerOffsets] = this.NetworkOptimizer.mapFromOriginal(layerIndices);
            
            % Preallocate the size buffers
            %  sizes: Size of every output activation
            sizes = cell(this.NumActivations, 1);
            %  layerOutputSizes: The sizes requested
            layerOutputSizes = cell(numOutputs, 1);
            
            % Propagate sizes through the layers, filling the sizes buffer
            maxLayerIndex = max(layerIndices);
            for i = 1:maxLayerIndex
                thisLayer = this.Layers{i};
                if any(i == this.InputLayerIndices)
                    % For an input layer, forwardPropagateSize sets
                    % the output size equal to the InputSize property.
                    % Since we don't want that, we force the output size
                    % to be equal to the specified input size.
                    [~, currentInputLayer] = find(this.InputLayerIndices == i);
                    inputSizesForThisLayer = inputSizes{currentInputLayer};
                    outputSizesForThisLayer = inputSizesForThisLayer;
                else
                    inputSizesForThisLayer = iGetInputsFromBuffer( ...
                        sizes, listOfBufferInputIndices{i});
                    outputSizesForThisLayer = thisLayer.forwardPropagateSize( ...
                        inputSizesForThisLayer);
                end
                
                sizes = iAssignOutputsToBuffer( ...
                    sizes, listOfBufferOutputIndices{i}, outputSizesForThisLayer);
            end
            
            % Copy sizes from the buffer to the output. Where a layer is a
            % FusedLayer, propagate through that layer to the internal
            % layer requested.
            for i = 1:numOutputs
                thisLayerIndex = layerIndices(i);
                thisLayer = this.Layers{thisLayerIndex};
                if iIsAFusedLayer(thisLayer)
                    inputSizesForThisLayer = iGetInputsFromBuffer( ...
                        sizes, listOfBufferInputIndices{thisLayerIndex});
                    outputSizesForThisLayer = thisLayer.forwardPropagateSize( ...
                        inputSizesForThisLayer, layerOffsets(i) );
                else
                    outputSizesForThisLayer = iGetInputsFromBuffer( ...
                        sizes, listOfBufferOutputIndices{thisLayerIndex} );
                end
                layerOutputSizes{i} = outputSizesForThisLayer;
            end
        end
        
        function layerGraph = makeTrainedLayerGraph(this)
            % makeTrainedLayerGraph - makes an internal Layer graph
            % with most recent values of learnable parameters
            layerGraph = iMakeInternalLayerGraph(this.OriginalLayers, this.OriginalConnections);
        end
    end
    
    methods( Access = private )
        
        function X = moveToEnvironment( this, X )
            if this.UseGpu
                X = iMoveToGpu(X);
            end
        end
        
    end
    
end


function layerGraph = iMakeInternalLayerGraph(layers, connections)
layerGraph = nnet.internal.cnn.LayerGraph(layers, connections);
end

function internalConnections = iExternalToInternalConnections( externalConnections )
externalEndNodes = externalConnections.EndNodes;
externalEndPorts = externalConnections.EndPorts;
numEndPortsPerEndNodes = cellfun(@(x) size(x,1), externalEndPorts);
internalEndPorts = cell2mat(externalEndPorts);
internalEndNodes = [repelem(externalEndNodes(:,1),numEndPortsPerEndNodes), repelem(externalEndNodes(:,2),numEndPortsPerEndNodes)];
internalConnections = [internalEndNodes(:,1),internalEndPorts(:,1),internalEndNodes(:,2),internalEndPorts(:,2)];
end

function X = iWrapInCell(X)
if ~iscell(X)
    X = {X};
end
end

function numInputLayers = iCountInputLayers(internalLayers)
numInputLayers = 0;
for i = 1:numel(internalLayers)
    if( iIsAnInputLayer(internalLayers{i}) )
        numInputLayers = numInputLayers + 1;
    end
end
end

function numOutputLayers = iCountOutputLayers(internalLayers)
numOutputLayers = 0;
for i = 1:numel(internalLayers)
    if( iIsAnOutputLayer(internalLayers{i}) )
        numOutputLayers = numOutputLayers + 1;
    end
end
end

function inputLayerIndices = iGetInputLayerIndices(internalLayers)
numLayers = numel(internalLayers);
inputLayerIndices = cell(1,numLayers);
for i = 1:numLayers
    if(iIsAnInputLayer(internalLayers{i}))
        inputLayerIndices{i} = i;
    end
end
inputLayerIndices = cat(2,inputLayerIndices{:});
end

function outputLayerIndices = iGetOutputLayerIndices(internalLayers)
numLayers = numel(internalLayers);
outputLayerIndices = cell(1,numLayers);
for i = 1:numLayers
    if(iIsAnOutputLayer(internalLayers{i}))
        outputLayerIndices{i} = i;
    end
end
outputLayerIndices = cat(2, outputLayerIndices{:});
end

function inputSizes = iGetInputSizes(sizes, inputLayerIndices)
numInputLayers = numel(inputLayerIndices);
inputSizes = cell(1, numInputLayers);
for i = 1:numInputLayers
    currentLayer = inputLayerIndices(i);
    inputSizes{i} = sizes{currentLayer};
end
end

function outputSizes = iGetOutputSizes(sizes, outputLayerIndices)
numOutputLayers = numel(outputLayerIndices);
outputSizes = cell(1, numOutputLayers);
for i = 1:numOutputLayers
    currentLayer = outputLayerIndices(i);
    outputSizes{i} = sizes{currentLayer};
end
end

function tf = iIsAnInputLayer(internalLayer)
tf = isa(internalLayer,'nnet.internal.cnn.layer.InputLayer');
end

function tf = iIsAnOutputLayer(internalLayer)
tf = isa(internalLayer,'nnet.internal.cnn.layer.OutputLayer');
end

function tf = iIsAFusedLayer(internalLayer)
tf = isa(internalLayer,'nnet.internal.cnn.layer.FusedLayer');
end

function activationsBuffer = iClearActivationsFromBuffer(activationsBuffer, indicesToClear)
activationsBuffer = nnet.internal.cnn.util.LayerGraphExecutionInfo.clearActivationsFromBuffer( ...
    activationsBuffer, indicesToClear);
end

function XForThisLayer = iGetTheseActivationsFromBuffer(activationsBuffer, inputIndices)
XForThisLayer = activationsBuffer(inputIndices);
if(iscell(XForThisLayer) && (numel(XForThisLayer) == 1))
    XForThisLayer = XForThisLayer{1};
end
end

function memoryBuffer = iAssignMemoryToBuffer(...
    memoryBuffer, ...
    bufferIndices, ...
    memory)
% FYI Batch norm stores its memory as a cell.
for i = 1:numel(bufferIndices)
    memoryBuffer{bufferIndices(i)} = memory;
end
end

function activationsBuffer = iAssignActivationsToBuffer( ...
    activationsBuffer, ...
    bufferIndices, ...
    activations)
if iscell(activations)
    activationsBuffer(bufferIndices) = activations;
else
    activationsBuffer{bufferIndices} = activations;
end
end

function activationsBuffer = iIncrementActivationsInBuffer(activationsBuffer, bufferIndices, activations)

numActivationsFromLayer = numel(bufferIndices);
if ~iscell(activations)
    if isempty(activationsBuffer{bufferIndices})
        activationsBuffer{bufferIndices} = activations;
    else
        activationsBuffer{bufferIndices} = activationsBuffer{bufferIndices} + activations;
    end
else
    for i = 1:numActivationsFromLayer
        if isempty(activationsBuffer{bufferIndices(i)})
            activationsBuffer{bufferIndices(i)} = activations{i};
        else
            activationsBuffer{bufferIndices(i)} = activationsBuffer{bufferIndices(i)}+ activations{i};
        end
    end
end
end

function internalLayer = iInferSize(internalLayer, inputSize, index)
if(~internalLayer.HasSizeDetermined)
    % Infer layer size if its size is not determined
    try
        internalLayer = internalLayer.inferSize(inputSize);
    catch e
        throwWrongLayerSizeException( e, index );
    end
else
    % Otherwise make sure the size of the layer is correct
    iAssertCorrectSize( internalLayer, index, inputSize );
end
end

function activationsBuffer = iAssignOutputsToBuffer( ...
    activationsBuffer, ...
    outputIndices, ...
    outputActivations)

numOutputsFromLayer = numel(outputIndices);
if ~iscell(outputActivations)
    activationsBuffer{outputIndices} = outputActivations;
else
    for i = 1:numOutputsFromLayer
        activationsBuffer{outputIndices(i)} = outputActivations{i};
    end
end
end

function iAssertCorrectSize( internalLayer, index, inputSize )
% iAssertCorrectSize   Check that layer size matches the input size,
% otherwise the architecture would be inconsistent.
if ~internalLayer.isValidInputSize( inputSize )
    exception = iCreateExceptionFromErrorID('nnet_cnn:inferParameters:WrongLayerSize', index);
    throwAsCaller(exception);
end
end

function throwWrongLayerSizeException(e, index)
% throwWrongLayerSizeException   Throws a getReshapeDims:notSameNumel exception as
% a WrongLayerSize exception
if (strcmp(e.identifier,'MATLAB:getReshapeDims:notSameNumel'))
    exception = iCreateExceptionFromErrorID('nnet_cnn:inferParameters:WrongLayerSize', index);
    throwAsCaller(exception)
else
    rethrow(e)
end
end

function exception = iCreateExceptionFromErrorID(errorID, varargin)
exception = MException(message(errorID, varargin{:}));
end

function XForThisLayer = iGetInputsFromBuffer(layerOutputs, inputIndices)
XForThisLayer = layerOutputs(inputIndices);
if(iscell(XForThisLayer) && (numel(XForThisLayer) == 1))
    XForThisLayer = XForThisLayer{1};
end
end

function cellOrArray = iGatherGPUCell(cellOrArray)
if iscell(cellOrArray)
    cellOrArray = cellfun(@iGatherGPUCell, cellOrArray, 'UniformOutput', false);
elseif isa(cellOrArray, 'gpuArray')
    cellOrArray = gather(cellOrArray);
end
end

function varargout = iExecuteWithStagedGPUOOMRecovery(varargin)
[varargout{1:nargout}] = nnet.internal.cnn.util.executeWithStagedGPUOOMRecovery(varargin{:});
end

function X = iMoveToGpu(X)
if iscell(X)
    X = cellfun(@iMoveToGpu, X, 'UniformOutput', false);
elseif isnumeric(X)
    X = gpuArray(X);
end
end
