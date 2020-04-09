using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworks
{
    public class RMSPropTraining : MiniBatchTraining
    {
        public RMSPropTraining(Network network, int batchSize, double learningRate)
            : base(batchSize)
        {
            LearningRate = learningRate;
            ResetNetwork(network);
        }

        public double LearningRate { get; }

        public override void ResetNetwork(Network network)
        {
            Network = network;
            LayerWrappers = new List<TrainingLayerWrapper>();

            foreach (var layer in network.Layers)
                LayerWrappers.Add(new RMSPropLayerWrapper(layer, LearningRate));
        }
    }
}
