using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworks
{
    public class StandardTraining : BatchTraining
    {
        public StandardTraining(Network network, double learningRate)
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
                LayerWrappers.Add(new StandardLayerWrapper(layer, LearningRate));
        }
    }
}
