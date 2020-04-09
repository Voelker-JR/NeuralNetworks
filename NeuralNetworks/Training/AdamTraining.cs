using System.Collections.Generic;

namespace NeuralNetworks
{
    public class AdamTraining : MiniBatchTraining
    {
        public AdamTraining(Network network, int batchSize, double learningRate)
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
                LayerWrappers.Add(new AdamLayerWrapper(layer, LearningRate));
        }
    }
}
