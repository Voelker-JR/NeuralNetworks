using System.Collections.Generic;

namespace NeuralNetworks
{
    public class RPROPTraining : BatchTraining
    {
        public RPROPTraining(Network network, double decreaseFactor, double increaseFactor)
        {
            DecreaseFactor = decreaseFactor;
            IncreaseFactor = increaseFactor;

            ResetNetwork(network);
        }

        public double DecreaseFactor { get; }

        public double IncreaseFactor { get; }

        public override void ResetNetwork(Network network)
        {
            Network = network;
            LayerWrappers = new List<TrainingLayerWrapper>();

            foreach (var layer in network.Layers)
                LayerWrappers.Add(new RPROPLayerWrapper(layer, DecreaseFactor, IncreaseFactor));
        }
    }
}
