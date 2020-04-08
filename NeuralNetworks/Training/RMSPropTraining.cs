using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworks
{
    public class RMSPropTraining : MiniBatchTraining
    {
        public RMSPropTraining(Network net, int batchSize, double learningRate)
            : base(net, batchSize)
        {
            LearningRate = learningRate;

            foreach (var layer in net.Layers)
                LayerWrappers.Add(new RMSPropLayerWrapper(layer, learningRate));
        }

        public double LearningRate { get; }
    }
}
