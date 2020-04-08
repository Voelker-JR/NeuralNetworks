using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworks
{
    public class StandardTraining : BatchTraining
    {
        public StandardTraining(Network net, double learningRate)
            : base(net)
        {
            LearningRate = learningRate;

            foreach (var layer in net.Layers)
                LayerWrappers.Add(new StandardLayerWrapper(layer, learningRate));
        }

        public double LearningRate { get; }
    }
}
