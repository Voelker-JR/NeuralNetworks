using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworks
{
    public class RPROPTraining : Training
    {
        public RPROPTraining(Net net, double decreaseFactor, double increaseFactor)
            : base(net)
        {
            DecreaseFactor = decreaseFactor;
            IncreaseFactor = increaseFactor;

            foreach (var layer in net.Layers)
                LayerWrappers.Add(new RPROPLayerWrapper(layer, decreaseFactor, increaseFactor));
        }

        public double DecreaseFactor { get; }

        public double IncreaseFactor { get; }
    }
}
