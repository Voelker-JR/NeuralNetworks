using System;
using System.Collections;
using System.Collections.Generic;

using VectorMath;

namespace NeuralNetworks
{
    public class Net
    {
        public Net(List<int> structure, AppliableFunction activation,
            AppliableFunction activationDeriv)
        {
            Structure = new List<int>(structure);

            // Create Structure.Count - 1 layers (drop input layer).
            for (int i = 1; i < Structure.Count; i++)
            {
                int inDim = Structure[i - 1];
                int outDim = Structure[i];

                Layers.Add(new Layer(outDim, inDim));
            }

            Activation = activation;
            ActivationDeriv = activationDeriv;
        }

        public List<int> Structure { get; private set; }

        public List<Layer> Layers { get; private set; }

        public AppliableFunction Activation { get; private set; }

        public AppliableFunction ActivationDeriv { get; private set; }

        public Vector Feed(Vector input)
        {
            Vector output = new Vector(input);

            // Propagate the input through the layers.
            foreach (var layer in Layers)
                output = layer.Feed(output);

            return output;
        }
    }
}
