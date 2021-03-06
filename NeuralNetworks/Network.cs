﻿using System.Collections.Generic;

using VectorMath;

namespace NeuralNetworks
{
    public class Network
    {
        public Network(List<int> structure, INetActivation activation)
        {
            Structure = new List<int>(structure);
            Layers = new List<Layer>();
            Activation = activation;

            Reset();
        }

        // Copy constructor
        public Network(Network original)
        {
            Structure = new List<int>(original.Structure);
            Layers = new List<Layer>();
            Activation = original.Activation;

            foreach (var layer in original.Layers)
            {
                var copyLayer = new Layer(layer.Output.Dim, layer.Input.Dim, Activation)
                {
                    Weights = new Matrix(layer.Weights)
                };

                Layers.Add(copyLayer);
            }
        }

        public List<int> Structure { get; }

        public List<Layer> Layers { get; }

        public INetActivation Activation { get; }

        public Vector Feed(Vector input)
        {
            Vector output = new Vector(input);

            // Propagate the input through the layers.
            foreach (var layer in Layers)
                output = layer.Feed(output);

            return output;
        }

        public void Reset()
        {
            Layers.Clear();

            // Create Structure.Count - 1 layers (drop input layer).
            for (int i = 1; i < Structure.Count; i++)
            {
                int inDim = Structure[i - 1];
                int outDim = Structure[i];

                if (i < Structure.Count - 1)
                    Layers.Add(new Layer(outDim, inDim, Activation));
                else
                    Layers.Add(new Layer(outDim, inDim, NetFunctions.Identity));
            }
        }
    }
}
