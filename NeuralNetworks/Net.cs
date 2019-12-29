using System.Collections.Generic;

using VectorMath;

namespace NeuralNetworks
{
    public class Net
    {
        public Net(List<int> structure, INetActivation activation)
        {
            Structure = new List<int>(structure);
            Layers = new List<Layer>();

            // Create Structure.Count - 1 layers (drop input layer).
            for (int i = 1; i < Structure.Count; i++)
            {
                int inDim = Structure[i - 1];
                int outDim = Structure[i];

                if (i < Structure.Count - 1)
                    Layers.Add(new Layer(outDim, inDim, activation));
                else
                    Layers.Add(new Layer(outDim, inDim, NetFunctions.Identity));
            }

            Activation = activation;
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
    }
}
