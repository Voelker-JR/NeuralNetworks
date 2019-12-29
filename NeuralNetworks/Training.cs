using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using VectorMath;

namespace NeuralNetworks
{
    public abstract class Training
    {
        public Training(Net net)
        {
            AssociatedNet = net;

            // This list has to be filled in the extending classes
            LayerWrappers = new List<TrainingLayerWrapper>();

            CurrentEpoch = 0;
            MaxEpochs = 0;
        }

        public Net AssociatedNet { get; }

        public List<TrainingLayerWrapper> LayerWrappers { get; }

        public List<TrainingPattern> Patterns { get; set; }

        public int CurrentEpoch { get; private set; }

        public int MaxEpochs { get; private set; }

        public void Run()
        {
            // Run through all patterns
            foreach (var pattern in Patterns)
            {
                // Run through *this* pattern "priority" times
                for (int i = 0; i < pattern.Priority; i++)
                {
                    Vector netOutput = AssociatedNet.Feed(pattern.Input);
                    Backpropagation(netOutput, pattern.Output);
                }
            }

            // Batch Training: Adjusting *after* running through all patterns.
            foreach (var layerWrapper in LayerWrappers)
            {
                layerWrapper.CalcWeightChanges(); // Implements the specific algorithm
                layerWrapper.ApplyWeightChanges();
            }
        }

        public Task Run(int epochs, CancellationToken ct)
        {
            MaxEpochs = epochs;

            return Task.Run(() =>
            {
                for (CurrentEpoch = 0; 
                    CurrentEpoch < MaxEpochs && !ct.IsCancellationRequested;
                    CurrentEpoch++)
                {
                    // No other training must be running on this net.
                    lock (AssociatedNet)
                    {
                        Run();
                    }
                }
            }, ct);
        }

        private void Backpropagation(Vector netOutput, Vector patternOutput)
        {
            Vector delta = null;
            Matrix successorWeights = null;

            // i is used for backwards iteration (see below)
            for (int i = 1; i <= LayerWrappers.Count; i++)
            {
                var layerWrapper = LayerWrappers[LayerWrappers.Count - i];
                var layer = layerWrapper.AssociatedLayer;

                // Get the activation function of this layer (output layer often: Identity)
                INetActivation activation = layerWrapper.AssociatedLayer.Activation;
                Vector activationDerivative;

                // Decide which type of derivative function this activation provides
                if (activation.ApplyOnArgument)  // If dout/dnet is a function of net
                    activationDerivative = layer.PreOutput.Apply(activation.ApplyDerivative);
                else                             // If dout/dnet is a function of out
                    activationDerivative = layer.Output.Apply(activation.ApplyDerivative);

                if (i == 1)  // Apply the activation derivative componentwise on the error
                    delta = (patternOutput - netOutput) ^ activationDerivative;
                else
                {
                    // Drop the last entry of the vector and adjust the delta vector
                    delta = Vector.Factory.FitIn(layer.Output.Dim, 0, delta * successorWeights);
                    delta ^= activationDerivative;  // Apply the activation derivative on delta
                }
                
                // Now update the differences of the weights by passing the dyadic product of
                // delta and the input vector of the layer to the LayerWrapper.
                layerWrapper.ApplyBackpropChanges(delta | layer.Input);

                // In order to get the weights of the successing layer, we have to save it here.
                successorWeights = layer.Weights;
            }
        }
    }
}
