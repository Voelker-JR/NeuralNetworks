using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using VectorMath;

namespace NeuralNetworks
{
    public abstract class Training
    {
        public Training(Network net)
        {
            AssociatedNet = net;

            // This list has to be filled in the extending classes
            LayerWrappers = new List<TrainingLayerWrapper>();

            CurrentEpoch = 1;
            MaxEpochs = 0;
            RegularizationRate = 0;
        }

        public Network AssociatedNet { get; }

        public List<TrainingLayerWrapper> LayerWrappers { get; }

        public List<TrainingPattern> Patterns { get; set; }

        public List<TrainingPattern> ValidationPatterns { get; set; }

        public double CurrentError { get; private set; }

        public int CurrentEpoch { get; private set; }

        public int MaxEpochs { get; private set; }

        public double RegularizationRate { get; set; }

        /// <summary>
        /// Implements batch, stochastic or mini-batch gradient descent.
        /// </summary>
        public abstract void Run();

        public Task Run(int epochs, CancellationToken ct)
        {
            MaxEpochs = epochs;

            return Task.Run(() =>
            {
                for (CurrentEpoch = 1; 
                    CurrentEpoch <= MaxEpochs && !ct.IsCancellationRequested;
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

        protected void Backpropagation(Vector netOutput, Vector patternOutput)
        {
            Vector delta = null;
            Matrix successorWeights = null;

            var outputDifference = patternOutput - netOutput;
            CurrentError = outputDifference * outputDifference;

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
                    delta = outputDifference ^ activationDerivative;
                else
                {
                    // Drop the last entry of the vector and adjust the delta vector
                    delta = Vector.Factory.FitIn(layer.Output.Dim, 0, delta * successorWeights);
                    delta ^= activationDerivative;  // Apply the activation derivative on delta
                }
                
                // Now update the differences of the weights by passing the dyadic product of
                // delta and the input vector of the layer to the LayerWrapper.
                layerWrapper.ApplyBackpropChanges((delta | layer.Input) - (RegularizationRate * layer.Weights));

                // In order to get the weights of the successing layer, we have to save it here.
                successorWeights = layer.Weights;
            }
        }
    }
}
