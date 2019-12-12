using VectorMath;

namespace NeuralNetworks
{
    public class Layer
    {
        public const double initialWeightAbsolute = 0.01;

        public Layer(int outDim, int inDim, AppliableFunction activation,
            AppliableFunction activationDeriv)
        {
            // Add one column for the bias values.
            Weights = Matrix.Factory.Random(outDim, inDim + 1,
                -initialWeightAbsolute, initialWeightAbsolute);

            Activation = activation;
            ActivationDeriv = activationDeriv;
        }

        public Vector Input { get; private set; }

        public Vector Output { get; private set; }

        public Matrix Weights { get; set; }

        public AppliableFunction Activation { get; private set; }

        public AppliableFunction ActivationDeriv { get; private set; }

        public Vector Feed(Vector rawInput)
        {
            // Insert the rawInput vector into a zero vector: (v, ..., v, 0).
            Input = Vector.Factory.FitIn(rawInput.Dim + 1, 0, rawInput);

            // Add the last basis vector to the input vector: (v, ..., v, 1).
            Input += Vector.Factory.BasisVector(rawInput.Dim + 1, rawInput.Dim);

            // Calculate the output vector by multiplicating with weights.
            // We also have to apply the activation function coordinatewise.
            Output = (Weights * Input).Apply(Activation);

            return Output;
        }
    }
}
