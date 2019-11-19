using VectorMath;

namespace NeuralNetworks
{
    public class Layer
    {
        private Vector input;
        private Vector output;

        public Layer()
        {
            //TODO: Assign random values to Weights
        }

        public Vector Input => input;

        public Vector Output => output;

        public Matrix Weights { get; set; }

        public Vector Feed(Vector rawInput)
        {
            // Extend the rawInput by one 1 at the end
            input = new Vector(rawInput.Dim + 1);

            for (int i = 0; i < input.Dim; i++)
                input[i] = rawInput[i];

            input[rawInput.Dim] = 1;
            //////////////////////////////////////////

            // Calculate the output vector by multiplicating with weights
            output = Weights * input;

            return output;
        }

        public Vector CalcDelta(Layer next)
        {

        }

        public Vector CalcDelta(Vector netOutput)
        {

        }


    }
}
