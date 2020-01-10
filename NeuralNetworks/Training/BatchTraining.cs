using VectorMath;

namespace NeuralNetworks
{
    public class BatchTraining : Training
    {
        public BatchTraining(Net net) : base(net)
        { }

        public override void Run()
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
                // Implements the specific algorithm
                layerWrapper.ApplyWeightChanges(CurrentEpoch);
            }
        }
    }
}
