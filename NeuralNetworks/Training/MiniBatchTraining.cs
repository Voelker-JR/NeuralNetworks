using System;
using System.Collections.Generic;
using VectorMath;

namespace NeuralNetworks
{
    public abstract class MiniBatchTraining : Training
    {
        public MiniBatchTraining(int batchSize)
        {
            BatchSize = batchSize;
        }

        public int BatchSize { get; }

        private List<TrainingPattern> CreateMiniBatch()
        {
            var miniBatch = new List<TrainingPattern>();

            // Create range from 0 to Patterns.Count - 1
            var miniBatchIndices = new List<int>();     
            for (int i = 0; i < Patterns.Count; i++)
                miniBatchIndices.Add(i);

            Random random = new Random();

            while (miniBatch.Count < BatchSize)
            {
                // Choose one index from remaining indices randomly
                var indexPick = random.Next(miniBatchIndices.Count);
                var patternPick = miniBatchIndices[indexPick];

                // Remove chosen index from index list
                miniBatchIndices.RemoveAt(indexPick);

                miniBatch.Add(Patterns[patternPick]);
            }

            return miniBatch;
        }

        public override void Run()
        {
            // Randomly pick patterns of number batchSize
            List<TrainingPattern> miniBatch;

            if (BatchSize == Patterns.Count)
                miniBatch = Patterns;
            else
                miniBatch = CreateMiniBatch();

            // Run through all patterns of the mini batch
            foreach (var pattern in miniBatch)
            {
                // Run through *this* pattern "priority" times
                for (int i = 0; i < pattern.Priority; i++)
                {
                    Vector netOutput = Network.Feed(pattern.Input);
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
