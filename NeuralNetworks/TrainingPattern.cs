using System;
using System.Collections.Generic;
using System.Text;
using VectorMath;

namespace NeuralNetworks
{
    public struct TrainingPattern
    {
        public TrainingPattern(Vector input, Vector output, int priority)
        {
            Input = input;
            Output = output;
            Priority = priority;
        }

        public TrainingPattern(Vector input, Vector output) : this(input, output, 1)
        { }

        public Vector Input { get; }

        public Vector Output { get; }

        public int Priority { get; }
    }
}
