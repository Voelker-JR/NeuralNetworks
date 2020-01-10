using System;
using VectorMath;

namespace NeuralNetworks
{
    public class RMSPropLayerWrapper : TrainingLayerWrapper
    {
        private const double epsilon = 1e-8;
        private const double beta = 0.9;

        private Matrix movingSquareAverage;

        public RMSPropLayerWrapper(Layer layer, double learningRate) : base(layer)
        {
            LearningRate = learningRate;

            // Fill with 0
            movingSquareAverage = new Matrix(WeightsDiff.Rows, WeightsDiff.Columns);
        }

        public double LearningRate { get; }

        public override void ApplyBackpropChanges(Matrix changes)
        {
            WeightsDiff += changes;
        }

        public override void ApplyWeightChanges(int epoch)
        {
            // Calculate
            movingSquareAverage =
                beta * movingSquareAverage + (1 - beta) * (WeightsDiff ^ WeightsDiff);

            // Apply on Weights
            AssociatedLayer.Weights += WeightsDiff ^
                movingSquareAverage.Apply(x => LearningRate / Math.Sqrt(x + epsilon));

            // Reset gradient
            WeightsDiff = new Matrix(WeightsDiff.Rows, WeightsDiff.Columns);
        }
    }
}
