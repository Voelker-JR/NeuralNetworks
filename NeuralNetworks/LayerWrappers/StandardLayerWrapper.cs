﻿using VectorMath;

namespace NeuralNetworks
{
    public class StandardLayerWrapper : TrainingLayerWrapper
    {
        public StandardLayerWrapper(Layer layer, double learningRate)
            : base(layer)
        {
            LearningRate = learningRate;
        }

        public double LearningRate { get; }

        public override void ApplyBackpropChanges(Matrix changes)
        {
            WeightsDiff += changes;
        }

        public override void ApplyWeightChanges(int epoch)
        {
            AssociatedLayer.Weights += LearningRate * WeightsDiff;

            // Reset the Matrix to zero
            int rows = WeightsDiff.Rows;
            int columns = WeightsDiff.Columns;

            WeightsDiff = new Matrix(rows, columns);
        }
    }
}
