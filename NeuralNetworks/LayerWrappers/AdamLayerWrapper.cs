using System;
using VectorMath;

namespace NeuralNetworks
{
    public class AdamLayerWrapper : TrainingLayerWrapper
    {
        private const double epsilon = 1e-8;

        private const double beta1 = 0.9;
        private const double beta2 = 0.999;

        /// <summary>
        /// First moment estimator for the gradient.
        /// </summary>
        private Matrix m;

        /// <summary>
        /// Second moment estimator for the gradient.
        /// </summary>
        private Matrix v;

        public AdamLayerWrapper(Layer layer, double learningRate) : base(layer)
        {
            LearningRate = learningRate;

            m = new Matrix(WeightsDiff.Rows, WeightsDiff.Columns);
            v = new Matrix(WeightsDiff.Rows, WeightsDiff.Columns);
        }

        public double LearningRate { get; }

        public override void ApplyBackpropChanges(Matrix changes)
        {
            WeightsDiff += changes;
        }

        public override void ApplyWeightChanges(int epoch)
        {
            // Calculate
            m = beta1 * m + (1 - beta1) * WeightsDiff;
            v = beta2 * v + (1 - beta2) * (WeightsDiff ^ WeightsDiff);

            Matrix mUnbiased = m / (1.0 - Math.Pow(beta1, epoch));
            Matrix vUnbiased = v / (1.0 - Math.Pow(beta2, epoch));

            // Apply on Weights
            AssociatedLayer.Weights += LearningRate * mUnbiased ^
                vUnbiased.Apply(x => 1.0 / (Math.Sqrt(x) + epsilon));

            // Reset gradient
            WeightsDiff = new Matrix(WeightsDiff.Rows, WeightsDiff.Columns);
        }
    }
}
