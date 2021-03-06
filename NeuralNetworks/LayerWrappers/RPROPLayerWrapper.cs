﻿using System;
using System.Collections.Generic;
using VectorMath;

namespace NeuralNetworks
{
    class RPROPLayerWrapper : TrainingLayerWrapper
    {
        private const double maxWeightsDiff = 1;
        private const double minWeightsDiff = 1e-6;

        private List<Matrix> gradients;

        public RPROPLayerWrapper(Layer layer,
            double decreaseFactor, double increaseFactor) : base(layer)
        {
            gradients = new List<Matrix>();
            PushGradient();

            DecreaseFactor = decreaseFactor;
            IncreaseFactor = increaseFactor;
        }

        public double DecreaseFactor { get; }

        public double IncreaseFactor { get; }

        private void PushGradient()
        {
            int rows = AssociatedLayer.Weights.Rows;
            int columns = AssociatedLayer.Weights.Columns;

            gradients.Insert(0, new Matrix(rows, columns));

            // while: If there are more than 4 elements (somehow)
            while (gradients.Count > 3)
                gradients.RemoveAt(3);
        }

        public override void ApplyBackpropChanges(Matrix changes)
        {
            gradients[0] += changes;
        }

        public override void ApplyWeightChanges(int epoch)
        {
            // Calculate
            CalcWeightChanges();

            // Apply on Weights
            AssociatedLayer.Weights +=
                WeightsDiff ^ gradients[0].Apply(NetFunctions.Sgn);

            // Resetting gradient by pushing a zero matrix into the chain.
            PushGradient();
        }

        /// <summary>
        /// Implements the RPROP-Algorithm.
        /// </summary>
        private void CalcWeightChanges()
        {
            if (gradients.Count < 3)
            {
                WeightsDiff = 0.001 * gradients[0];
                return;
            }

            for (int i = 0; i < WeightsDiff.Rows; i++)
            {
                for (int j = 0; j < WeightsDiff.Columns; j++)
                {
                    double condition1 = gradients[0][i, j] * gradients[1][i, j];
                    double condition2 = gradients[1][i, j] * gradients[2][i, j];

                    // Increase or decrease depending on the conditions above
                    if (condition1 > 0 && condition2 >= 0)
                    {
                        WeightsDiff[i, j] *= IncreaseFactor;

                        // Check upper bound
                        WeightsDiff[i, j] = Math.Min(WeightsDiff[i, j], maxWeightsDiff);
                    }
                    else if (condition1 < 0)
                    {
                        WeightsDiff[i, j] *= DecreaseFactor;

                        // Check lower bound
                        WeightsDiff[i, j] = Math.Max(WeightsDiff[i, j], minWeightsDiff);
                    }
                }
            }
        }
    }
}
