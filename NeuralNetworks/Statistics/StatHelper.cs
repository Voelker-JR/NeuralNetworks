using System;
using System.Collections.Generic;
using System.Linq;
using VectorMath;

namespace NeuralNetworks.Statistics
{
    public static class StatHelper
    {
        public static double NextGaussian(Random random, double mean, double stdDeviation)
        {
            // Generate gaussian random value based on Box-Muller transform
            double x1 = 1.0 - random.NextDouble();
            double x2 = 1.0 - random.NextDouble();

            // Standard normal distributed evaluation [N(0, 1)]
            double gaussian = Math.Sqrt(-2.0 * Math.Log(x1)) * Math.Cos(2.0 * Math.PI * x2);

            // Var(gaussian * stdDeviation + mean) = stdDeviation^2 * 1 and
            // E[gaussian * stdDeviation + mean] = 0 + mean.
            return gaussian * stdDeviation + mean;
        }

        public static List<double> Standardize(List<double> data)
        {
            StatAnalysis analysis = new StatAnalysis(data);

            return new List<double>(
                from d in data
                select (d - analysis.Mean) / Math.Sqrt(analysis.Variance));
        }

        public static List<double> VectorAsList(Vector v)
        {
            return new List<double>(v.Data);
        }

        public static List<double> MatrixAsList(Matrix m)
        {
            var result = new List<double>(m.Rows * m.Columns);

            for (int i = 0; i < m.Rows; i++)
                for (int j = 0; j < m.Columns; j++)
                    result.Add(m[i, j]);

            return result;
        }

        public static Dictionary<double, double> ListAsDistribution(List<double> list, int regions)
        {
            Dictionary<double, double> result = new Dictionary<double, double>();

            double minimum = list.Min();
            double listExtent = list.Max() - list.Min();
            double interval = listExtent / regions;

            for (int i = 0; i < regions; i++)
            {
                // xValue set in the mid of the interval
                double xValue = (i + 0.5) * interval + minimum;

                // select all values in [i * interval, (i + 1) * interval) + minimum.
                double count = (from value in list
                                where value >= i * interval + minimum
                                && value < (i + 1) * interval + minimum
                                select value)
                                .Count();

                result.Add(xValue, count / list.Count);
            }

            return result;
        }
    }
}
