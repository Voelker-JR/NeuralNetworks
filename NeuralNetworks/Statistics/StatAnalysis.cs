using System.Collections.Generic;

namespace NeuralNetworks.Statistics
{
    public class StatAnalysis
    {
        public StatAnalysis(List<double> data)
        {
            Data = data;
            CalculateMean();
            CalculateVariance();
        }

        public List<double> Data { get; }

        public double Mean { get; private set; }

        public double Variance { get; private set; }

        private void CalculateMean()
        {
            Mean = 0;

            foreach (double d in Data)
                Mean += d;

            Mean /= Data.Count;
        }

        private void CalculateVariance()
        {
            Variance = 0;

            foreach (double d in Data)
            {
                var centeredValue = (d - Mean);
                Variance += centeredValue * centeredValue;
            }

            Variance /= Data.Count - 1;
        }
    }
}
