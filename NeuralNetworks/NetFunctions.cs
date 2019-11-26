using System;

namespace NeuralNetworks
{
    public static class NetFunctions
    {
        public static double Sigmoid(double x)
        {
            double e_x = Math.Exp(x);

            return e_x / (1.0 + e_x);
        }

        public static double SigmoidInv(double y)
        {
            return Math.Log(y / (1 - y));
        }

        public static double Sgn(double d)
        {
            if (d == 0) return 0;

            return d > 0 ? 1 : -1;
        }
    }
}
