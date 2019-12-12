using System;

namespace NeuralNetworks
{
    public delegate 

    public abstract class NetFunction
    {
        public bool ApplyOnOutput { get; private set; }

        public abstract double Activation(double d);

        public abstract double ActivationDeriv(double d);

        public static double Sgn(double d)
        {
            if (d == 0) return 0;

            return d > 0 ? 1 : -1;
        }
    }

    public class Rectifier : NetFunction
    {
        public override double Activation(double x)
        {
            return Math.Max(0, x);
        }

        public override double ActivationDeriv(double x)
        {
            return (x >= 0) ? 1 : 0;
        }
    }

    public class Sigmoid : NetFunction
    {
        public override double Activation(double x)
        {
            double e_x = Math.Exp(x);

            return e_x / (1.0 + e_x);
        }

        public override double ActivationDeriv(double d)
        {
            return y * (1 - y);
        }
    }
}
