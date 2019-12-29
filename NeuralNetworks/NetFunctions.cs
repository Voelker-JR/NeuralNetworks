using System;

namespace NeuralNetworks
{
    /*
        This section of interfaces is to generate an abstract setting
        of which the activation functions can naturally evolve.
    */

    public interface INetActivation
    {
        bool ApplyOnArgument { get; }

        double Apply(double x);

        double ApplyDerivative(double x);
    }

    /*
        This section generates a few activation functions based on
        the interfaces above. These are implemented as Singleton classes,
        which are derived from the concerning interface.
    */
    
    public class RectifierActivation : INetActivation
    {
        private RectifierActivation() { }

        public static RectifierActivation Instance { get; } = new RectifierActivation();

        public bool ApplyOnArgument => true;

        public double Apply(double x)
        {
            return Math.Max(0, x);
        }

        public double ApplyDerivative(double x)
        {
            return (x >= 0) ? 1 : 0;
        }
    }

    public class IdentityActivation : INetActivation
    {
        private IdentityActivation() { }

        public static IdentityActivation Instance { get; } = new IdentityActivation();

        public bool ApplyOnArgument => true;

        public double Apply(double x)
        {
            return x;
        }

        public double ApplyDerivative(double x)
        {
            return 1;
        }
    }

    public class SigmoidActivation : INetActivation
    {
        private SigmoidActivation() { }

        public static SigmoidActivation Instance { get; } = new SigmoidActivation();

        public bool ApplyOnArgument => false;

        public double Apply(double x)
        {
            double e_x = Math.Exp(x);

            return e_x / (1.0 + e_x);
        }

        public double ApplyDerivative(double y)
        {
            return y * (1 - y);
        }
    }

    /*
        This section gathers all relevant functions for
        managing the neural network.
    */

    public static class NetFunctions
    {
        public static readonly RectifierActivation Rectifier =
            RectifierActivation.Instance;

        public static readonly IdentityActivation Identity =
            IdentityActivation.Instance;

        public static readonly SigmoidActivation Sigmoid =
            SigmoidActivation.Instance;

        public static double Sgn(double x)
        {
            if (x == 0) return 0;

            return (x > 0) ? 1 : -1;
        }
    }
}
