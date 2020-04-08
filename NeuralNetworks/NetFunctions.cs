using NeuralNetworks.Statistics;
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

        /// <summary>
        /// Initialization function for the weight matrix of a layer.
        /// The values are randomly generated.
        /// </summary>
        /// <param name="random">A Random object to generate the base values.</param>
        /// <param name="outDim">The output dimension of this layer.</param>
        /// <param name="inDim">The input dimension of this layer.</param>
        /// <returns></returns>        
        double WeightInitialization(Random random, int outDim, int inDim);
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

        public double WeightInitialization(Random random, int outDim, int inDim)
        {
            return StatHelper.NextGaussian(random, 0, Math.Sqrt(2.0 / inDim));
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

        public double WeightInitialization(Random random, int outDim, int inDim)
        {
            // Uniform distribution on [-0.5, 0.5)
            return random.NextDouble() - 0.5;
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
            return y * (1.0 - y);
        }

        public double WeightInitialization(Random random, int outDim, int inDim)
        {
            return StatHelper.NextGaussian(random, 0, Math.Sqrt(2.0 / (inDim + outDim)));
        }
    }

    public class SoftplusActivation : INetActivation
    {
        private SoftplusActivation() { }

        public static SoftplusActivation Instance { get; } = new SoftplusActivation();

        public bool ApplyOnArgument => true;

        public double Apply(double x)
        {
            return Math.Log(1.0 + Math.Exp(x));
        }

        public double ApplyDerivative(double x)
        {
            double e_x = Math.Exp(x);

            return e_x / (1.0 + e_x);
        }

        public double WeightInitialization(Random random, int outDim, int inDim)
        {
            return StatHelper.NextGaussian(random, 0, Math.Sqrt(2.0 / (inDim + outDim)));
        }
    }

    public class SELUActivation : INetActivation
    {
        private const double alpha = 1.6732632423543772848170;
        private const double lambda = 1.0507009873554804934193;

        private SELUActivation() { }

        public static SELUActivation Instance { get; } = new SELUActivation();

        public bool ApplyOnArgument => true;

        public double Apply(double x)
        {
            return lambda * ((x > 0) ? x : alpha * Math.Exp(x) - alpha);
        }

        public double ApplyDerivative(double x)
        {
            return lambda * ((x > 0) ? 1 : alpha * Math.Exp(x));
        }

        public double WeightInitialization(Random random, int outDim, int inDim)
        {
            return StatHelper.NextGaussian(random, 0, Math.Sqrt(1.0 / inDim));
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

        public static readonly SoftplusActivation Softplus =
            SoftplusActivation.Instance;

        public static readonly SELUActivation SELU =
            SELUActivation.Instance;

        public static double Sgn(double x)
        {
            if (x == 0) return 0;

            return (x > 0) ? 1 : -1;
        }
    }
}
