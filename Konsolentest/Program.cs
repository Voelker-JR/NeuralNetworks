﻿using System;
using System.Collections.Generic;
using System.Threading;
using NeuralNetworks;
using VectorMath;

namespace Konsolentest
{
    class Program
    {
        static void Main()
        {
            List<int> structure = new List<int>() { 2, 3, 2, 1 };
            Network net = new Network(structure, NetFunctions.Rectifier);

            List<Vector> inputs = new List<Vector>() {
                new Vector(new double[] {0, 0}),
                new Vector(new double[] {0, 1}),
                new Vector(new double[] {1, 0}),
                new Vector(new double[] {1, 1})
            };

            List<Vector> outputs = new List<Vector>() {
                new Vector(new double[] {0}),
                new Vector(new double[] {1}),
                new Vector(new double[] {1}),
                new Vector(new double[] {1})
            };

            List<TrainingPattern> patterns = new List<TrainingPattern>();
            for (int i = 0; i < 4; i++)
                patterns.Add(new TrainingPattern(inputs[i], outputs[i]));

            Training training = new AdamTraining(net, 2, 0.001)
            {
                Patterns = patterns
            };

            CancellationTokenSource source = new CancellationTokenSource();
            CancellationToken token = source.Token;

            // Training
            training.Run(5000, token).Wait();

            // Validation
            foreach (var input in inputs)
            {
                Console.WriteLine(input.ToString() + ": " + net.Feed(input).ToString());
            }

            Console.WriteLine();
            foreach (var layer in net.Layers)
            {
                Console.WriteLine(layer.Weights.Apply(x => Math.Round(x, 2)).ToString());
            }
        }
    }
}
