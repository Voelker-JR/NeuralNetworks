namespace NeuralNetworks
{
    public class RPROPTraining : BatchTraining
    {
        public RPROPTraining(Network net, double decreaseFactor, double increaseFactor)
            : base(net)
        {
            DecreaseFactor = decreaseFactor;
            IncreaseFactor = increaseFactor;

            foreach (var layer in net.Layers)
                LayerWrappers.Add(new RPROPLayerWrapper(layer, decreaseFactor, increaseFactor));
        }

        public double DecreaseFactor { get; }

        public double IncreaseFactor { get; }
    }
}
