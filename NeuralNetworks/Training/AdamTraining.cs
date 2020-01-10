namespace NeuralNetworks
{
    public class AdamTraining : MiniBatchTraining
    {
        public AdamTraining(Net net, int batchSize, double learningRate)
            : base(net, batchSize)
        {
            LearningRate = learningRate;

            foreach (var layer in net.Layers)
                LayerWrappers.Add(new AdamLayerWrapper(layer, learningRate));
        }

        public double LearningRate { get; }
    }
}
