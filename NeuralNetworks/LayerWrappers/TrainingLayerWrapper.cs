using VectorMath;

namespace NeuralNetworks
{
    public abstract class TrainingLayerWrapper
    {
        public TrainingLayerWrapper(Layer layer)
        {
            AssociatedLayer = layer;

            int rows = layer.Weights.Rows;
            int columns = layer.Weights.Columns;

            WeightsDiff = new Matrix(rows, columns);
        }

        public Layer AssociatedLayer { get; }

        public Matrix WeightsDiff { get; set; }

        public abstract void ApplyBackpropChanges(Matrix changes);

        public abstract void ApplyWeightChanges(int epoch);
    }
}
