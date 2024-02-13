using MathNet.Numerics.LinearAlgebra;

namespace MNIST_From_Scratch.DataTypes
{
    public sealed class LayerCache
    {
        public Matrix<float> LayerInputs { get; set; }
        public Matrix<float> LayerOutputs { get; set; }

        public LayerCache(Matrix<float> layerInputs, Matrix<float> layerOutputs)
        {
            LayerInputs = layerInputs;
            LayerOutputs = layerOutputs;
        }
    }
}
