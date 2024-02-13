using MathNet.Numerics.LinearAlgebra;

namespace MNIST_From_Scratch.DataTypes
{
    public sealed class LayerGradients
    {
        /// <summary>
        /// Can be null for layer 0. RowCount == batchSize
        /// </summary>
        public Matrix<float>? InputGradient { get; set; }
        /// <summary>
        /// Sum of all gradients from a batch
        /// </summary>
        public Matrix<float> WeightGradient { get; set; }
        /// <summary>
        /// RowCount == batchSize
        /// </summary>
        public Matrix<float> BiasGradients { get; set; }

        public LayerGradients(Matrix<float>? inputGradient, Matrix<float> weightGradient, Matrix<float> biasGradients)
        {
            InputGradient = inputGradient;
            WeightGradient = weightGradient;
            BiasGradients = biasGradients;
        }
    }
}
