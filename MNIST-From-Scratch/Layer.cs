using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using MNIST_From_Scratch.DataTypes;
using System.Text.Json.Serialization;

namespace MNIST_From_Scratch
{
    public sealed class Layer
    {
        [JsonIgnore]
        internal Matrix<float> Weights { get; set; }
        public float[,] SerDesWeights
        {
            get => Weights.ToArray();
            set => Weights = Matrix<float>.Build.DenseOfArray(value);
        }
        [JsonIgnore]
        internal Vector<float> Biases { get; set; }
        public float[] SerDesBiases
        {
            get => Biases.ToArray();
            set => Biases = Vector<float>.Build.DenseOfArray(value);
        }

        public bool Softmax { get; init; }
        
        public Layer(int inputSize, int outputSize, bool softmax = false)
        {
            double stdDevHe = Math.Sqrt(2.0 / inputSize);
            Weights = Matrix<float>.Build.Random(outputSize, inputSize, new Normal(0, stdDevHe));

            Biases = Vector<float>.Build.Dense(outputSize, 0);
            Softmax = softmax;
        }

        public Vector<float> Forward(Vector<float> input)
        {
            Vector<float> output = Weights * input;
            output += Biases;

            // ReLU
            output.MapInplace(x => Math.Max(x, 0));

            return Softmax ? SoftmaxVector(output) : output;
        }

        public Matrix<float> Forward(Matrix<float> inputBatch)
        {
            Matrix<float> outputBatch = Weights * inputBatch;

            // Create matrix of biases matching input batch
            // This passes in an array of references to the same Biases vector
            Matrix<float> biasMatrix = Matrix<float>.Build.DenseOfColumnVectors(
                Enumerable.Repeat(Biases, inputBatch.ColumnCount));

            // Apply biases
            outputBatch += biasMatrix;

            // Apply ReLU if not softmax layer
            if (!Softmax) outputBatch.MapInplace(x => Math.Max(x, 0));
            
            return Softmax ? SoftmaxColumns(outputBatch) : outputBatch;
        }

        /// <summary>
        /// If softmax layer, <code>outputGradients</code> should be the gradient of the input to the softmax function
        /// </summary>
        public LayerGradients Backward(LayerCache layerCache, Matrix<float> outputGradients)
        {
            Matrix<float> inputBatch = layerCache.LayerInputs;
            Matrix<float> outputBatch = layerCache.LayerOutputs;

            Matrix<float> preActivationGradients;
            // No activation if softmax layer
            if (Softmax) preActivationGradients = outputGradients;
            else
            {
                Matrix<float> reluDerivative = outputBatch.Map(x => x > 0 ? 1f : 0f);
                preActivationGradients = outputGradients.PointwiseMultiply(reluDerivative);
            }

            Matrix<float> weightGradients = preActivationGradients * inputBatch.Transpose();
            Matrix<float> inputBatchError = Weights.TransposeThisAndMultiply(preActivationGradients);

            return new LayerGradients(inputBatchError, weightGradients, biasGradients: preActivationGradients);
        }

        public void Update(LayerGradients gradients, float learningRate)
        {
            // Div WeightGradient, which is sum of all gradients in batch, by batchSize, then mul by learningRate
            Weights -= gradients.WeightGradient / gradients.BiasGradients.ColumnCount * learningRate;
            Biases -= gradients.BiasGradients.RowSums() / gradients.BiasGradients.ColumnCount * learningRate;
        }

        static Vector<float> SoftmaxVector(Vector<float> input)
        {
            var result = input.Clone();

            // Shift values <= 0 to prevent overflow. Doesn't affect result
            result -= result.Max();
            result = result.PointwiseExp();
            float sum = result.Sum();
            return result / sum;
        }

        static Matrix<float> SoftmaxColumns(Matrix<float> matrix)
        {
            var result = matrix.Clone();

            for (int columnIndex = 0; columnIndex < matrix.ColumnCount; columnIndex++)
            {
                var column = matrix.Column(columnIndex);
                result.SetColumn(columnIndex, SoftmaxVector(column));
            }

            return result;
        }
    }
}