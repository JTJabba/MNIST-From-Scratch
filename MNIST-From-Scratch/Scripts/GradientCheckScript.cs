using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using MNIST_From_Scratch.DataTypes;

namespace MNIST_From_Scratch.Scripts
{
    internal sealed class GradientCheckScript : Script
    {
        public override string Name => "GradientCheck";

        public override void Run(Dictionary<string, string> arguments)
        {
            const int SIZE = 5;
            const float STEP = 1e-5f;

            int testCount = GetArgument<int>(arguments, "testCount");

            float weightDeviationSum = 0;
            for (int i = 0; i < testCount; i++)
            {
                NeuralNet neuralNet = new();
                neuralNet.Layers.Add(new Layer(SIZE, SIZE, true));

                Matrix<float> input = Matrix<float>.Build.Random(
                    SIZE, 1, new Normal(0, 1));
                Matrix<float> randomTarget = Matrix<float>.Build.Random(SIZE, 1, new Normal(0, .75));

                LayerCache[] networkCache = neuralNet.ForwardTraining(input);
                LayerGradients[] networkGradients = neuralNet.Backward(networkCache, randomTarget);

                float benchmarkLoss = NeuralNet.ComputeCrossEntropyLoss(
                    networkCache.Last().LayerOutputs, randomTarget);

                {
                    NeuralNet clonedNetwork = CloneNetwork(neuralNet);
                    clonedNetwork.Layers[0].Weights[0, 0] += STEP;
                    LayerCache[] clonedNetworkCache = clonedNetwork.ForwardTraining(input);
                    float clonedNetworkLoss = NeuralNet.ComputeCrossEntropyLoss(
                        clonedNetworkCache.Last().LayerOutputs, randomTarget);
                    float numericalWeightGradient = (clonedNetworkLoss - benchmarkLoss) / STEP;
                    weightDeviationSum += Math.Abs(networkGradients[0].WeightGradient[0, 0] - numericalWeightGradient);
                    Console.WriteLine(
                        "Calculated weight gradient: " + networkGradients[0].WeightGradient[0, 0] +
                        "\nNumerical weight gradient: " + numericalWeightGradient);
                }
            }
            Console.WriteLine("Average weight deviation: " + weightDeviationSum / testCount);


            NeuralNet CloneNetwork(NeuralNet neuralNet) =>
                NeuralNet.Deserialize(neuralNet.Serialize());
        }
    }
}
