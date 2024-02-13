using MathNet.Numerics.LinearAlgebra;

namespace MNIST_From_Scratch.Scripts
{
    internal sealed class InferenceScript : Script
    {
        public override string Name => "Inference";

        public override void Run(Dictionary<string, string> arguments)
        {
            string modelPath = GetArgument<string>(arguments, "modelPath");
            string imagePath = GetArgument<string>(arguments, "imagePath");

            NeuralNet neuralNet = NeuralNet.LoadFromFile(modelPath);
            Vector<float> image = DataProcessing.GetImageVector(imagePath);

            Vector<float> modelOutput = neuralNet.ForwardInference(image);

            // Get top 3 probabilities with indices
            var top3 = modelOutput
                .EnumerateIndexed()
                .OrderByDescending(pair => pair.Item2) // Sort by probability descending
                .Take(3) // Take top 3
                .Select(pair => new { Digit = pair.Item1, Probability = pair.Item2 * 100 }) // Project to new anonymous type
                .ToList();

            Console.WriteLine("Top 3 probabilities:");
            foreach (var item in top3)
            {
                Console.WriteLine($"    {item.Digit} - {item.Probability:F2}%");
            }
        }
    }
}
