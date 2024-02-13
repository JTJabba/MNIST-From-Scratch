using MathNet.Numerics.LinearAlgebra;
using MNIST_From_Scratch.DataTypes;

namespace MNIST_From_Scratch.Scripts
{
    internal sealed class TrainingScript : Script
    {
        public override string Name => "Training";

        public override void Run(Dictionary<string, string> arguments)
        {
            string trainImagesPath = GetArgument<string>(arguments, nameof(trainImagesPath));
            string testImagesPath = GetArgument<string>(arguments, nameof(testImagesPath));
            string trainLabelsPath = GetArgument<string>(arguments, nameof(trainLabelsPath));
            string testLabelsPath = GetArgument<string>(arguments, nameof(testLabelsPath));
            int batchSize = GetArgument<int>(arguments, nameof(batchSize));
            float learningRate = GetArgument<float>(arguments, nameof(learningRate));
            int steps = GetArgument<int>(arguments, nameof(steps));
            int testFrequency = GetArgument<int>(arguments, nameof(testFrequency));
            int testBatchSize = GetArgument<int>(arguments, nameof(testBatchSize));
            int checkpointSaveFrequency = GetArgument<int>(arguments, nameof(checkpointSaveFrequency));
            string checkpointSavePath = GetArgument<string>(arguments, nameof(checkpointSavePath));
            bool loadFromCheckpoint = TryGetArgument(arguments, "checkpointLoadPath", out string? checkpointLoadPath);

            NeuralNet neuralNet;
            if (loadFromCheckpoint)
            {
                neuralNet = NeuralNet.LoadFromFile(checkpointLoadPath!);
            } else
            {
                neuralNet = new NeuralNet();
                neuralNet.Layers.Add(new Layer(784, 20));
                neuralNet.Layers.Add(new Layer(20, 20));
                neuralNet.Layers.Add(new Layer(20, 10, softmax: true));
                neuralNet.WriteToFile(checkpointSavePath + "checkpoint_initialization.json");
            }

            Matrix<float> trainingImages = DataProcessing.LoadIdx3Images(trainImagesPath);
            Matrix<float> testImages = DataProcessing.LoadIdx3Images(testImagesPath);
            Vector<float> trainingLabels = DataProcessing.LoadIdx1Labels(trainLabelsPath);
            Vector<float> testLabels = DataProcessing.LoadIdx1Labels(testLabelsPath);

            int stepsCompleted = 0;
            foreach (var (batchImages, batchLabels) in GetBatches(trainingImages, trainingLabels, batchSize))
            {
                LayerCache[] networkCache = neuralNet.ForwardTraining(batchImages);
                LayerGradients[] networkGradients = neuralNet.Backward(
                    networkCache, DataProcessing.ConvertLabelsToOneHot(batchLabels));
                // Update, div learning rate by 2 if over 50% trained
                neuralNet.Update(networkGradients, learningRate/* / (1 + stepsCompleted / steps)*/);

                stepsCompleted++;
                if (stepsCompleted % testFrequency == 0) TestLoss();
                if (stepsCompleted >= steps)
                {
                    neuralNet.WriteToFile(checkpointSavePath + "checkpoint_final" + ".json");
                    return;
                }
                if (stepsCompleted % checkpointSaveFrequency == 0)
                {
                    neuralNet.WriteToFile(checkpointSavePath + "checkpoint_" + stepsCompleted / checkpointSaveFrequency + ".json");
                }
            }

            void TestLoss()
            {
                var (batchImages, batchLabels) = GetBatches(testImages, testLabels, testBatchSize).First();
                LayerCache[] networkCache = neuralNet.ForwardTraining(batchImages);
                Matrix<float> output = networkCache.Last().LayerOutputs;
                Vector<float> outputDist = output.RowSums() / output.ColumnCount;
                float loss = NeuralNet.ComputeCrossEntropyLoss(output, DataProcessing.ConvertLabelsToOneHot(batchLabels));

                float avgHighest = output.EnumerateColumns().Select(column => column.Maximum()).Sum() / output.ColumnCount;
                Console.WriteLine($"Loss: {loss}, step {stepsCompleted} of {steps} ({(float)stepsCompleted / steps * 100:F2}%) Avg highest: {avgHighest}");
                //Console.WriteLine($"Avg distribution: {outputDist[0]}, {outputDist[1]}, {outputDist[2]}, {outputDist[3]}, {outputDist[4]}, {outputDist[5]}, {outputDist[6]}, {outputDist[7]}, {outputDist[8]}, {outputDist[9]}, ");
            }
        }


        IEnumerable<(Matrix<float>, Vector<float>)> GetBatches(Matrix<float> images, Vector<float> labels, int batchSize)
        {
            int batchNumberInEpoch = 0;
            int trainingSetSize = labels.Count;
            int[] randomizedIndexes = Array.Empty<int>();
            if (batchSize > trainingSetSize) throw new ArgumentException(
                $"Requested batchSize of {batchSize} exceeds trainingSetSize of {trainingSetSize}");
            while (true)
            {
                batchNumberInEpoch++;
                if (batchNumberInEpoch == 1) // Reshuffle indexes at start of epoch
                {
                    randomizedIndexes = Enumerable.Range(0, trainingSetSize).ToArray();
                    Shuffle(randomizedIndexes);
                }

                // Calculate the start index for the current batch
                int start = (batchNumberInEpoch - 1) * batchSize;
                // If there's not a full batch left continue to the next epoch
                if (start + batchSize > trainingSetSize)
                {
                    batchNumberInEpoch = 0;
                    continue;
                }

                // Get array of batchIndicies, can't use span easily in next step
                int[] batchIndices = randomizedIndexes.Skip(start).Take(batchSize).ToArray();

                Matrix<float> batchImages = Matrix<float>.Build.DenseOfColumnVectors(
                    batchIndices.Select(index => images.Column(index)));
                Vector<float> batchLabels = Vector<float>.Build.DenseOfEnumerable(
                    batchIndices.Select(index => labels[index]));

                yield return (batchImages, batchLabels);
            }

            void Shuffle(int[] array)
            {
                Random rand = new Random();
                for (int i = array.Length - 1; i > 0; i--)
                {
                    int j = rand.Next(i + 1);
                    int temp = array[i];
                    array[i] = array[j];
                    array[j] = temp;
                }
            }
        }
    }
}
