using MathNet.Numerics.LinearAlgebra;
using MNIST_From_Scratch.DataTypes;
using Newtonsoft.Json;
using System.IO;

namespace MNIST_From_Scratch
{
    public sealed class NeuralNet
    {
        public List<Layer> Layers = new();

        public static NeuralNet Deserialize(string json)
        {
            var settings = new JsonSerializerSettings { };
            var result = JsonConvert.DeserializeObject<NeuralNet>(json, settings);
            if (result is null)
                throw new JsonSerializationException($"Got null object deserializing NeuralNet");
            return result;
        }

        public static NeuralNet LoadFromFile(string path) => Deserialize(File.ReadAllText(path));

        public static float ComputeCrossEntropyLoss(Matrix<float> predictions, Matrix<float> oneHotLabels)
        {
            int sampleCount = predictions.ColumnCount;
            float loss = 0f;

            for (int i = 0; i < sampleCount; i++)
            {
                for (int j = 0; j < predictions.RowCount; j++)
                {
                    // Safe-guard against log(0) which is undefined
                    float prediction = Math.Max(predictions[j, i], 1e-15f);
                    loss += (float)(-oneHotLabels[j, i] * Math.Log(prediction));
                }
            }

            return loss / sampleCount;
        }

        public NeuralNet() { }

        public string Serialize()
        {
            var settings = new JsonSerializerSettings
            {
                Formatting = Formatting.Indented,
            };
            return JsonConvert.SerializeObject(this, settings);
        }

        public void WriteToFile(string path) => File.WriteAllText(path, Serialize());

        public Vector<float> ForwardInference(Vector<float> input)
        {
            Vector<float> layerOutput = input;

            foreach (var layer in Layers)
            {
                layerOutput = layer.Forward(layerOutput);
            }

            return layerOutput;
        }

        public LayerCache[] ForwardTraining(Matrix<float> input)
        {
            LayerCache[] networkCache = new LayerCache[Layers.Count];
            Matrix<float> layerInput = input, layerOutput;

            for (int i = 0; i < Layers.Count; i++)
            {
                layerOutput = Layers[i].Forward(layerInput);
                networkCache[i] = new LayerCache(layerInput, layerOutput);
                layerInput = layerOutput;
            }

            return networkCache;
        }

        /// <summary>
        /// This implementation assumes the last layer is a softmax layer,
        /// and that it should be optimizing for cross-entropy loss
        /// </summary>
        public LayerGradients[] Backward(LayerCache[] networkCache, Matrix<float> expectedOutput)
        {
            LayerGradients[] gradients = new LayerGradients[Layers.Count];

            // Softmax input gradient to cross-entropy loss
            Matrix<float> errorGradient = networkCache[networkCache.Length - 1].LayerOutputs - expectedOutput;

            // For last layer, networkCache won't have input error of next layer
            gradients[Layers.Count - 1] = Layers.Last().Backward(networkCache.Last(), errorGradient);

            // Get gradients for rest of layers
            for (int i = Layers.Count - 2; i >= 0; i--)
            {
                errorGradient = gradients[i + 1].InputGradient!;
                gradients[i] = Layers[i].Backward(networkCache[i], errorGradient);
            }

            return gradients;
        }

        public void Update(LayerGradients[] gradients, float learningRate)
        {
            for (int i = 0; i < Layers.Count; i++)
            {
                Layers[i].Update(gradients[i], learningRate);
            }
        }
    }
}
