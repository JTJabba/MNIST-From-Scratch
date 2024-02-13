using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.PixelFormats;
using MathNet.Numerics.LinearAlgebra;

namespace MNIST_From_Scratch
{
    internal sealed class DataProcessing
    {
        public static Vector<float> GetImageVector(string imagePath)
        {
            // Load the image, automatically converts to greyscale
            var image = Image.Load<L8>(imagePath);

            // Resize to 28x28 pixels
            image.Mutate(x => x.Resize(28, 28));

            // Convert the 28x28 image to a 784-element vector
            float[] imageVector = new float[28 * 28];
            for (int y = 0; y < 28; y++)
            {
                for (int x = 0; x < 28; x++)
                {
                    // Normalize pixel value to be between 0 and 1
                    imageVector[y * 28 + x] = image[x, y].PackedValue / 255f;
                }
            }

            return Vector<float>.Build.Dense(imageVector);
        }

        public static Matrix<float> LoadIdx3Images(string path)
        {
            using var stream = new FileStream(path, FileMode.Open);
            using var reader = new BinaryReader(stream);

            int magicNumber = ReadBigEndianInt(reader);
            if (magicNumber != 2051) throw new Exception("Invalid magic number in image file");

            int numberOfImages = ReadBigEndianInt(reader);
            int rows = ReadBigEndianInt(reader);
            int cols = ReadBigEndianInt(reader);

            var imagesMatrix = Matrix<float>.Build.Dense(rows * cols, numberOfImages);

            for (int i = 0; i < numberOfImages; i++)
            {
                for (int j = 0; j < rows * cols; j++)
                {
                    // Normalize pixel values to [0, 1] and assign to matrix
                    imagesMatrix[j, i] = reader.ReadByte() / 255f;
                }
            }

            return imagesMatrix;
        }

        public static Vector<float> LoadIdx1Labels(string labelPath)
        {
            using var stream = new FileStream(labelPath, FileMode.Open);
            using var reader = new BinaryReader(stream);

            int magicNumber = ReadBigEndianInt(reader);
            if (magicNumber != 2049) throw new Exception("Invalid magic number in label file");

            int numberOfLabels = ReadBigEndianInt(reader);
            var labels = new float[numberOfLabels];

            for (int i = 0; i < numberOfLabels; i++)
            {
                labels[i] = reader.ReadByte();
            }

            return Vector<float>.Build.Dense(labels);
        }

        public static Matrix<float> ConvertLabelsToOneHot(Vector<float> labels)
        {
            int numLabels = labels.Count;
            var oneHotEncoded = Matrix<float>.Build.Dense(10, numLabels);

            for (int i = 0; i < numLabels; i++)
            {
                byte label = (byte)labels[i];
                oneHotEncoded[label, i] = 1f; // Set the appropriate row to 1 for the label
            }

            return oneHotEncoded;
        }

        /// <summary>
        /// Used for verifying images are loaded correctly
        /// </summary>
        /// <exception cref="ArgumentOutOfRangeException"></exception>
        public static void SaveImageFromMatrixColumn(Matrix<float> imagesMatrix, int columnIndex, string outputPath)
        {
            if (columnIndex < 0 || columnIndex >= imagesMatrix.ColumnCount)
            {
                throw new ArgumentOutOfRangeException(nameof(columnIndex), "Column index is out of range.");
            }

            const int width = 28;
            const int height = 28;
            using var image = new Image<L8>(width, height);

            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    // Calculate the pixel value
                    float pixelValue = imagesMatrix[i * width + j, columnIndex];
                    byte pixelByte = (byte)(pixelValue * 255); // Convert back from [0, 1] to [0, 255]
                    image[j, i] = new L8(pixelByte);
                }
            }

            image.SaveAsPng(outputPath);
        }

        static int ReadBigEndianInt(BinaryReader reader)
        {
            var bytes = reader.ReadBytes(4);
            if (BitConverter.IsLittleEndian)
                Array.Reverse(bytes);
            return BitConverter.ToInt32(bytes, 0);
        }

    }
}
