using MathNet.Numerics.LinearAlgebra;

namespace MNIST_From_Scratch.Scripts
{
    internal sealed class DecodeImageScript : Script
    {
        public override string Name => "DecodeImage";

        public override void Run(Dictionary<string, string> arguments)
        {
            string imagesPath = GetArgument<string>(arguments, "imagesPath");
            int imageIndex = GetArgument<int>(arguments, "imageIndex");
            string outPath = GetArgument<string>(arguments, "outPath");

            Matrix<float> encodedImages = DataProcessing.LoadIdx3Images(imagesPath);
            DataProcessing.SaveImageFromMatrixColumn(encodedImages, imageIndex, outPath);
        }
    }
}
