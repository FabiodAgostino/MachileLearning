using DetectorModel.Services;
using System;
using System.IO;
using System.Threading.Tasks;


namespace MachineLearning
{
    internal class Program
    {
        [STAThread]
        static async Task Main()
        {

            // Imposta i percorsi
            string datasetPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Dataset");
            string modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Models", "mulo-detector.zip");
            Directory.CreateDirectory(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Models"));

            var o = new ObjectDetector();
            var pathMule = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "mulo.png");
            var obj = o.DetectObject(pathMule, modelPath);
        }



    }
}