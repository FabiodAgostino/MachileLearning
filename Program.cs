using MachineLearning.Services2;
using System;
using System.IO;

namespace MachineLearning
{
    internal class Program
    {
        [STAThread]
        static void Main()
        {
            //if (!File.Exists("MuloDetector.zip"))
            //{
            //    Console.WriteLine("🎓 Modello non trovato, avvio training...");
            //    var trainer = new ModelTrainer("Dataset", "MuloModel.zip", "MuloRegressorX.zip", "MuloRegressorY.zip"); 
            //    trainer.Train();
            //}

            // Imposta i percorsi
            string datasetPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Dataset");
            string modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Models", "mulo-detector.zip");

            // Assicurati che la directory Models esista
            Directory.CreateDirectory(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Models"));

            // Inizializza il detector
            var detector = new MuloDetectorObjectDetection2(datasetPath, modelPath);

            // Addestra il modello (da eseguire una tantum)
            detector.TrainAndSaveModel();


            //Application.EnableVisualStyles();
            //Application.SetCompatibleTextRenderingDefault(false); //
            //var labeler = new CoordinateLabelerService("Dataset/dataset.csv");
            //labeler.Run();

            //var r = new RenameImages();
            //r.Rename();

            //var basePath = AppDomain.CurrentDomain.BaseDirectory;
            //var datasetPath = Path.Combine(basePath, "Dataset");
            //var trainer = new ModelTrainer(
            //    datasetPath: datasetPath,
            //    modelSavePath: "MuloDetector.zip");
            //trainer.Train();
            //Console.WriteLine("Finito");
            //if (args.Length > 0 && args[0] == "--train")
            //{
            //    //Console.WriteLine("Avvio addestramento modello...");
            //    //var trainer = new ModelTrainer(
            //    //    datasetPath: "Dataset",
            //    //    modelSavePath: "MuloDetector.zip");
            //    //trainer.Train();
            //}
            //else if (args.Length > 0 && args[0] == "--collect")
            //{
            //    Console.WriteLine("Avvio raccolta dataset...");
            //    var datasetBuilder = new Dataset();
            //    datasetBuilder.RaccogliDataset(numScreenshots: 200);
            //}
            //else
            //{
            //    Console.WriteLine("Avvio detector in tempo reale...");
            //    // Codice per rilevamento in tempo reale (implementato più avanti)
            //}
        }



    }
}