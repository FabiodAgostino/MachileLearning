using MachineLearning.NewServices;
using MachineLearning.Services;
using System;
using System.IO;

/**
 * Questo servizio è progettato per rilevare la presenza e la posizione degli oggetti nell'ambiente
 * di gioco di Ultima Online. Il servizio implementa una soluzione di visione artificiale che:
 * 
 * 1. CLASSIFICAZIONE: Identifica se in un'immagine è presente un oggetto o meno
 * 2. LOCALIZZAZIONE: Determina le coordinate precise (X, Y) del oggetto nell'immagine
 * 
 * 
 * Funzionalità:
 * - Caricamento automatico del dataset da una struttura di cartelle organizzata
 * - Lettura di annotazioni di coordinate da file CSV
 * - Addestramento parallelo di modelli per classificazione e localizzazione
 * - Valutazione delle prestazioni del modello con metriche di accuratezza
 * - Salvataggio dei modelli in un'unica directory per uso futuro
 * - Inferenza su nuove immagini per rilevare e localizzare i muli
 * - Gestione avanzata degli errori durante caricamento e predizione
 * 
 * 
 * Struttura attesa del dataset:
 * /dataset/
 *    /Mulo/ - contiene immagini di muli ovvero gli oggetti (.png)
 *    /NonMulo/ - contiene immagini senza muli ovvero senza oggetti (.png)
 *    dataset.csv - file con formato: ImagePath,IsMulo,X,Y
 * 
 * Note particolari:
 * - Per immagini "NonMulo", le coordinate X,Y vengono impostate a (0,0)
 * - Le coordinate X,Y predette vengono restituite solo quando viene rilevato un mulo
 * - Il modello riporta metriche di accuratezza e performance al termine dell'addestramento
 * 
 * Librerie utilizzate:
 * - Microsoft.ML" Version="4.0.2"
 * - Microsoft.ML.DataView" Version="4.0.2"
 * - Microsoft.ML.ImageAnalytics" Version="4.0.2" 
 * - Microsoft.ML.OnnxRuntime" Version="1.21.0" 
 * - Microsoft.ML (2.0.0+)
 * - Microsoft.ML.OnnxTransformer" Version="4.0.2" 
 * - Microsoft.ML.Vision" Version="4.0.2" 
 * - Microsoft.ML.ImageAnalytics (2.0.0+)
 * - SciSharp.TensorFlow.Redist" Version="1.15.0" 
 * - Microsoft.ML.FastTree" Version="4.0.2" 
 * - SixLabors.ImageSharp" Version="3.1.7" 
 */


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
            var m = new ObjectDetectorImproved(datasetPath, modelPath, "Mulo");
            m.TrainAndSaveModel();
            // Inizializza il detector
            //var detector = new MuloDetectorObjectDetection2(datasetPath, modelPath);
            //string mulo = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Mulo.png");

            //detector.DetectMulo(mulo);

            //// Addestra il modello (da eseguire una tantum)
            //detector.TrainAndSaveModel();


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