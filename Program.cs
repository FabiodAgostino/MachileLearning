using MachineLearning;
using MachineLearning.Services;


var basePath = AppDomain.CurrentDomain.BaseDirectory;
var datasetPath = Path.Combine(basePath, "Dataset");
var trainer = new ModelTrainer(
    datasetPath: datasetPath,
    modelSavePath: "MuloDetector.zip");
trainer.Train();
Console.WriteLine("Finito");
if (args.Length > 0 && args[0] == "--train")
{
    //Console.WriteLine("Avvio addestramento modello...");
    //var trainer = new ModelTrainer(
    //    datasetPath: "Dataset",
    //    modelSavePath: "MuloDetector.zip");
    //trainer.Train();
}
else if (args.Length > 0 && args[0] == "--collect")
{
    Console.WriteLine("Avvio raccolta dataset...");
    var datasetBuilder = new Dataset();
    datasetBuilder.RaccogliDataset(numScreenshots: 200);
}
else
{
    Console.WriteLine("Avvio detector in tempo reale...");
    // Codice per rilevamento in tempo reale (implementato più avanti)
}