using MachineLearning.Services;

var basePath = AppDomain.CurrentDomain.BaseDirectory;
var datasetPath = Path.Combine(basePath, "Dataset");
string modelSavePath = "";
string categoryToTrain = "";

while (true)
{
    Console.WriteLine("Scegli quale modello addestrare: 1)Mulo, 2)Zaino, 3)Interfaccia, 4)AntiMacro");
    var response = Console.ReadLine();
    bool success = Int32.TryParse(response, out int choice);

    if (success)
    {
        switch (choice)
        {
            case 1:
                modelSavePath = "MuloDetector.zip";
                categoryToTrain = "Mulo";
                break;
            case 2:
                modelSavePath = "ZainoDetector.zip";
                categoryToTrain = "Zaino";
                break;
            case 3:
                modelSavePath = "InterfacciaDetector.zip";
                categoryToTrain = "Interfaccia";
                break;
            case 4:
                modelSavePath = "AntiMacroDetector.zip";
                categoryToTrain = "AntiMacro";
                break;
            default:
                Console.WriteLine("Scelta non valida. Riprova.");
                continue;
        }
        if (File.Exists(Path.Combine(datasetPath, modelSavePath)))
        {
            Console.WriteLine($"Il file '{modelSavePath}' esiste già. Vuoi sovrascriverlo? (s/n)");
            var overwriteResponse = Console.ReadLine().ToLower();

            if (overwriteResponse != "s" && overwriteResponse != "si")
            {
                Console.WriteLine("Operazione annullata.");
                continue; // Torna al menu di selezione
            }
        }

        break;
    }
    Console.WriteLine("Input non valido. Inserisci un numero.");
}


var trainer = new ModelTrainer(
    datasetPath: datasetPath,
    modelSavePath: modelSavePath,
    categoryToTrain: categoryToTrain);

trainer.Train();
Console.WriteLine("Finito");