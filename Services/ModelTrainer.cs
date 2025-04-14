using MachineLearningModels;
using Microsoft.ML;
using Microsoft.ML.Vision;
using System.Diagnostics;

namespace MachineLearning.Services
{
    public class ModelTrainer
    {
        private readonly MLContext _mlContext;
        private readonly string _datasetPath;
        private readonly string _modelSavePath;
        private readonly string _categoryToTrain;


        public ModelTrainer(string datasetPath, string modelSavePath, string categoryToTrain="Mulo")
        {
            _mlContext = new MLContext(seed: 42);
            _datasetPath = datasetPath;
            _modelSavePath = modelSavePath;
            _categoryToTrain = categoryToTrain;
        }

        public void Train()
        {
            Console.WriteLine($"Caricamento dataset per la categoria {_categoryToTrain}...");
            var data = LoadImagesFromDirectory(_datasetPath, _categoryToTrain);
            var dataView = _mlContext.Data.LoadFromEnumerable(data);

            // Stampa lo schema delle colonne per verificare i tipi
            var preview = dataView.Preview();
            foreach (var column in preview.Schema)
            {
                Console.WriteLine($"Nome Colonna: {column.Name}, Tipo: {column.Type}");
            }


            var trainTestData = _mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

            Console.WriteLine($"Immagini caricate: {data.Count} (Mulo: {data.Count(x => x.Label == "Mulo")}, NonMulo: {data.Count(x => x.Label == "NonMulo")})");

            Console.WriteLine("Definizione pipeline di addestramento con ResNet...");
            var pipeline = BuildTrainingPipeline();

            Console.WriteLine("Inizio addestramento...");
            var stopwatch = Stopwatch.StartNew();
            var model = pipeline.Fit(trainTestData.TrainSet);
            stopwatch.Stop();
            Console.WriteLine($"Addestramento completato in {stopwatch.ElapsedMilliseconds / 1000.0} secondi");

            // Valutazione del modello
            Console.WriteLine("Valutazione del modello...");
            var predictions = model.Transform(trainTestData.TestSet);
            var metrics = _mlContext.MulticlassClassification.Evaluate(predictions, labelColumnName: "LabelKey");
            Console.WriteLine($"Accuratezza: {metrics.MicroAccuracy:P2}");
            Console.WriteLine($"LogLoss: {metrics.LogLoss:#.##}");

            // Salvataggio del modello
            Console.WriteLine($"Salvataggio del modello su {_modelSavePath}...");
            _mlContext.Model.Save(model, trainTestData.TrainSet.Schema, _modelSavePath);
            Console.WriteLine("Modello salvato con successo");
        }

        private IEstimator<ITransformer> BuildTrainingPipeline()
        {
            // Mappatura della label in chiavi numeriche
            var mapValueToKey = _mlContext.Transforms.Conversion.MapValueToKey(
                inputColumnName: "Label",
                outputColumnName: "LabelKey");

            // Caricamento raw delle immagini
            var loadRawImages = _mlContext.Transforms.LoadRawImageBytes(
                outputColumnName: "Image",
                imageFolder: _datasetPath,
                inputColumnName: nameof(CategoryImageData.ImagePath));

            // Definizione del trainer Image Classification con ResNet
            var trainer = _mlContext.MulticlassClassification.Trainers.ImageClassification(
                new ImageClassificationTrainer.Options
                {
                    FeatureColumnName = "Image",
                    LabelColumnName = "LabelKey",
                    Arch = ImageClassificationTrainer.Architecture.ResnetV2101,
                    Epoch = 50,
                    BatchSize = 10,
                    LearningRate = 0.01f,
                    MetricsCallback = (metrics) => Console.WriteLine(metrics),
                    ValidationSet = null
                });

            // Mappatura delle chiavi ai valori originali dopo la predizione
            var mapKeyToValue = _mlContext.Transforms.Conversion.MapKeyToValue(
                inputColumnName: "PredictedLabel",
                outputColumnName: "PredictedLabelValue");

            // Costruzione della pipeline
            var pipeline = mapValueToKey
                .Append(loadRawImages)
                .Append(trainer)
                .Append(mapKeyToValue);

            return pipeline;
        }

        private List<CategoryImageData> LoadImagesFromDirectory(string directory, string category)
        {
            var data = new List<CategoryImageData>();

            // Carica immagini positive
            var positiveDir = Path.Combine(directory, category);
            if (Directory.Exists(positiveDir))
            {
                var files = Directory.GetFiles(positiveDir, "*.png");
                foreach (var file in files)
                {
                    data.Add(CategoryImageData.FromFile(file, category));
                }
            }

            // Carica immagini negative
            var negativeDir = Path.Combine(directory, $"Non{category}");
            if (Directory.Exists(negativeDir))
            {
                var files = Directory.GetFiles(negativeDir, "*.png");
                foreach (var file in files)
                {
                    data.Add(CategoryImageData.FromFile(file, $"Non{category}"));
                }
            }

            return data;
        }
    }

 
}
