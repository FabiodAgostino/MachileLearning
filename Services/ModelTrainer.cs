using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Vision;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace MachineLearning.Services
{
    public class ModelTrainer
    {
        private readonly MLContext _mlContext;
        private readonly string _datasetPath;
        private readonly string _modelSavePath;

        public ModelTrainer(string datasetPath, string modelSavePath)
        {
            _mlContext = new MLContext(seed: 42);
            _datasetPath = datasetPath;
            _modelSavePath = modelSavePath;
        }

        public void Train()
        {
            Console.WriteLine("Caricamento dataset...");
            var data = LoadImagesFromDirectory(_datasetPath);
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

            // Caricamento raw delle immagini: questa trasformazione legge il file immagine e lo converte in un array di byte.
            var loadRawImages = _mlContext.Transforms.LoadRawImageBytes(
                outputColumnName: "Image",
                imageFolder: _datasetPath,
                inputColumnName: nameof(MuloImageData.ImagePath));

            // NOTA: se vuoi eseguire un pre-processing (ad es. resize) potresti farlo in anticipo,
            // oppure utilizzare una trasformazione custom, visto che LoadRawImageBytes restituisce i byte dell'immagine originale.

            // Definizione del trainer Image Classification con ResNet.
            // Il trainer utilizza il column "Image" che ora è di tipo VarVector<byte> come richiesto.
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
                    ValidationSet = null // Se disponi di un validation set puoi impostarlo qui
                });

            // Mappatura delle chiavi ai valori originali dopo la predizione
            var mapKeyToValue = _mlContext.Transforms.Conversion.MapKeyToValue(
                inputColumnName: "PredictedLabel",
                outputColumnName: "PredictedLabelValue");

            // Costruzione della pipeline concatenando i vari trasformatori
            var pipeline = mapValueToKey
                .Append(loadRawImages)
                .Append(trainer)
                .Append(mapKeyToValue);

            return pipeline;
        }

        private List<MuloImageData> LoadImagesFromDirectory(string directory)
        {
            var data = new List<MuloImageData>();

            // Carica immagini positive (Mulo)
            var muloDir = Path.Combine(directory, "Mulo");
            if (Directory.Exists(muloDir))
            {
                var files = Directory.GetFiles(muloDir, "*.png");
                foreach (var file in files)
                {
                    data.Add(MuloImageData.FromFile(file, "Mulo"));
                }
            }

            // Carica immagini negative (NonMulo)
            var nonMuloDir = Path.Combine(directory, "NonMulo");
            if (Directory.Exists(nonMuloDir))
            {
                var files = Directory.GetFiles(nonMuloDir, "*.png");
                foreach (var file in files)
                {
                    data.Add(MuloImageData.FromFile(file, "NonMulo"));
                }
            }

            return data;
        }
    }

    public class MuloImageData
    {
        [LoadColumn(0)]
        public string ImagePath { get; set; }

        [LoadColumn(1)]
        public string Label { get; set; }

        public static MuloImageData FromFile(string imagePath, string label)
        {
            return new MuloImageData
            {
                ImagePath = imagePath,
                Label = label
            };
        }
    }

    public class MuloPrediction
    {
        public string ImagePath { get; set; }
        public string Label { get; set; }
        public float[] Score { get; set; }
        public string PredictedLabelValue { get; set; }
    }
}
