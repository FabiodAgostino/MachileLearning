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
        private readonly string _classifierModelPath;
        private readonly string _regressorModelPathX;
        private readonly string _regressorModelPathY;

        public ModelTrainer(string datasetPath, string classifierModelPath, string regressorModelPathX, string regressorModelPathY)
        {
            _mlContext = new MLContext(seed: 42);
            _datasetPath = datasetPath;
            _classifierModelPath = classifierModelPath;
            _regressorModelPathX = regressorModelPathX;
            _regressorModelPathY = regressorModelPathY;
        }

        public void Train()
        {
            Console.WriteLine("🔍 Caricamento immagini...");
            var allData = LoadImagesFromDirectory(_datasetPath);
            var classifierData = _mlContext.Data.LoadFromEnumerable(allData);

            Console.WriteLine("🧪 Split classificatore...");
            var trainTestSplit = _mlContext.Data.TrainTestSplit(classifierData, testFraction: 0.2);

            TrainClassifier(trainTestSplit.TrainSet, trainTestSplit.TestSet);

            Console.WriteLine("🔎 Filtro solo 'Mulo' per regressione...");
            var regressionData = allData.Where(d => d.Label == "Mulo").ToList();
            var regressorDataView = _mlContext.Data.LoadFromEnumerable(regressionData);
            var regressionSplit = _mlContext.Data.TrainTestSplit(regressorDataView, testFraction: 0.2);

            TrainRegressor(regressionSplit, _regressorModelPathX, nameof(MuloImageData.X));
            TrainRegressor(regressionSplit, _regressorModelPathY, nameof(MuloImageData.Y));
        }

        private void TrainClassifier(IDataView trainData, IDataView testData)
        {
            Console.WriteLine("🏁 Inizio addestramento classificatore...");

            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey(
                     outputColumnName: "LabelKey", inputColumnName: "Label")
     .Append(_mlContext.Transforms.LoadRawImageBytes(
         outputColumnName: "Image", imageFolder: _datasetPath, inputColumnName: nameof(MuloImageData.ImagePath)))
     .Append(_mlContext.MulticlassClassification.Trainers.ImageClassification(new ImageClassificationTrainer.Options
     {
         FeatureColumnName = "Image",
         LabelColumnName = "LabelKey",
         Arch = ImageClassificationTrainer.Architecture.ResnetV2101,
         Epoch = 50,
         BatchSize = 10,
         LearningRate = 0.01f,
         MetricsCallback = m =>
         {
             if (m.Train != null)
                 Console.WriteLine($"Epoca {m.Train.Epoch} - Accuracy: {m.Train.Accuracy:P2}");
         }
     }))
     .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));

            var model = pipeline.Fit(trainData);

            var predictions = model.Transform(testData);
            var metrics = _mlContext.MulticlassClassification.Evaluate(predictions, labelColumnName: "LabelKey");

            Console.WriteLine($"✅ Classificatore completato - Accuratezza: {metrics.MicroAccuracy:P2}");

            _mlContext.Model.Save(model, trainData.Schema, _classifierModelPath);
            Console.WriteLine($"💾 Modello classificatore salvato in {_classifierModelPath}");
        }

        private void TrainRegressor(DataOperationsCatalog.TrainTestData split, string modelPath, string targetColumn)
        {
            Console.WriteLine($"📈 Inizio regressione per {targetColumn}...");

            var pipeline = _mlContext.Transforms.LoadImages(
                    outputColumnName: "inputImage",
                    imageFolder: _datasetPath,
                    inputColumnName: nameof(MuloImageData.ImagePath))
                .Append(_mlContext.Transforms.ResizeImages(
                    outputColumnName: "inputImage",
                    imageWidth: 224,
                    imageHeight: 224,
                    inputColumnName: "inputImage"))
                .Append(_mlContext.Transforms.ExtractPixels(
                    outputColumnName: "Image",
                    inputColumnName: "inputImage"))
                .Append(_mlContext.Regression.Trainers.FastTree(
                    labelColumnName: targetColumn,
                    featureColumnName: "Image"));

            var model = pipeline.Fit(split.TrainSet);

            var predictions = model.Transform(split.TestSet);
            var metrics = _mlContext.Regression.Evaluate(predictions, labelColumnName: targetColumn);

            Console.WriteLine($"✅ Regressore {targetColumn} completato - R²: {metrics.RSquared:0.##}");

            _mlContext.Model.Save(model, split.TrainSet.Schema, modelPath);
            Console.WriteLine($"💾 Modello regressore {targetColumn} salvato in {modelPath}");
        }


        private List<MuloImageData> LoadImagesFromDirectory(string folder)
        {
            var images = new List<MuloImageData>();
            var categories = new[] { "Mulo", "NonMulo" };

            // Carica le coordinate da CSV
            var labelDict = new Dictionary<string, (float X, float Y)>();
            var csvPath = Path.Combine(folder, "dataset.csv");

            if (File.Exists(csvPath))
            {
                var lines = File.ReadAllLines(csvPath).Skip(1); // Salta intestazione
                foreach (var line in lines)
                {
                    var parts = line.Split(',');
                    if (parts.Length >= 4)
                    {
                        var fileName = parts[0]; // es: Mulo1.png
                        var x = float.Parse(parts[2]);
                        var y = float.Parse(parts[3]);
                        labelDict[fileName] = (x, y);
                    }
                }
            }

            foreach (var category in categories)
            {
                var dir = Path.Combine(folder, category);
                if (!Directory.Exists(dir)) continue;

                foreach (var file in Directory.GetFiles(dir, "*.png"))
                {
                    var fileName = Path.GetFileName(file); // es: Mulo1.png

                    float x = 0, y = 0;
                    if (category == "Mulo" && labelDict.TryGetValue(fileName, out var coords))
                    {
                        x = coords.X;
                        y = coords.Y;
                    }

                    images.Add(new MuloImageData
                    {
                        ImagePath = Path.Combine(category, fileName),
                        Label = category,
                        X = x,
                        Y = y
                    });
                }
            }

            return images;
        }


    }

    public class MuloImageData
    {
        public string ImagePath { get; set; }
        public string Label { get; set; }
        public float X { get; set; }
        public float Y { get; set; }

        public static MuloImageData FromFile(string imagePath, string label, float x, float y)
            => new MuloImageData { ImagePath = imagePath, Label = label, X = x, Y = y };
    }

}
