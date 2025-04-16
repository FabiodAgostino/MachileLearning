using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Vision;

namespace MachineLearning.Services2
{
    /// <summary>
    /// Classe per il rilevamento dei muli utilizzando transfer learning per la classificazione
    /// e modelli di regressione per stimare le coordinate.
    /// </summary>
    public class MuloDetectorObjectDetection2
    {
        private readonly MLContext _mlContext;
        private readonly string _datasetPath;
        private readonly string _modelPath;
        private readonly int _imageWidth = 224;
        private readonly int _imageHeight = 224;

        public MuloDetectorObjectDetection2(string datasetPath, string modelPath)
        {
            _mlContext = new MLContext(seed: 42);
            _datasetPath = datasetPath;
            _modelPath = modelPath;
        }

        /// <summary>
        /// Addestra e salva il modello combinato.
        /// Per la classificazione viene usato LoadRawImageBytes che produce un VarVector&lt;Byte&gt;,
        /// mentre per la regressione si usa LoadImages, ResizeImages ed ExtractPixels per ottenere il vettore di float.
        /// </summary>
        public void TrainAndSaveModel()
        {
            Console.WriteLine("🔍 Caricamento immagini dal dataset...");
            var imageData = LoadImagesWithAnnotations();
            var dataView = _mlContext.Data.LoadFromEnumerable(imageData);
            var trainTestData = _mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

            Console.WriteLine("🏗️ Costruzione pipeline di addestramento...");

            // ----------------- CLASSIFICAZIONE -----------------
            // Per la classificazione usiamo LoadRawImageBytes:
            // Questa trasformazione legge direttamente i byte grezzi dell'immagine dal file e restituisce un VarVector<Byte>.
            var classificationPipeline = _mlContext.Transforms.Conversion.MapValueToKey(
                                              outputColumnName: "LabelKey",
                                              inputColumnName: nameof(MuloImageData.Label))
                                          .Append(_mlContext.Transforms.LoadRawImageBytes(
                                              outputColumnName: "ImageBytes",
                                              imageFolder: _datasetPath,
                                              inputColumnName: nameof(MuloImageData.ImagePath)))
                                          .Append(_mlContext.MulticlassClassification.Trainers.ImageClassification(
                                              new ImageClassificationTrainer.Options
                                              {
                                                  FeatureColumnName = "ImageBytes",
                                                  LabelColumnName = "LabelKey",
                                                  Arch = ImageClassificationTrainer.Architecture.ResnetV2101,
                                                  Epoch = 50,
                                                  BatchSize = 10,
                                                  LearningRate = 0.01f,
                                                  MetricsCallback = (metrics) => Console.WriteLine(metrics)
                                              }))
                                          .Append(_mlContext.Transforms.Conversion.MapKeyToValue(
                                              outputColumnName: "PredictedLabel",
                                              inputColumnName: "PredictedLabel"));

            // ----------------- REGRESSIONE -----------------
            // Per la regressione usiamo la pipeline LoadImages -> ResizeImages -> ExtractPixels (default: Vector<Single>)
            var regressionPipeline = _mlContext.Transforms.LoadImages(
                                            outputColumnName: "Image",
                                            imageFolder: _datasetPath,
                                            inputColumnName: nameof(MuloImageData.ImagePath))
                                     .Append(_mlContext.Transforms.ResizeImages(
                                            outputColumnName: "Image",
                                            imageWidth: _imageWidth,
                                            imageHeight: _imageHeight,
                                            inputColumnName: "Image"))
                                     .Append(_mlContext.Transforms.ExtractPixels(
                                            outputColumnName: "ImagePixels",
                                            inputColumnName: "Image",
                                            interleavePixelColors: true,
                                            offsetImage: 0));

            var xRegressionPipeline = regressionPipeline
                .Append(_mlContext.Transforms.CopyColumns(
                    outputColumnName: "XFeatures",
                    inputColumnName: "ImagePixels"))
                .Append(_mlContext.Regression.Trainers.Sdca(
                    labelColumnName: nameof(MuloImageData.X),
                    featureColumnName: "XFeatures"));

            var yRegressionPipeline = regressionPipeline
                .Append(_mlContext.Transforms.CopyColumns(
                    outputColumnName: "YFeatures",
                    inputColumnName: "ImagePixels"))
                .Append(_mlContext.Regression.Trainers.Sdca(
                    labelColumnName: nameof(MuloImageData.Y),
                    featureColumnName: "YFeatures"));

            Console.WriteLine("🏁 Inizio addestramento modelli...");

            Console.WriteLine("   Addestramento classificatore (Mulo/NonMulo)...");
            var classificationModel = classificationPipeline.Fit(trainTestData.TrainSet);

            Console.WriteLine("   Addestramento regressore per X...");
            var xRegressionModel = xRegressionPipeline.Fit(trainTestData.TrainSet);

            Console.WriteLine("   Addestramento regressore per Y...");
            var yRegressionModel = yRegressionPipeline.Fit(trainTestData.TrainSet);

            // Per salvare la trasformazione base per la regressione
            var transformationPipeline = regressionPipeline.Fit(trainTestData.TrainSet);

            var combinedModel = new CombinedMuloModel(
                _mlContext,
                transformationPipeline,
                classificationModel,
                xRegressionModel,
                yRegressionModel);

            Console.WriteLine("💾 Salvataggio modello combinato...");
            combinedModel.Save(_modelPath);
            Console.WriteLine($"✅ Modello salvato con successo in: {_modelPath}");

            // Valutazione del modello di classificazione
            Console.WriteLine("🔎 Valutazione modello di classificazione...");
            var testClassificationPipeline = _mlContext.Transforms.Conversion.MapValueToKey(
                                                  outputColumnName: "LabelKey",
                                                  inputColumnName: nameof(MuloImageData.Label))
                                              .Append(_mlContext.Transforms.LoadRawImageBytes(
                                                  outputColumnName: "ImageBytes",
                                                  imageFolder: _datasetPath,
                                                  inputColumnName: nameof(MuloImageData.ImagePath)))
                                              .Fit(trainTestData.TestSet)
                                              .Transform(trainTestData.TestSet);
            var testPredictions = classificationModel.Transform(testClassificationPipeline);
            var metrics = _mlContext.MulticlassClassification.Evaluate(
                testPredictions, labelColumnName: "LabelKey");

            Console.WriteLine($"✅ Addestramento completato");
            Console.WriteLine($"   Accuratezza classificazione: {metrics.MicroAccuracy:P2}");
            Console.WriteLine($"   Log Loss: {metrics.LogLoss:F4}");
        }

        /// <summary>
        /// Rileva la presenza e le coordinate del mulo in una immagine.
        /// </summary>
        public MuloDetectionResult DetectMulo(string imagePath)
        {
            CombinedMuloModel combinedModel;
            try
            {
                combinedModel = CombinedMuloModel.Load(_modelPath, _mlContext);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Errore caricamento modello: {ex.Message}");
                return new MuloDetectionResult
                {
                    IsMuloDetected = false,
                    X = 0,
                    Y = 0,
                    Confidence = 0,
                    ErrorMessage = $"Errore caricamento modello: {ex.Message}"
                };
            }

            var imageData = new MuloImageData
            {
                ImagePath = imagePath,
                Label = string.Empty,
                X = 0,
                Y = 0
            };

            MuloImagePrediction prediction;
            try
            {
                prediction = combinedModel.Predict(imageData);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Errore durante la predizione: {ex.Message}");
                return new MuloDetectionResult
                {
                    IsMuloDetected = false,
                    X = 0,
                    Y = 0,
                    Confidence = 0,
                    ErrorMessage = $"Errore durante la predizione: {ex.Message}"
                };
            }

            float confidence = (prediction.Score != null && prediction.Score.Length > 0)
                ? prediction.Score.Max()
                : 0;
            bool isMulo = prediction.PredictedLabel.Equals("Mulo", StringComparison.OrdinalIgnoreCase);

            return new MuloDetectionResult
            {
                IsMuloDetected = isMulo,
                X = isMulo ? prediction.X : 0,
                Y = isMulo ? prediction.Y : 0,
                Confidence = confidence
            };
        }

        /// <summary>
        /// Carica le immagini e le annotazioni dal dataset.
        /// Struttura attesa:
        /// /dataset/
        ///    /Mulo/ - immagini di muli (.png)
        ///    /NonMulo/ - immagini senza muli (.png)
        ///    dataset.csv - CSV con: ImagePath,IsMulo,X,Y
        /// </summary>
        private List<MuloImageData> LoadImagesWithAnnotations()
        {
            var images = new List<MuloImageData>();
            var categories = new[] { "Mulo", "NonMulo" };

            var labelDict = new Dictionary<string, (float X, float Y)>();
            var csvPath = Path.Combine(_datasetPath, "dataset.csv");

            if (File.Exists(csvPath))
            {
                Console.WriteLine($"📄 Caricamento annotazioni da: {csvPath}");
                var lines = File.ReadAllLines(csvPath).Skip(1);
                foreach (var line in lines)
                {
                    var parts = line.Split(',');
                    if (parts.Length >= 4)
                    {
                        var fullPath = parts[0];
                        var fileName = Path.GetFileName(fullPath);

                        if (float.TryParse(parts[2], out var x) &&
                            float.TryParse(parts[3], out var y))
                        {
                            labelDict[fileName] = (x, y);
                        }
                    }
                }
                Console.WriteLine($"   Caricate {labelDict.Count} annotazioni");
            }
            else
            {
                Console.WriteLine($"⚠️ File CSV non trovato: {csvPath}");
            }

            foreach (var category in categories)
            {
                var categoryDir = Path.Combine(_datasetPath, category);
                if (!Directory.Exists(categoryDir))
                {
                    Console.WriteLine($"⚠️ Directory non trovata: {categoryDir}");
                    continue;
                }

                foreach (var file in Directory.GetFiles(categoryDir, "*.png"))
                {
                    var fileName = Path.GetFileName(file);
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

            Console.WriteLine($"📊 Dataset caricato: {images.Count} immagini");
            Console.WriteLine($"   - Muli: {images.Count(i => i.Label == "Mulo")}");
            Console.WriteLine($"   - Non Muli: {images.Count(i => i.Label == "NonMulo")}");
            return images;
        }
    }

    /// <summary>
    /// Classe che combina la pipeline di trasformazione, il classificatore e i regressori.
    /// </summary>
    public class CombinedMuloModel
    {
        private readonly MLContext _mlContext;
        private readonly ITransformer _transformationPipeline;
        private readonly ITransformer _classificationModel;
        private readonly ITransformer _xRegressionModel;
        private readonly ITransformer _yRegressionModel;

        private readonly PredictionEngine<MuloImageData, MuloClassificationPrediction> _classificationEngine;
        private readonly PredictionEngine<MuloImageData, MuloXRegressionPrediction> _xRegressionEngine;
        private readonly PredictionEngine<MuloImageData, MuloYRegressionPrediction> _yRegressionEngine;

        public CombinedMuloModel(
            MLContext mlContext,
            ITransformer transformationPipeline,
            ITransformer classificationModel,
            ITransformer xRegressionModel,
            ITransformer yRegressionModel)
        {
            _mlContext = mlContext;
            _transformationPipeline = transformationPipeline;
            _classificationModel = classificationModel;
            _xRegressionModel = xRegressionModel;
            _yRegressionModel = yRegressionModel;

            _classificationEngine = _mlContext.Model.CreatePredictionEngine<MuloImageData, MuloClassificationPrediction>(_classificationModel);
            _xRegressionEngine = _mlContext.Model.CreatePredictionEngine<MuloImageData, MuloXRegressionPrediction>(_xRegressionModel);
            _yRegressionEngine = _mlContext.Model.CreatePredictionEngine<MuloImageData, MuloYRegressionPrediction>(_yRegressionModel);
        }

        public MuloImagePrediction Predict(MuloImageData input)
        {
            var classificationPrediction = _classificationEngine.Predict(input);
            var xPrediction = _xRegressionEngine.Predict(input);
            var yPrediction = _yRegressionEngine.Predict(input);

            return new MuloImagePrediction
            {
                PredictedLabel = classificationPrediction.PredictedLabel,
                Score = classificationPrediction.Score,
                X = xPrediction.X,
                Y = yPrediction.Y
            };
        }

        public void Save(string directoryPath)
        {
            Directory.CreateDirectory(directoryPath);
            _mlContext.Model.Save(_transformationPipeline, inputSchema: null, Path.Combine(directoryPath, "transform.zip"));
            _mlContext.Model.Save(_classificationModel, inputSchema: null, Path.Combine(directoryPath, "classification.zip"));
            _mlContext.Model.Save(_xRegressionModel, inputSchema: null, Path.Combine(directoryPath, "x_regression.zip"));
            _mlContext.Model.Save(_yRegressionModel, inputSchema: null, Path.Combine(directoryPath, "y_regression.zip"));
        }

        public static CombinedMuloModel Load(string directoryPath, MLContext mlContext)
        {
            var transformationPipeline = mlContext.Model.Load(Path.Combine(directoryPath, "transform.zip"), out _);
            var classificationModel = mlContext.Model.Load(Path.Combine(directoryPath, "classification.zip"), out _);
            var xRegressionModel = mlContext.Model.Load(Path.Combine(directoryPath, "x_regression.zip"), out _);
            var yRegressionModel = mlContext.Model.Load(Path.Combine(directoryPath, "y_regression.zip"), out _);

            return new CombinedMuloModel(mlContext, transformationPipeline, classificationModel, xRegressionModel, yRegressionModel);
        }
    }

    // ====================== CLASSI DI DATI E PREDIZIONE ======================

    public class MuloImageData
    {
        public string ImagePath { get; set; }
        public string Label { get; set; }
        public float X { get; set; }
        public float Y { get; set; }
    }

    public class MuloClassificationPrediction
    {
        public string PredictedLabel { get; set; }
        public float[] Score { get; set; }
    }

    public class MuloXRegressionPrediction
    {
        public float X { get; set; }
    }

    public class MuloYRegressionPrediction
    {
        public float Y { get; set; }
    }

    public class MuloImagePrediction
    {
        public string PredictedLabel { get; set; }
        public float[] Score { get; set; }
        public float X { get; set; }
        public float Y { get; set; }
    }

    public class MuloDetectionResult
    {
        public bool IsMuloDetected { get; set; }
        public float X { get; set; }
        public float Y { get; set; }
        public float Confidence { get; set; }
        public string ErrorMessage { get; set; }
        public override string ToString()
        {
            if (!string.IsNullOrEmpty(ErrorMessage))
                return $"Errore: {ErrorMessage}";

            return IsMuloDetected
                ? $"Mulo rilevato con confidenza {Confidence:P2} alle coordinate ({X}, {Y})"
                : $"Nessun mulo rilevato (confidenza: {Confidence:P2})";
        }
    }

    // ====================== PROGRAMMA DI TEST (OPTIONALE) ======================

}