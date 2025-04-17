//buono ma lavora male sulla regressione.


using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Vision;
using Microsoft.ML.Transforms;

namespace MachineLearning.Services
{
    /// <summary>
    /// Classe per il rilevamento e la localizzazione dei muli nelle immagini di Ultima Online.
    /// Implementa sia classificazione che regressione delle coordinate tramite ML.NET.
    /// </summary>
    public class MuloDetector
    {
        private readonly MLContext _mlContext;
        private readonly string _datasetPath;
        private readonly string _modelPath;
        private readonly int _imageWidth = 224;
        private readonly int _imageHeight = 224;

        // Fattori di normalizzazione per le coordinate
        private float _maxX = 1.0f;
        private float _maxY = 1.0f;

        public MuloDetector(string datasetPath, string modelPath)
        {
            _mlContext = new MLContext(seed: 42);
            _datasetPath = datasetPath;
            _modelPath = modelPath;
        }

        /// <summary>
        /// Addestra e salva il modello combinato che gestisce sia classificazione che localizzazione.
        /// </summary>
        public void TrainAndSaveModel()
        {
            Console.WriteLine("🔍 Caricamento immagini dal dataset...");
            var imageData = LoadImagesWithAnnotations();

            // Calcola i fattori di normalizzazione per le coordinate
            CalculateNormalizationFactors(imageData);

            // Normalizza le coordinate nel dataset
            var normalizedData = NormalizeCoordinates(imageData);

            var dataView = _mlContext.Data.LoadFromEnumerable(normalizedData);
            var trainTestData = _mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

            Console.WriteLine("🏗️ Costruzione pipeline di addestramento...");

            // ----------------- CLASSIFICAZIONE -----------------
            Console.WriteLine("📊 Preparazione pipeline di classificazione...");
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
                                                  MetricsCallback = (metrics) => Console.WriteLine(metrics.Train != null ? $"Classificazione - Epoca: {metrics.Train.Epoch}, Accuratezza: {metrics.Train.Accuracy:F4}" : "Elaboro...")
                                              }))
                                          .Append(_mlContext.Transforms.Conversion.MapKeyToValue(
                                              outputColumnName: "PredictedLabel",
                                              inputColumnName: "PredictedLabel"));

            // ----------------- REGRESSIONE DELLE COORDINATE -----------------
            // Utilizziamo una pipeline separata ma avanzata per estrarre features per la regressione

            Console.WriteLine("📊 Preparazione pipeline per estrazione features di regressione...");

            // Pipeline di base per il preprocessing dell'immagine
            var imagePreprocessingPipeline = _mlContext.Transforms.LoadImages(
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
                    offsetImage: 117)); // Valore medio di ImageNet per normalizzazione

            // Utilizziamo la trasformazione di convoluzione per ottenere features più ricche
            var featureExtractionPipeline = imagePreprocessingPipeline
                .Append(_mlContext.Transforms.ConvertToImage(
    outputColumnName: "ImagePreprocessed",
    imageWidth: _imageWidth,
    imageHeight: _imageHeight,
    inputColumnName: "ImagePixels"))
                .Append(_mlContext.Transforms.ResizeImages("ImageResized", _imageWidth, _imageHeight, "ImagePreprocessed"))
                .Append(_mlContext.Transforms.ExtractPixels("Features", "ImageResized",
                    interleavePixelColors: true,
                    offsetImage: 117,
                    scaleImage: 1f / 255f))
                .Append(_mlContext.Transforms.NormalizeMinMax("NormalizedFeatures", "Features"));

            Console.WriteLine("📊 Preparazione pipeline di regressione X...");
            var xRegressionPipeline = featureExtractionPipeline
                .Append(_mlContext.Transforms.CopyColumns(
                    outputColumnName: "XFeatures",
                    inputColumnName: "NormalizedFeatures"))
                .Append(_mlContext.Regression.Trainers.FastTree(
                    labelColumnName: nameof(MuloImageData.X),
                    featureColumnName: "XFeatures",
                    numberOfLeaves: 20,
                    numberOfTrees: 100,
                    minimumExampleCountPerLeaf: 10));

            Console.WriteLine("📊 Preparazione pipeline di regressione Y...");
            var yRegressionPipeline = featureExtractionPipeline
                .Append(_mlContext.Transforms.CopyColumns(
                    outputColumnName: "YFeatures",
                    inputColumnName: "NormalizedFeatures"))
                .Append(_mlContext.Regression.Trainers.FastTree(
                    labelColumnName: nameof(MuloImageData.Y),
                    featureColumnName: "YFeatures",
                    numberOfLeaves: 20,
                    numberOfTrees: 100,
                    minimumExampleCountPerLeaf: 10));

            Console.WriteLine("🏁 Inizio addestramento modelli...");

            Console.WriteLine("   Addestramento classificatore (Mulo/NonMulo)...");
            var classificationModel = classificationPipeline.Fit(trainTestData.TrainSet);

            Console.WriteLine("   Addestramento modello di feature extraction...");
            var featureModel = featureExtractionPipeline.Fit(trainTestData.TrainSet);

            Console.WriteLine("   Addestramento regressore per X...");
            var xRegressionModel = xRegressionPipeline.Fit(trainTestData.TrainSet);

            Console.WriteLine("   Addestramento regressore per Y...");
            var yRegressionModel = yRegressionPipeline.Fit(trainTestData.TrainSet);

            // Creazione del modello combinato
            var combinedModel = new CombinedMuloModel(
                _mlContext,
                featureModel,
                classificationModel,
                xRegressionModel,
                yRegressionModel,
                _maxX,
                _maxY);

            Console.WriteLine("💾 Salvataggio modello combinato...");
            combinedModel.Save(_modelPath);
            Console.WriteLine($"✅ Modello salvato con successo in: {_modelPath}");

            // Valutazione del modello di classificazione
            Console.WriteLine("🔎 Valutazione modello di classificazione...");
            var testPredictions = classificationModel.Transform(trainTestData.TestSet);
            var classMetrics = _mlContext.MulticlassClassification.Evaluate(
                testPredictions, labelColumnName: "LabelKey", predictedLabelColumnName: "PredictedLabel");

            Console.WriteLine($"   Accuratezza classificazione: {classMetrics.MicroAccuracy:P2}");
            Console.WriteLine($"   Log Loss: {classMetrics.LogLoss:F4}");

            // Valutazione dei modelli di regressione
            Console.WriteLine("🔎 Valutazione modello di regressione X...");
            var xTransformed = xRegressionModel.Transform(trainTestData.TestSet);
            var xMetrics = _mlContext.Regression.Evaluate(xTransformed,
                labelColumnName: nameof(MuloImageData.X),
                scoreColumnName: "Score");

            Console.WriteLine($"   X - R² Score: {xMetrics.RSquared:F4}");
            Console.WriteLine($"   X - MSE: {xMetrics.MeanSquaredError:F4}");
            Console.WriteLine($"   X - RMSE: {xMetrics.RootMeanSquaredError:F4}");

            Console.WriteLine("🔎 Valutazione modello di regressione Y...");
            var yTransformed = yRegressionModel.Transform(trainTestData.TestSet);
            var yMetrics = _mlContext.Regression.Evaluate(yTransformed,
                labelColumnName: nameof(MuloImageData.Y),
                scoreColumnName: "Score");

            Console.WriteLine($"   Y - R² Score: {yMetrics.RSquared:F4}");
            Console.WriteLine($"   Y - MSE: {yMetrics.MeanSquaredError:F4}");
            Console.WriteLine($"   Y - RMSE: {yMetrics.RootMeanSquaredError:F4}");

            Console.WriteLine($"✅ Addestramento completato con successo!");

            // Debug: Verifica la distribuzione delle previsioni
            Console.WriteLine("🔍 Analisi distribuzione delle previsioni X e Y...");
            var muloTestData = new List<MuloImageData>();
            foreach (var item in trainTestData.TestSet.Preview().RowView)
            {
                var label = item.Values.FirstOrDefault(v => v.Key == nameof(MuloImageData.Label)).Value;
                if (label?.ToString() == "Mulo")
                {
                    var path = item.Values.FirstOrDefault(v => v.Key == nameof(MuloImageData.ImagePath)).Value?.ToString();
                    if (path != null)
                    {
                        muloTestData.Add(new MuloImageData
                        {
                            ImagePath = path,
                            Label = "Mulo"
                        });
                    }
                }
            }

            if (muloTestData.Count > 0)
            {
                Console.WriteLine($"Testando previsioni su {muloTestData.Count} immagini di muli...");
                var resultsList = new List<(float PredX, float PredY)>();

                foreach (var img in muloTestData.Take(5)) // Limita per brevità
                {
                    var prediction = combinedModel.Predict(img);
                    resultsList.Add((prediction.X, prediction.Y));
                    Console.WriteLine($"Immagine: {img.ImagePath}, Coord. predette: X={prediction.X:F2}, Y={prediction.Y:F2}");
                }

                if (resultsList.Count > 0)
                {
                    Console.WriteLine($"Media X predetta: {resultsList.Average(r => r.PredX):F2}");
                    Console.WriteLine($"Media Y predetta: {resultsList.Average(r => r.PredY):F2}");
                    Console.WriteLine($"Dev. Std X: {Math.Sqrt(resultsList.Average(r => Math.Pow(r.PredX - resultsList.Average(s => s.PredX), 2))):F2}");
                    Console.WriteLine($"Dev. Std Y: {Math.Sqrt(resultsList.Average(r => Math.Pow(r.PredY - resultsList.Average(s => s.PredY), 2))):F2}");
                }
            }
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

            Console.WriteLine($"Previsione: {(isMulo ? "Mulo" : "Non Mulo")} con confidenza {confidence:P2}");
            if (isMulo)
            {
                Console.WriteLine($"Coordinate predette: X={prediction.X:F2}, Y={prediction.Y:F2}");
            }

            return new MuloDetectionResult
            {
                IsMuloDetected = isMulo,
                X = prediction.X,
                Y = prediction.Y,
                Confidence = confidence
            };
        }

        /// <summary>
        /// Calcola i fattori di normalizzazione per le coordinate X e Y nel dataset
        /// </summary>
        private void CalculateNormalizationFactors(List<MuloImageData> imageData)
        {
            if (imageData.Count == 0) return;

            var muloImages = imageData.Where(i => i.Label == "Mulo").ToList();
            if (muloImages.Count == 0) return;

            _maxX = muloImages.Max(i => Math.Abs(i.X));
            _maxY = muloImages.Max(i => Math.Abs(i.Y));

            // Evita divisione per zero
            _maxX = _maxX > 0 ? _maxX : 1.0f;
            _maxY = _maxY > 0 ? _maxY : 1.0f;

            Console.WriteLine($"Fattori di normalizzazione calcolati: MaxX={_maxX}, MaxY={_maxY}");
        }

        /// <summary>
        /// Normalizza le coordinate X e Y nel dataset tra 0 e 1
        /// </summary>
        private List<MuloImageData> NormalizeCoordinates(List<MuloImageData> imageData)
        {
            var normalizedData = new List<MuloImageData>();
            foreach (var item in imageData)
            {
                normalizedData.Add(new MuloImageData
                {
                    ImagePath = item.ImagePath,
                    Label = item.Label,
                    X = item.X / _maxX,  // Normalizza X
                    Y = item.Y / _maxY   // Normalizza Y
                });
            }

            Console.WriteLine($"Coordinate normalizzate nel dataset");
            return normalizedData;
        }

        /// <summary>
        /// Carica le immagini e le annotazioni dal dataset.
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
                var lines = File.ReadAllLines(csvPath).Skip(1); // Salta l'header
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
                Console.WriteLine($"   Caricate {labelDict.Count} annotazioni di coordinate");
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
                        Console.WriteLine($"Immagine {fileName}: Coordinate X={x}, Y={y}");
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

            // Stampa alcune statistiche sulle coordinate
            var muloImages = images.Where(i => i.Label == "Mulo").ToList();
            if (muloImages.Count > 0)
            {
                Console.WriteLine($"   - Statistiche coordinate X: Min={muloImages.Min(i => i.X)}, Max={muloImages.Max(i => i.X)}, Media={muloImages.Average(i => i.X):F2}");
                Console.WriteLine($"   - Statistiche coordinate Y: Min={muloImages.Min(i => i.Y)}, Max={muloImages.Max(i => i.Y)}, Media={muloImages.Average(i => i.Y):F2}");
            }

            return images;
        }
    }

    /// <summary>
    /// Classe che combina i modelli di classificazione e regressione per le coordinate.
    /// </summary>
    public class CombinedMuloModel
    {
        private readonly MLContext _mlContext;
        private readonly ITransformer _featureModel;
        private readonly ITransformer _classificationModel;
        private readonly ITransformer _xRegressionModel;
        private readonly ITransformer _yRegressionModel;
        private readonly float _maxX;
        private readonly float _maxY;

        private readonly PredictionEngine<MuloImageData, MuloClassificationPrediction> _classificationEngine;
        private readonly PredictionEngine<MuloImageData, MuloXRegressionPrediction> _xRegressionEngine;
        private readonly PredictionEngine<MuloImageData, MuloYRegressionPrediction> _yRegressionEngine;

        public CombinedMuloModel(
            MLContext mlContext,
            ITransformer featureModel,
            ITransformer classificationModel,
            ITransformer xRegressionModel,
            ITransformer yRegressionModel,
            float maxX,
            float maxY)
        {
            _mlContext = mlContext;
            _featureModel = featureModel;
            _classificationModel = classificationModel;
            _xRegressionModel = xRegressionModel;
            _yRegressionModel = yRegressionModel;
            _maxX = maxX;
            _maxY = maxY;

            _classificationEngine = _mlContext.Model.CreatePredictionEngine<MuloImageData, MuloClassificationPrediction>(_classificationModel);
            _xRegressionEngine = _mlContext.Model.CreatePredictionEngine<MuloImageData, MuloXRegressionPrediction>(_xRegressionModel);
            _yRegressionEngine = _mlContext.Model.CreatePredictionEngine<MuloImageData, MuloYRegressionPrediction>(_yRegressionModel);
        }

        public MuloImagePrediction Predict(MuloImageData input)
        {
            // Prima esegui la classificazione
            var classificationPrediction = _classificationEngine.Predict(input);

            // Poi esegui la regressione solo se necessario
            float normalizedX = 0;
            float normalizedY = 0;

            if (classificationPrediction.PredictedLabel.Equals("Mulo", StringComparison.OrdinalIgnoreCase))
            {
                var xPrediction = _xRegressionEngine.Predict(input);
                var yPrediction = _yRegressionEngine.Predict(input);

                normalizedX = xPrediction.X;
                normalizedY = yPrediction.Y;

                Console.WriteLine($"Predizione raw normalizzata: X={normalizedX:F4}, Y={normalizedY:F4}");
            }

            // Denormalizza le coordinate
            float denormalizedX = normalizedX * _maxX;
            float denormalizedY = normalizedY * _maxY;

            return new MuloImagePrediction
            {
                PredictedLabel = classificationPrediction.PredictedLabel,
                Score = classificationPrediction.Score,
                X = denormalizedX,
                Y = denormalizedY
            };
        }

        public void Save(string directoryPath)
        {
            Directory.CreateDirectory(directoryPath);

            // Salva tutti i modelli
            _mlContext.Model.Save(_featureModel, null, Path.Combine(directoryPath, "features.zip"));
            _mlContext.Model.Save(_classificationModel, null, Path.Combine(directoryPath, "classification.zip"));
            _mlContext.Model.Save(_xRegressionModel, null, Path.Combine(directoryPath, "x_regression.zip"));
            _mlContext.Model.Save(_yRegressionModel, null, Path.Combine(directoryPath, "y_regression.zip"));

            // Salva i fattori di normalizzazione
            using (var writer = new StreamWriter(Path.Combine(directoryPath, "normalization.csv")))
            {
                writer.WriteLine($"{_maxX},{_maxY}");
            }
        }

        public static CombinedMuloModel Load(string directoryPath, MLContext mlContext)
        {
            var featureModel = mlContext.Model.Load(Path.Combine(directoryPath, "features.zip"), out _);
            var classificationModel = mlContext.Model.Load(Path.Combine(directoryPath, "classification.zip"), out _);
            var xRegressionModel = mlContext.Model.Load(Path.Combine(directoryPath, "x_regression.zip"), out _);
            var yRegressionModel = mlContext.Model.Load(Path.Combine(directoryPath, "y_regression.zip"), out _);

            // Carica i fattori di normalizzazione
            float maxX = 1.0f;
            float maxY = 1.0f;

            var normPath = Path.Combine(directoryPath, "normalization.csv");
            if (File.Exists(normPath))
            {
                var line = File.ReadAllLines(normPath).FirstOrDefault();
                if (line != null)
                {
                    var parts = line.Split(',');
                    if (parts.Length >= 2)
                    {
                        float.TryParse(parts[0], out maxX);
                        float.TryParse(parts[1], out maxY);
                    }
                }
            }

            Console.WriteLine($"Modello caricato con fattori di normalizzazione: MaxX={maxX}, MaxY={maxY}");

            return new CombinedMuloModel(
                mlContext,
                featureModel,
                classificationModel,
                xRegressionModel,
                yRegressionModel,
                maxX,
                maxY);
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
                ? $"Mulo rilevato con confidenza {Confidence:P2} alle coordinate ({X:F1}, {Y:F1})"
                : $"Nessun mulo rilevato (confidenza: {Confidence:P2})";
        }
    }
}