using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using Microsoft.ML.Vision;
using System.Drawing;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Onnx;
using Tensorflow;

namespace MachineLearning.NewServices
{
    /// <summary>
    /// Versione migliorata del rilevatore e localizzatore di oggetti nelle immagini.
    /// Implementa approcci più avanzati per la regressione delle coordinate.
    /// </summary>
    public class ObjectDetectorImproved
    {
        private readonly MLContext _mlContext;
        private readonly string _datasetPath;
        private readonly string _modelPath;
        private readonly int _imageWidth = 224;
        private readonly int _imageHeight = 224;
        private readonly string _objectType;

        // Parametri per la valutazione del modello
        private readonly int _epochCount = 100;
        private readonly int _batchSize = 10;
        private readonly float _learningRate = 0.001f;
        private readonly int _ensembleSize = 5; // Numero di modelli nell'ensemble

        public ObjectDetectorImproved(string datasetPath, string modelPath, string objectType = "Object")
        {
            _mlContext = new MLContext(seed: 42);
            _datasetPath = datasetPath;
            _modelPath = modelPath;
            _objectType = objectType;
        }

        /// <summary>
        /// Addestra e salva il modello di rilevamento oggetti migliorato.
        /// </summary>
        public void TrainAndSaveModel()
        {
            Console.WriteLine($"🔍 Caricamento immagini per rilevamento {_objectType}...");
            var imageData = LoadImagesWithAnnotations();

            // Verifica dataset
            VerifyDataset(imageData);

            // Normalizza e prepara il dataset con augmentation
            var augmentedData = ApplyDataAugmentation(imageData);
            Console.WriteLine($"Dataset aumentato: {augmentedData.Count} esempi (originale: {imageData.Count})");

            var dataView = _mlContext.Data.LoadFromEnumerable(augmentedData);
            var trainTestData = _mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

            Console.WriteLine("🏗️ Costruzione pipeline di addestramento migliorata...");

            // Prepara la pipeline di classificazione
            var classificationPipeline = BuildClassificationPipeline();

            // Prepara la pipeline unificata di localizzazione
            var localizationPipeline = BuildUnifiedLocalizationPipeline();

            Console.WriteLine("🏁 Inizio addestramento modelli...");

            // Addestramento modello di classificazione
            Console.WriteLine("   Addestramento classificatore...");
            var classificationModel = classificationPipeline.Fit(trainTestData.TrainSet);

            // Addestramento modello di localizzazione unificato
            Console.WriteLine("   Addestramento localizzatore unificato...");
            var localizationModel = localizationPipeline.Fit(trainTestData.TrainSet);

            // Addestramento modelli ensemble per X e Y
            Console.WriteLine("   Addestramento ensemble di modelli per localizzazione...");
            var ensembleXModels = TrainEnsemble(trainTestData.TrainSet, "X");
            var ensembleYModels = TrainEnsemble(trainTestData.TrainSet, "Y");

            // Valutazione modello di classificazione
            Console.WriteLine("🔎 Valutazione modello di classificazione...");
            var classificationMetrics = EvaluateClassificationModel(classificationModel, trainTestData.TestSet);

            // Valutazione modello di localizzazione unificato
            Console.WriteLine("🔎 Valutazione modello di localizzazione unificato...");
            var localizationMetrics = EvaluateUnifiedLocalizationModel(localizationModel, trainTestData.TestSet);

            // Valutazione modelli ensemble
            Console.WriteLine("🔎 Valutazione ensemble per coordinate X...");
            var ensembleXMetrics = EvaluateEnsembleModel(ensembleXModels, trainTestData.TestSet, "X");

            Console.WriteLine("🔎 Valutazione ensemble per coordinate Y...");
            var ensembleYMetrics = EvaluateEnsembleModel(ensembleYModels, trainTestData.TestSet, "Y");

            // Creazione del modello combinato migliorato
            var combinedModel = new ImprovedCombinedModel(
                _mlContext,
                classificationModel,
                localizationModel,
                ensembleXModels,
                ensembleYModels,
                _imageWidth,
                _imageHeight,
                _objectType);

            Console.WriteLine("💾 Salvataggio modello combinato migliorato...");
            combinedModel.Save(_modelPath);
            Console.WriteLine($"✅ Modello migliorato salvato con successo in: {_modelPath}");

            // Test del modello su alcuni esempi
            TestModelOnSamples(combinedModel, trainTestData.TestSet);
        }

        //// <summary>
        /// Addestra un ensemble di modelli per migliorare la regressione
        /// </summary>
        private List<ITransformer> TrainEnsemble(IDataView trainData, string coordinateType)
        {
            var models = new List<ITransformer>();

            // Base pipeline per l'estrazione di feature e preparazione dati
            var featureExtractionPipeline = _mlContext.Transforms.LoadImages(
                    outputColumnName: "Image",
                    imageFolder: _datasetPath,
                    inputColumnName: nameof(ObjectImageData.ImagePath))
                .Append(_mlContext.Transforms.ResizeImages(
                    outputColumnName: "ImageResized",
                    imageWidth: _imageWidth,
                    imageHeight: _imageHeight,
                    inputColumnName: "Image"))
                .Append(_mlContext.Transforms.ExtractPixels(
                    outputColumnName: "PixelValues",
                    inputColumnName: "ImageResized",
                    interleavePixelColors: true,
                    offsetImage: 117,
                    scaleImage: 1.0f / 255.0f))
                .Append(_mlContext.Transforms.Concatenate(
                    outputColumnName: "Features",
                    "PixelValues"))
                .Append(_mlContext.Transforms.NormalizeMinMax(
                    outputColumnName: "NormalizedFeatures",
                    inputColumnName: "Features"))
                // Copy the right coordinate to a standard column name
                .Append(_mlContext.Transforms.CopyColumns(
                    outputColumnName: coordinateType,  // "X" or "Y"
                    inputColumnName: coordinateType)); // "X" or "Y"

            // Algoritmi di regressione diversi per l'ensemble
            var regressors = new List<IEstimator<ITransformer>>
    {
        _mlContext.Regression.Trainers.FastTree(
            labelColumnName: coordinateType,
            featureColumnName: "NormalizedFeatures",
            numberOfLeaves: 20,
            numberOfTrees: 100,
            minimumExampleCountPerLeaf: 10),

        _mlContext.Regression.Trainers.FastForest(
            labelColumnName: coordinateType,
            featureColumnName: "NormalizedFeatures",
            numberOfTrees: 100,
            numberOfLeaves: 20),

        _mlContext.Regression.Trainers.LbfgsPoissonRegression(
            labelColumnName: coordinateType,
            featureColumnName: "NormalizedFeatures"),

        _mlContext.Regression.Trainers.Sdca(
            labelColumnName: coordinateType,
            featureColumnName: "NormalizedFeatures"),

        _mlContext.Regression.Trainers.LightGbm(
            labelColumnName: coordinateType,
            featureColumnName: "NormalizedFeatures",
            numberOfIterations: 100)
    };

            // Addestra ogni modello nell'ensemble
            for (int i = 0; i < Math.Min(_ensembleSize, regressors.Count); i++)
            {
                var pipeline = featureExtractionPipeline.Append(regressors[i])
                    .Append(_mlContext.Transforms.CopyColumns(
                        outputColumnName: $"Score{coordinateType}",
                        inputColumnName: "Score"));

                Console.WriteLine($"   Addestramento modello ensemble {i + 1}/{_ensembleSize} per {coordinateType}...");
                var model = pipeline.Fit(trainData);
                models.Add(model);
            }

            return models;
        }

        private RegressionMetrics EvaluateEnsembleModel(List<ITransformer> models, IDataView testData, string coordinateType)
        {
            if (models.Count == 0)
                return null;

            // Get the original data with labels first
            var originalData = _mlContext.Data.CreateEnumerable<ObjectImageData>(
                testData, reuseRowObject: false).ToList();

            // Filter to only get records with our target object type
            var targetRecords = originalData
                .Where(x => x.Label == _objectType)
                .ToList();

            if (targetRecords.Count == 0)
            {
                Console.WriteLine($"   No {_objectType} records found in test data");
                return null;
            }

            // Extract ground truth values
            var groundTruth = targetRecords
                .Select(x => coordinateType == "X" ? x.X : x.Y)
                .ToList();

            Console.WriteLine($"   Found {groundTruth.Count} ground truth values for {coordinateType}");

            // Create a list to store the predictions of each model
            var allPredictions = new List<List<float>>();

            // Get predictions from each model
            foreach (var model in models)
            {
                try
                {
                    var predictions = model.Transform(testData);

                    // Create a temporary class to capture the predictions and image path for joining
                    var predictionResults = _mlContext.Data.CreateEnumerable<ModelPredictionWithPath>(
                        predictions, reuseRowObject: false).ToList();

                    // Join predictions with original data to get only predictions for target objects
                    var targetPredictions = (from pred in predictionResults
                                             join orig in targetRecords
                                             on pred.ImagePath equals orig.ImagePath
                                             select pred.Score).ToList();

                    if (targetPredictions.Count > 0)
                    {
                        allPredictions.Add(targetPredictions);
                        Console.WriteLine($"   Model extracted {targetPredictions.Count} predictions for {coordinateType}");
                    }
                    else
                    {
                        Console.WriteLine($"   Warning: No predictions extracted for {coordinateType} from model");
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"   Error extracting predictions: {ex.Message}");
                }
            }

            if (allPredictions.Count == 0 || allPredictions[0].Count == 0)
            {
                Console.WriteLine($"   No valid predictions found for {coordinateType}");
                return null;
            }

            // Calculate ensemble predictions (averaging all models)
            var ensemblePredictions = new List<float>();
            int minCount = Math.Min(allPredictions.Min(p => p.Count), groundTruth.Count);

            for (int i = 0; i < minCount; i++)
            {
                float sum = 0;
                for (int j = 0; j < allPredictions.Count; j++)
                {
                    sum += allPredictions[j][i];
                }
                ensemblePredictions.Add(sum / allPredictions.Count);
            }

            // Create evaluation data
            var predictionData = new List<PredictionData>();
            for (int i = 0; i < Math.Min(ensemblePredictions.Count, groundTruth.Count); i++)
            {
                predictionData.Add(new PredictionData
                {
                    Label = groundTruth[i],
                    Score = ensemblePredictions[i]
                });
            }

            if (predictionData.Count == 0)
            {
                Console.WriteLine($"   No matching prediction and ground truth pairs for {coordinateType}");
                return null;
            }

            // Load data for evaluation
            var predictionDataView = _mlContext.Data.LoadFromEnumerable(predictionData);

            // Calculate metrics using ML.NET API
            var metrics = _mlContext.Regression.Evaluate(
                predictionDataView,
                labelColumnName: nameof(PredictionData.Label),
                scoreColumnName: nameof(PredictionData.Score));

            // Print results
            Console.WriteLine($"   {coordinateType} Ensemble - R² Score: {metrics.RSquared:F4}");
            Console.WriteLine($"   {coordinateType} Ensemble - MSE: {metrics.MeanSquaredError:F4}");
            Console.WriteLine($"   {coordinateType} Ensemble - RMSE: {metrics.RootMeanSquaredError:F4}");

            return metrics;
        }


        // Classe di supporto per memorizzare le coppie previsione-realtà
        private class PredictionData
        {
            public float Label { get; set; }
            public float Score { get; set; }
        }

        /// <summary>
        /// Costruisce la pipeline per il modello di classificazione
        /// </summary>
        private IEstimator<ITransformer> BuildClassificationPipeline()
        {
            Console.WriteLine("📊 Preparazione pipeline di classificazione...");

            // Pipeline di classificazione con TensorFlow tramite ImageClassification API
            return _mlContext.Transforms.Conversion.MapValueToKey(
                    outputColumnName: "LabelKey",
                    inputColumnName: nameof(ObjectImageData.Label))
                .Append(_mlContext.Transforms.LoadRawImageBytes(
                    outputColumnName: "ImageBytes",
                    imageFolder: _datasetPath,
                    inputColumnName: nameof(ObjectImageData.ImagePath)))
                .Append(_mlContext.MulticlassClassification.Trainers.ImageClassification(
                    new ImageClassificationTrainer.Options
                    {
                        FeatureColumnName = "ImageBytes",
                        LabelColumnName = "LabelKey",
                        Arch = ImageClassificationTrainer.Architecture.ResnetV2101,
                        Epoch = _epochCount,
                        BatchSize = _batchSize,
                        LearningRate = _learningRate,
                        MetricsCallback = (metrics) =>
                            Console.WriteLine(metrics.Train != null ?
                                $"Classificazione - Epoca: {metrics.Train.Epoch}, Accuratezza: {metrics.Train.Accuracy:F4}" :
                                "Elaboro...")
                    }))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue(
                    outputColumnName: "PredictedLabel",
                    inputColumnName: "PredictedLabel"));
        }

        /// <summary>
        /// Costruisce una pipeline unificata per la localizzazione
        /// </summary>
        private IEstimator<ITransformer> BuildUnifiedLocalizationPipeline()
        {
            Console.WriteLine("📊 Preparazione pipeline unificata di localizzazione...");

            // Pipeline di base per l'estrazione di feature
            var featureExtractionPipeline = _mlContext.Transforms.LoadImages(
                    outputColumnName: "Image",
                    imageFolder: _datasetPath,
                    inputColumnName: nameof(ObjectImageData.ImagePath))
                .Append(_mlContext.Transforms.ResizeImages(
                    outputColumnName: "ImageResized",
                    imageWidth: _imageWidth,
                    imageHeight: _imageHeight,
                    inputColumnName: "Image"))
                .Append(_mlContext.Transforms.ExtractPixels(
                    outputColumnName: "PixelValues",
                    inputColumnName: "ImageResized",
                    interleavePixelColors: true,
                    offsetImage: 117,
                    scaleImage: 1.0f / 255.0f))
                .Append(_mlContext.Transforms.Concatenate(
                    outputColumnName: "Features",
                    "PixelValues"))
                .Append(_mlContext.Transforms.NormalizeMinMax(
                    outputColumnName: "NormalizedFeatures",
                    inputColumnName: "Features"));

            // Invece di concatenare X e Y e usare un modello di regressione standard,
            // possiamo utilizzare ONNX o un approccio diverso che supporti output multidimensionali

            // Opzione 1: Utilizza due modelli separati, uno per X e uno per Y
            var xPipeline = featureExtractionPipeline
                .Append(_mlContext.Transforms.CopyColumns(
                    outputColumnName: "CoordinateX",
                    inputColumnName: nameof(ObjectImageData.X)))
                .Append(_mlContext.Regression.Trainers.LightGbm(
                    labelColumnName: "CoordinateX",
                    featureColumnName: "NormalizedFeatures",
                    numberOfIterations: 200))
                .Append(_mlContext.Transforms.CopyColumns(
                    outputColumnName: "PredictedX",
                    inputColumnName: "Score"));

            var yPipeline = featureExtractionPipeline
                .Append(_mlContext.Transforms.CopyColumns(
                    outputColumnName: "CoordinateY",
                    inputColumnName: nameof(ObjectImageData.Y)))
                .Append(_mlContext.Regression.Trainers.LightGbm(
                    labelColumnName: "CoordinateY",
                    featureColumnName: "NormalizedFeatures",
                    numberOfIterations: 200))
                .Append(_mlContext.Transforms.CopyColumns(
                    outputColumnName: "PredictedY",
                    inputColumnName: "Score"));

            // Combina i risultati
            var combinedPipeline = xPipeline
                .Append(yPipeline)
                .Append(_mlContext.Transforms.CustomMapping<CombinedInput, CombinedOutput>(
                    mapAction: (input, output) => {
                        output.UnifiedCoordinates = new float[2];
                        output.UnifiedCoordinates[0] = input.PredictedX;
                        output.UnifiedCoordinates[1] = input.PredictedY;
                    },
                    contractName: "CombineCoordinates"));

            return combinedPipeline;
        }

        // Classi di supporto per il mapping personalizzato
        private class CombinedInput
        {
            public float PredictedX { get; set; }
            public float PredictedY { get; set; }
        }

        private class CombinedOutput
        {
            [VectorType(2)]
            public float[] UnifiedCoordinates { get; set; }
        }

        /// <summary>
        /// Valuta il modello di localizzazione unificato
        /// </summary>
        private (double, double) EvaluateUnifiedLocalizationModel(ITransformer model, IDataView testData)
        {
            var predictions = model.Transform(testData);

            var predictionsList = _mlContext.Data.CreateEnumerable<UnifiedLocalizationPrediction>(
                predictions, reuseRowObject: false)
                .Where(x => x.Label == _objectType)
                .ToList();

            if (predictionsList.Count == 0)
                return (0, 0);

            // Calcola manualmente le metriche di regressione
            double mseX = 0, mseY = 0;
            foreach (var pred in predictionsList)
            {
                // Estrai X e Y dalla previsione
                var predX = pred.UnifiedCoordinates.Length > 0 ? pred.UnifiedCoordinates[0] : 0;
                var predY = pred.UnifiedCoordinates.Length > 1 ? pred.UnifiedCoordinates[1] : 0;

                mseX += Math.Pow(predX - pred.X, 2);
                mseY += Math.Pow(predY - pred.Y, 2);
            }

            mseX /= predictionsList.Count;
            mseY /= predictionsList.Count;

            double rmseX = Math.Sqrt(mseX);
            double rmseY = Math.Sqrt(mseY);

            Console.WriteLine($"   Modello unificato - MSE X: {mseX:F4}, RMSE X: {rmseX:F4}");
            Console.WriteLine($"   Modello unificato - MSE Y: {mseY:F4}, RMSE Y: {rmseY:F4}");

            return (rmseX, rmseY);
        }

        /// <summary>
        /// Valuta il modello di classificazione
        /// </summary>
        private MulticlassClassificationMetrics EvaluateClassificationModel(ITransformer model, IDataView testData)
        {
            var predictions = model.Transform(testData);
            var metrics = _mlContext.MulticlassClassification.Evaluate(
                predictions,
                labelColumnName: "LabelKey",
                predictedLabelColumnName: "PredictedLabel");

            Console.WriteLine($"   Accuratezza classificazione: {metrics.MicroAccuracy:P2}");
            Console.WriteLine($"   Log Loss: {metrics.LogLoss:F4}");
            Console.WriteLine($"   Matrice di confusione:");
            Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());

            return metrics;
        }

        /// <summary>
        /// Aggiunge augmentation ai dati per migliorare la generalizzazione
        /// </summary>
        private List<ObjectImageData> ApplyDataAugmentation(List<ObjectImageData> imageData)
        {
            Console.WriteLine("🔄 Applicazione data augmentation...");

            var augmentedData = new List<ObjectImageData>(imageData);
            var positiveExamples = imageData.Where(img => img.Label == _objectType).ToList();
            var random = new Random(42);

            // Solo per gli esempi positivi
            foreach (var example in positiveExamples)
            {
                // Aggiungi variazioni di posizione
                for (int i = 0; i < 3; i++)
                {
                    float xOffset = (float)(random.NextDouble() * 10 - 5); // -5 a +5
                    float yOffset = (float)(random.NextDouble() * 10 - 5); // -5 a +5

                    augmentedData.Add(new ObjectImageData
                    {
                        ImagePath = example.ImagePath,
                        Label = example.Label,
                        X = Math.Max(0, Math.Min(_imageWidth, example.X + xOffset)),
                        Y = Math.Max(0, Math.Min(_imageHeight, example.Y + yOffset))
                    });
                }
            }

            return augmentedData;
        }

        /// <summary>
        /// Verifica la qualità del dataset caricato
        /// </summary>
        private void VerifyDataset(List<ObjectImageData> imageData)
        {
            Console.WriteLine("VERIFICA DATASET:");
            var positiveImages = imageData.Where(i => i.Label == _objectType).ToList();
            var negativeImages = imageData.Where(i => i.Label != _objectType).ToList();

            Console.WriteLine($"Numero totale di immagini: {imageData.Count}");
            Console.WriteLine($"Numero di immagini con {_objectType}: {positiveImages.Count}");
            Console.WriteLine($"Numero di immagini senza {_objectType}: {negativeImages.Count}");

            if (positiveImages.Count > 0)
            {
                Console.WriteLine($"Numero di immagini con coordinate non-zero: {positiveImages.Count(i => i.X != 0 || i.Y != 0)}");
                Console.WriteLine($"Range X: [{positiveImages.Min(i => i.X)} - {positiveImages.Max(i => i.X)}]");
                Console.WriteLine($"Range Y: [{positiveImages.Min(i => i.Y)} - {positiveImages.Max(i => i.Y)}]");
                Console.WriteLine($"Media X: {positiveImages.Average(i => i.X):F2}");
                Console.WriteLine($"Media Y: {positiveImages.Average(i => i.Y):F2}");
                Console.WriteLine($"Deviazione standard X: {Math.Sqrt(positiveImages.Average(i => Math.Pow(i.X - positiveImages.Average(x => x.X), 2))):F2}");
                Console.WriteLine($"Deviazione standard Y: {Math.Sqrt(positiveImages.Average(i => Math.Pow(i.Y - positiveImages.Average(y => y.Y), 2))):F2}");
            }

            // Mostra alcuni esempi
            Console.WriteLine("ESEMPI DAL DATASET:");
            foreach (var img in positiveImages.Take(5))
            {
                Console.WriteLine($"Immagine positiva: {img.ImagePath}, X={img.X:F2}, Y={img.Y:F2}");
            }
            foreach (var img in negativeImages.Take(2))
            {
                Console.WriteLine($"Immagine negativa: {img.ImagePath}");
            }
        }

        /// <summary>
        /// Test del modello su alcuni esempi del dataset
        /// </summary>
        private void TestModelOnSamples(ImprovedCombinedModel model, IDataView testData)
        {
            Console.WriteLine("🔍 Test del modello migliorato su esempi di test:");

            // Estrai al massimo 10 esempi dal test set
            var testSamples = _mlContext.Data.CreateEnumerable<ObjectImageData>(testData, reuseRowObject: false)
                .Where(i => i.Label == _objectType)
                .Take(10)
                .ToList();

            if (testSamples.Count == 0)
            {
                Console.WriteLine("   Nessun esempio positivo trovato nel test set.");
                return;
            }

            var resultsList = new List<(string ImagePath, float TrueX, float TrueY, float PredX, float PredY, float Confidence)>();

            foreach (var sample in testSamples)
            {
                var result = model.Predict(sample);

                Console.WriteLine($"Immagine: {sample.ImagePath}");
                Console.WriteLine($"   Vero: {sample.Label}, Rilevato: {(result.IsObjectDetected ? _objectType : "No" + _objectType)} (Conf: {result.Confidence:P2})");

                if (result.IsObjectDetected)
                {
                    Console.WriteLine($"   Coordinate vere: ({sample.X:F2}, {sample.Y:F2})");
                    Console.WriteLine($"   Coordinate predette: ({result.X:F2}, {result.Y:F2})");
                    Console.WriteLine($"   Errore: ({Math.Abs(sample.X - result.X):F2}, {Math.Abs(sample.Y - result.Y):F2})");

                    resultsList.Add((sample.ImagePath, sample.X, sample.Y, result.X, result.Y, result.Confidence));
                }
            }

            if (resultsList.Count > 0)
            {
                // Calcola metriche aggregate
                float avgErrorX = resultsList.Average(r => Math.Abs(r.TrueX - r.PredX));
                float avgErrorY = resultsList.Average(r => Math.Abs(r.TrueY - r.PredY));
                float avgConfidence = resultsList.Average(r => r.Confidence);

                Console.WriteLine("\nMetriche aggregate:");
                Console.WriteLine($"   Errore medio X: {avgErrorX:F2}");
                Console.WriteLine($"   Errore medio Y: {avgErrorY:F2}");
                Console.WriteLine($"   Confidenza media: {avgConfidence:P2}");
            }
        }

        /// <summary>
        /// Rileva la presenza e le coordinate dell'oggetto in una immagine.
        /// </summary>
        public ObjectDetectionResult DetectObject(string imagePath)
        {
            ImprovedCombinedModel combinedModel;
            try
            {
                combinedModel = ImprovedCombinedModel.Load(_modelPath, _mlContext, _objectType);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Errore caricamento modello: {ex.Message}");
                return new ObjectDetectionResult
                {
                    IsObjectDetected = false,
                    X = 0,
                    Y = 0,
                    Confidence = 0,
                    ErrorMessage = $"Errore caricamento modello: {ex.Message}"
                };
            }

            var imageData = new ObjectImageData
            {
                ImagePath = imagePath,
                Label = string.Empty,
                X = 0,
                Y = 0
            };

            ObjectPrediction prediction;
            try
            {
                prediction = combinedModel.Predict(imageData);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Errore durante la predizione: {ex.Message}");
                return new ObjectDetectionResult
                {
                    IsObjectDetected = false,
                    X = 0,
                    Y = 0,
                    Confidence = 0,
                    ErrorMessage = $"Errore durante la predizione: {ex.Message}"
                };
            }

            float confidence = prediction.Score != null && prediction.Score.Length > 0
        ? prediction.Score.Max()
        : 0;

            bool isObjectDetected = prediction.PredictedLabel.Equals(_objectType, StringComparison.OrdinalIgnoreCase);

            Console.WriteLine($"Previsione: {(isObjectDetected ? _objectType : "Non " + _objectType)} con confidenza {confidence:P2}");
            if (isObjectDetected)
            {
                Console.WriteLine($"Coordinate predette: X={prediction.X:F2}, Y={prediction.Y:F2}");
            }

            return new ObjectDetectionResult
            {
                IsObjectDetected = isObjectDetected,
                X = prediction.X,
                Y = prediction.Y,
                Confidence = confidence,
                ObjectType = _objectType
            };
        }

        /// <summary>
        /// Carica le immagini e le annotazioni dal dataset.
        /// </summary>
        private List<ObjectImageData> LoadImagesWithAnnotations()
        {
            var images = new List<ObjectImageData>();
            var categories = new[] { _objectType, "Non" + _objectType };

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
                    if (category == _objectType && labelDict.TryGetValue(fileName, out var coords))
                    {
                        x = coords.X;
                        y = coords.Y;
                    }

                    images.Add(new ObjectImageData
                    {
                        ImagePath = Path.Combine(category, fileName),
                        Label = category,
                        X = x,
                        Y = y
                    });
                }
            }

            Console.WriteLine($"📊 Dataset caricato: {images.Count} immagini");
            Console.WriteLine($"   - {_objectType}: {images.Count(i => i.Label == _objectType)}");
            Console.WriteLine($"   - Non {_objectType}: {images.Count(i => i.Label != _objectType)}");

            // Stampa alcune statistiche sulle coordinate
            var objectImages = images.Where(i => i.Label == _objectType).ToList();
            if (objectImages.Count > 0)
            {
                Console.WriteLine($"   - Statistiche coordinate X: Min={objectImages.Min(i => i.X)}, Max={objectImages.Max(i => i.X)}, Media={objectImages.Average(i => i.X):F2}");
                Console.WriteLine($"   - Statistiche coordinate Y: Min={objectImages.Min(i => i.Y)}, Max={objectImages.Max(i => i.Y)}, Media={objectImages.Average(i => i.Y):F2}");
            }
            // Completion of LoadImagesWithAnnotations() method
            if (objectImages.Count > 0)
            {
                Console.WriteLine($"   - Statistiche coordinate X: Min={objectImages.Min(i => i.X)}, Max={objectImages.Max(i => i.X)}, Media={objectImages.Average(i => i.X):F2}");
                Console.WriteLine($"   - Statistiche coordinate Y: Min={objectImages.Min(i => i.Y)}, Max={objectImages.Max(i => i.Y)}, Media={objectImages.Average(i => i.Y):F2}");
            }

            // Verifica bilanciamento del dataset
            if (images.Count(i => i.Label == _objectType) < images.Count * 0.2)
            {
                Console.WriteLine("⚠️ ATTENZIONE: Dataset sbilanciato. Meno del 20% delle immagini contiene l'oggetto.");
                Console.WriteLine("   Considera di aumentare gli esempi positivi o utilizzare pesi di classe.");
            }

            return images;
        }

        /// <summary>
        /// Classe per rappresentare il modello combinato migliorato
        /// </summary>
        public class ImprovedCombinedModel
        {
            private readonly MLContext _mlContext;
            private readonly ITransformer _classificationModel;
            private readonly ITransformer _localizationModel;
            private readonly List<ITransformer> _ensembleXModels;
            private readonly List<ITransformer> _ensembleYModels;
            private readonly int _imageWidth;
            private readonly int _imageHeight;
            private readonly string _objectType;
            private readonly PredictionEngine<ObjectImageData, ObjectPrediction> _classificationPredictionEngine;
            private readonly PredictionEngine<ObjectImageData, UnifiedLocalizationPrediction> _localizationPredictionEngine;
            private readonly List<PredictionEngine<ObjectImageData, ObjectLocalizationEvalData>> _ensembleXPredictionEngines;
            private readonly List<PredictionEngine<ObjectImageData, ObjectLocalizationEvalData>> _ensembleYPredictionEngines;

            // Nuovo: rete neurale per la localizzazione
            private readonly ITransformer _neuralLocalizationModel;
            private readonly PredictionEngine<ObjectImageData, ObjectLocalizationEvalData> _neuralPredictionEngine;

            public ImprovedCombinedModel(
                MLContext mlContext,
                ITransformer classificationModel,
                ITransformer localizationModel,
                List<ITransformer> ensembleXModels,
                List<ITransformer> ensembleYModels,
                int imageWidth,
                int imageHeight,
                string objectType,
                ITransformer neuralLocalizationModel = null)
            {
                _mlContext = mlContext;
                _classificationModel = classificationModel;
                _localizationModel = localizationModel;
                _ensembleXModels = ensembleXModels;
                _ensembleYModels = ensembleYModels;
                _imageWidth = imageWidth;
                _imageHeight = imageHeight;
                _objectType = objectType;
                _neuralLocalizationModel = neuralLocalizationModel;

                // Crea prediction engines
                _classificationPredictionEngine = _mlContext.Model.CreatePredictionEngine<ObjectImageData, ObjectPrediction>(_classificationModel);
                _localizationPredictionEngine = _mlContext.Model.CreatePredictionEngine<ObjectImageData, UnifiedLocalizationPrediction>(_localizationModel);

                _ensembleXPredictionEngines = new List<PredictionEngine<ObjectImageData, ObjectLocalizationEvalData>>();
                foreach (var model in _ensembleXModels)
                {
                    _ensembleXPredictionEngines.Add(_mlContext.Model.CreatePredictionEngine<ObjectImageData, ObjectLocalizationEvalData>(model));
                }

                _ensembleYPredictionEngines = new List<PredictionEngine<ObjectImageData, ObjectLocalizationEvalData>>();
                foreach (var model in _ensembleYModels)
                {
                    _ensembleYPredictionEngines.Add(_mlContext.Model.CreatePredictionEngine<ObjectImageData, ObjectLocalizationEvalData>(model));
                }

                // Crea prediction engine per il modello neurale
                if (_neuralLocalizationModel != null)
                {
                    _neuralPredictionEngine = _mlContext.Model.CreatePredictionEngine<ObjectImageData, ObjectLocalizationEvalData>(_neuralLocalizationModel);
                }
            }

            /// <summary>
            /// Carica il modello combinato dal disco
            /// </summary>
            public static ImprovedCombinedModel Load(string modelPath, MLContext mlContext, string objectType)
            {
                // Carica i modelli dal percorso
                var modelDir = modelPath;
                if (!Directory.Exists(modelDir))
                {
                    Directory.CreateDirectory(modelDir);
                }

                var classificationModelPath = Path.Combine(modelDir, "classification.zip");
                var localizationModelPath = Path.Combine(modelDir, "localization.zip");
                var neuralLocalizationModelPath = Path.Combine(modelDir, "neural_localization.zip");
                var configPath = Path.Combine(modelDir, "config.json");

                // Verifica che i file esistano
                if (!File.Exists(classificationModelPath) || !File.Exists(localizationModelPath))
                {
                    throw new FileNotFoundException("Modelli non trovati. Esegui l'addestramento prima.");
                }

                // Carica configurazione
                var configJson = File.ReadAllText(configPath);
                var config = System.Text.Json.JsonSerializer.Deserialize<ModelConfig>(configJson);

                // Carica i modelli
                var classificationModel = mlContext.Model.Load(classificationModelPath, out _);
                var localizationModel = mlContext.Model.Load(localizationModelPath, out _);

                // Carica i modelli ensemble
                var ensembleXModels = new List<ITransformer>();
                var ensembleYModels = new List<ITransformer>();

                for (int i = 0; i < config.EnsembleSize; i++)
                {
                    var xModelPath = Path.Combine(modelDir, $"ensemble_x_{i}.zip");
                    var yModelPath = Path.Combine(modelDir, $"ensemble_y_{i}.zip");

                    if (File.Exists(xModelPath))
                    {
                        ensembleXModels.Add(mlContext.Model.Load(xModelPath, out _));
                    }

                    if (File.Exists(yModelPath))
                    {
                        ensembleYModels.Add(mlContext.Model.Load(yModelPath, out _));
                    }
                }

                // Carica il modello neurale se esiste
                ITransformer neuralLocalizationModel = null;
                if (File.Exists(neuralLocalizationModelPath))
                {
                    neuralLocalizationModel = mlContext.Model.Load(neuralLocalizationModelPath, out _);
                }

                return new ImprovedCombinedModel(
                    mlContext,
                    classificationModel,
                    localizationModel,
                    ensembleXModels,
                    ensembleYModels,
                    config.ImageWidth,
                    config.ImageHeight,
                    objectType,
                    neuralLocalizationModel);
            }

            /// <summary>
            /// Salva il modello combinato su disco
            /// </summary>
            public void Save(string modelPath)
            {
                var modelDir = modelPath;
                if (!Directory.Exists(modelDir))
                {
                    Directory.CreateDirectory(modelDir);
                }

                var classificationModelPath = Path.Combine(modelDir, "classification.zip");
                var localizationModelPath = Path.Combine(modelDir, "localization.zip");
                var neuralLocalizationModelPath = Path.Combine(modelDir, "neural_localization.zip");
                var configPath = Path.Combine(modelDir, "config.json");

                // Salva i modelli principali
                _mlContext.Model.Save(_classificationModel, null, classificationModelPath);
                _mlContext.Model.Save(_localizationModel, null, localizationModelPath);

                // Salva i modelli dell'ensemble
                for (int i = 0; i < _ensembleXModels.Count; i++)
                {
                    var xModelPath = Path.Combine(modelDir, $"ensemble_x_{i}.zip");
                    _mlContext.Model.Save(_ensembleXModels[i], null, xModelPath);
                }

                for (int i = 0; i < _ensembleYModels.Count; i++)
                {
                    var yModelPath = Path.Combine(modelDir, $"ensemble_y_{i}.zip");
                    _mlContext.Model.Save(_ensembleYModels[i], null, yModelPath);
                }

                // Salva il modello neurale se esiste
                if (_neuralLocalizationModel != null)
                {
                    _mlContext.Model.Save(_neuralLocalizationModel, null, neuralLocalizationModelPath);
                }

                // Salva configurazione
                var config = new ModelConfig
                {
                    ImageWidth = _imageWidth,
                    ImageHeight = _imageHeight,
                    ObjectType = _objectType,
                    EnsembleSize = Math.Max(_ensembleXModels.Count, _ensembleYModels.Count),
                    HasNeuralModel = _neuralLocalizationModel != null
                };

                var configJson = System.Text.Json.JsonSerializer.Serialize(config);
                File.WriteAllText(configPath, configJson);
            }

            /// <summary>
            /// Effettua la predizione combinando i risultati dei vari modelli
            /// </summary>
            public ObjectPrediction Predict(ObjectImageData data)
            {
                // Predizione di classificazione
                var classificationResult = _classificationPredictionEngine.Predict(data);

                // Predizione di localizzazione unificata
                var localizationResult = _localizationPredictionEngine.Predict(data);
                float predX = 0, predY = 0;

                // Estrai le coordinate dal risultato
                if (localizationResult.UnifiedCoordinates.Length >= 2)
                {
                    predX = localizationResult.UnifiedCoordinates[0];
                    predY = localizationResult.UnifiedCoordinates[1];
                }

                // Predizioni dell'ensemble per X e Y
                float ensembleX = 0, ensembleY = 0;

                if (_ensembleXPredictionEngines.Count > 0)
                {
                    float sumX = 0;
                    foreach (var engine in _ensembleXPredictionEngines)
                    {
                        var prediction = engine.Predict(data);
                        sumX += prediction.ScoreX;
                    }
                    ensembleX = sumX / _ensembleXPredictionEngines.Count;
                }

                if (_ensembleYPredictionEngines.Count > 0)
                {
                    float sumY = 0;
                    foreach (var engine in _ensembleYPredictionEngines)
                    {
                        var prediction = engine.Predict(data);
                        sumY += prediction.ScoreY;
                    }
                    ensembleY = sumY / _ensembleYPredictionEngines.Count;
                }

                // Predizione con modello neurale se disponibile
                float neuralX = 0, neuralY = 0;
                if (_neuralPredictionEngine != null)
                {
                    var neuralPrediction = _neuralPredictionEngine.Predict(data);
                    neuralX = neuralPrediction.ScoreX;
                    neuralY = neuralPrediction.ScoreY;
                }

                // Combina le predizioni con pesi
                float finalX, finalY;

                if (_neuralPredictionEngine != null)
                {
                    // Se abbiamo il modello neurale, diamo più peso a quello
                    finalX = (0.2f * predX) + (0.3f * ensembleX) + (0.5f * neuralX);
                    finalY = (0.2f * predY) + (0.3f * ensembleY) + (0.5f * neuralY);
                }
                else
                {
                    // Altrimenti, combiniamo uniforme e ensemble
                    finalX = (0.4f * predX) + (0.6f * ensembleX);
                    finalY = (0.4f * predY) + (0.6f * ensembleY);
                }

                // Limita le coordinate all'interno dell'immagine
                finalX = Math.Max(0, Math.Min(_imageWidth, finalX));
                finalY = Math.Max(0, Math.Min(_imageHeight, finalY));

                // Risultato finale
                var result = new ObjectPrediction
                {
                    PredictedLabel = classificationResult.PredictedLabel,
                    Score = classificationResult.Score,
                    X = finalX,
                    Y = finalY
                };

                return result;
            }
        }

        /// <summary>
        /// Classe per la configurazione del modello
        /// </summary>
        public class ModelConfig
        {
            public int ImageWidth { get; set; }
            public int ImageHeight { get; set; }
            public string ObjectType { get; set; }
            public int EnsembleSize { get; set; }
            public bool HasNeuralModel { get; set; }
        }
    }

    /// <summary>
    /// Classe per le predizioni di localizzazione unificate
    /// </summary>
    public class UnifiedLocalizationPrediction : ObjectImageData
    {
        [VectorType(2)]
        public float[] UnifiedCoordinates { get; set; }
    }

    /// <summary>
    /// Classe per l'input del modello
    /// </summary>
    public class ObjectImageData
    {
        [LoadColumn(0)]
        public string ImagePath { get; set; }

        [LoadColumn(1)]
        public string Label { get; set; }

        [LoadColumn(2)]
        public float X { get; set; }

        [LoadColumn(3)]
        public float Y { get; set; }
    }

    /// <summary>
    /// Classe per la valutazione della localizzazione
    /// </summary>
    public class ObjectLocalizationEvalData : ObjectImageData
    {
        public float ScoreX { get; set; }
        public float ScoreY { get; set; }
    }

    /// <summary>
    /// Classe per il risultato della predizione
    /// </summary>
    public class ObjectPrediction
    {
        public string PredictedLabel { get; set; }

        [VectorType(2)]
        public float[] Score { get; set; }

        public float X { get; set; }

        public float Y { get; set; }

        // Aggiungi questa proprietà calcolata
        public bool IsObjectDetected => !string.IsNullOrEmpty(PredictedLabel) &&
                                      !PredictedLabel.StartsWith("Non");

        // Aggiungi questa proprietà calcolata
        public float Confidence => Score != null && Score.Length > 0 ? Score.Max() : 0;
    }

    /// <summary>
    /// Classe per il risultato del rilevamento
    /// </summary>
    public class ObjectDetectionResult
    {
        public bool IsObjectDetected { get; set; }
        public float X { get; set; }
        public float Y { get; set; }
        public float Confidence { get; set; }
        public string ObjectType { get; set; }
        public string ErrorMessage { get; set; }
    }

    // Define a class to properly capture the model output with image path for joining
    public class ModelPredictionWithPath
    {
        public string ImagePath { get; set; }
        public float Score { get; set; }
    }
}