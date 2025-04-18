using DetectorModel.Models;
using DetectorModel.Services;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Vision;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
namespace MachineLearning.Services
{
    /// <summary>
    /// Versione semplificata e ottimizzata del rilevatore e localizzatore di oggetti nelle immagini.
    /// Implementa un approccio bilanciato tra precisione e velocità di addestramento.
    /// </summary>
    public class ObjectDetectorTrainer
    {
        private readonly MLContext _mlContext;
        private readonly string _datasetPath;
        private readonly string _modelPath;
        private readonly int _imageWidth = 224;
        private readonly int _imageHeight = 224;
        private readonly string _objectType;

        // Parametri per l'addestramento
        private readonly int _epochCount = 50;
        private readonly int _batchSize = 10;
        private readonly float _learningRate = 0.001f;

        public ObjectDetectorTrainer(string datasetPath, string modelPath, string objectType = "Object")
        {
            _mlContext = new MLContext(seed: 42);
            _datasetPath = datasetPath;
            _modelPath = modelPath;
            _objectType = objectType;
        }

        /// <summary>
        /// Addestra e salva il modello di rilevamento oggetti.
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

            Console.WriteLine("🏗️ Costruzione pipeline di addestramento...");

            // Prepara i modelli di base
            var classificationPipeline = BuildClassificationPipeline();
            var xLocalizationPipeline = BuildXLocalizationPipeline();
            var yLocalizationPipeline = BuildYLocalizationPipeline();

            Console.WriteLine("🏁 Inizio addestramento modelli...");

            // Addestramento modello di classificazione
            Console.WriteLine("   Addestramento classificatore...");
            var classificationModel = classificationPipeline.Fit(trainTestData.TrainSet);

            // Addestramento modelli di localizzazione
            Console.WriteLine("   Addestramento localizzatore X...");
            var xLocalizationModel = xLocalizationPipeline.Fit(trainTestData.TrainSet);

            Console.WriteLine("   Addestramento localizzatore Y...");
            var yLocalizationModel = yLocalizationPipeline.Fit(trainTestData.TrainSet);

            // Valutazione modello di classificazione
            Console.WriteLine("🔎 Valutazione modello di classificazione...");
            var classificationMetrics = EvaluateClassificationModel(classificationModel, trainTestData.TestSet);

            // Valutazione modelli di localizzazione
            Console.WriteLine("🔎 Valutazione modelli di localizzazione...");
            var localizationMetrics = EvaluateLocalizationModels(xLocalizationModel, yLocalizationModel, trainTestData.TestSet);

            // Creazione del modello combinato
            var combinedModel = new CombinedModel(
                _mlContext,
                classificationModel,
                xLocalizationModel,
                yLocalizationModel,
                _imageWidth,
                _imageHeight,
                _objectType);

            Console.WriteLine("💾 Salvataggio modello combinato...");
            combinedModel.Save(_modelPath);
            Console.WriteLine($"✅ Modello salvato con successo in: {_modelPath}");

            // Test del modello su alcuni esempi
            TestModelOnSamples(combinedModel, trainTestData.TestSet);
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
        /// Costruisce la pipeline per la regressione della coordinata X
        /// </summary>
        private IEstimator<ITransformer> BuildXLocalizationPipeline()
        {
            Console.WriteLine("📊 Preparazione pipeline di localizzazione X...");

            // Pipeline per la regressione di X
            return _mlContext.Transforms.LoadImages(
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
                .Append(_mlContext.Regression.Trainers.FastTree(
                    labelColumnName: nameof(ObjectImageData.X),
                    featureColumnName: "NormalizedFeatures",
                    numberOfTrees: 100,
                    numberOfLeaves: 20));
        }

        /// <summary>
        /// Costruisce la pipeline per la regressione della coordinata Y
        /// </summary>
        private IEstimator<ITransformer> BuildYLocalizationPipeline()
        {
            Console.WriteLine("📊 Preparazione pipeline di localizzazione Y...");

            // Pipeline per la regressione di Y
            return _mlContext.Transforms.LoadImages(
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
                .Append(_mlContext.Regression.Trainers.FastTree(
                    labelColumnName: nameof(ObjectImageData.Y),
                    featureColumnName: "NormalizedFeatures",
                    numberOfTrees: 100,
                    numberOfLeaves: 20));
        }

        /// <summary>
        /// Valuta i modelli di localizzazione per X e Y
        /// </summary>
        private (double, double) EvaluateLocalizationModels(ITransformer xModel, ITransformer yModel, IDataView testData)
        {
            var xPredictions = xModel.Transform(testData);
            var yPredictions = yModel.Transform(testData);

            // Filtriamo solo gli esempi positivi (dove Label == _objectType)
            // Nota: Non usiamo FilterRowsByColumn perché lavora solo su colonne numeriche
            var positiveExamples = _mlContext.Data.CreateEnumerable<ObjectImageData>(
                testData, reuseRowObject: false)
                .Where(x => x.Label == _objectType)
                .ToList();

            // Se non ci sono esempi positivi, restituiamo zero
            if (positiveExamples.Count == 0)
                return (0, 0);

            // Carichiamo i dati filtrati come nuovo IDataView
            var filteredTestData = _mlContext.Data.LoadFromEnumerable(positiveExamples);
            var filteredXPredictions = xModel.Transform(filteredTestData);
            var filteredYPredictions = yModel.Transform(filteredTestData);

            // Calcola le metriche di regressione
            var xMetrics = _mlContext.Regression.Evaluate(filteredXPredictions, labelColumnName: nameof(ObjectImageData.X));
            var yMetrics = _mlContext.Regression.Evaluate(filteredYPredictions, labelColumnName: nameof(ObjectImageData.Y));

            Console.WriteLine($"   Modello X - RMSE: {xMetrics.RootMeanSquaredError:F4}, R²: {xMetrics.RSquared:F4}");
            Console.WriteLine($"   Modello Y - RMSE: {yMetrics.RootMeanSquaredError:F4}, R²: {yMetrics.RSquared:F4}");

            return (xMetrics.RootMeanSquaredError, yMetrics.RootMeanSquaredError);
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
        private void TestModelOnSamples(CombinedModel model, IDataView testData)
        {
            Console.WriteLine("🔍 Test del modello su esempi di test:");

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
                Console.WriteLine($"   Vero: {sample.Label}, Rilevato: {(result.IsObjectDetected ? _objectType : "Non" + _objectType)} (Conf: {result.Confidence:P2})");

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

                // Calcola la percentuale di esempi con errore sotto i 5 pixel
                float percentUnder5px = resultsList.Count(r =>
                    Math.Abs(r.TrueX - r.PredX) < 5 &&
                    Math.Abs(r.TrueY - r.PredY) < 5) * 100.0f / resultsList.Count;

                Console.WriteLine($"   Percentuale predizioni con errore < 5 pixel: {percentUnder5px:F1}%");
            }
        }

        /// <summary>
        /// Rileva la presenza e le coordinate dell'oggetto in una immagine.
        /// </summary>
        public ObjectDetectionResult DetectObject(string imagePath)
        {
            CombinedModel combinedModel;
            try
            {
                combinedModel = CombinedModel.Load(_modelPath, _mlContext, _objectType);
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

            float confidence = prediction.Confidence;

            bool isObjectDetected = prediction.IsObjectDetected;

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
                        var fileName = Path.GetFileName(parts[0]);

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

            // Verifica bilanciamento del dataset
            if (images.Count(i => i.Label == _objectType) < images.Count * 0.2)
            {
                Console.WriteLine("⚠️ ATTENZIONE: Dataset sbilanciato. Meno del 20% delle immagini contiene l'oggetto.");
                Console.WriteLine("   Considera di aumentare gli esempi positivi o utilizzare pesi di classe.");
            }

            return images;
        }
    }
}