using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using Microsoft.ML.Vision;
using Microsoft.ML.Transforms;
using System.Drawing;
using System.Threading.Tasks;
using Microsoft.ML.Transforms.Onnx;
using static MachineLearning.Services.ObjectLocalizationPrediction;

namespace MachineLearning.Services
{
    /// <summary>
    /// Classe per il rilevamento e la localizzazione di oggetti nelle immagini.
    /// Implementa sia classificazione che localizzazione usando ML.NET e TensorFlow.
    /// </summary>
    public class ObjectDetector
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

        public ObjectDetector(string datasetPath, string modelPath, string objectType = "Object")
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

            // Normalizza e prepara il dataset
            var normalizedData = PreprocessData(imageData);

            var dataView = _mlContext.Data.LoadFromEnumerable(normalizedData);
            var trainTestData = _mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

            Console.WriteLine("🏗️ Costruzione pipeline di addestramento...");

            // Prepara la pipeline di classificazione
            var classificationPipeline = BuildClassificationPipeline();

            // Prepara le pipeline di localizzazione per X e Y
            var localizationXPipeline = BuildLocalizationPipeline();
            var localizationYPipeline = BuildLocalizationYPipeline();

            Console.WriteLine("🏁 Inizio addestramento modelli...");

            // Addestramento modello di classificazione
            Console.WriteLine("   Addestramento classificatore...");
            var classificationModel = classificationPipeline.Fit(trainTestData.TrainSet);

            // Addestramento modelli di localizzazione X e Y
            Console.WriteLine("   Addestramento localizzatore X...");
            var localizationXModel = localizationXPipeline.Fit(trainTestData.TrainSet);

            Console.WriteLine("   Addestramento localizzatore Y...");
            var localizationYModel = localizationYPipeline.Fit(trainTestData.TrainSet);

            // Valutazione modello di classificazione
            Console.WriteLine("🔎 Valutazione modello di classificazione...");
            var classificationMetrics = EvaluateClassificationModel(classificationModel, trainTestData.TestSet);

            // Valutazione modelli di localizzazione X e Y
            Console.WriteLine("🔎 Valutazione modello di localizzazione X...");
            var xMetrics = EvaluateLocalizationXModel(localizationXModel, trainTestData.TestSet);

            Console.WriteLine("🔎 Valutazione modello di localizzazione Y...");
            var yMetrics = EvaluateLocalizationYModel(localizationYModel, trainTestData.TestSet);

            // Creazione del modello combinato
            var combinedModel = new CombinedObjectDetectionModel(
                _mlContext,
                classificationModel,
                localizationXModel,
                localizationYModel,
                _imageWidth,
                _imageHeight, _objectType);

            Console.WriteLine("💾 Salvataggio modello combinato...");
            combinedModel.Save(_modelPath);
            Console.WriteLine($"✅ Modello salvato con successo in: {_modelPath}");

            // Test del modello su alcuni esempi
            TestModelOnSamples(combinedModel, trainTestData.TestSet);
        }

        /// <summary>
        /// Valuta il modello di localizzazione X
        /// </summary>
        private RegressionMetrics EvaluateLocalizationXModel(ITransformer model, IDataView testData)
        {
            var predictions = model.Transform(testData);

            // Uso alternativo: Converto in enumerable e filtro manualmente anziché FilterRowsByColumn
            var testSamples = _mlContext.Data.CreateEnumerable<ObjectLocalizationEvalXData>(
                predictions, reuseRowObject: false)
                .Where(x => x.Label == _objectType)
                .ToList();

            // Creo un nuovo DataView dai dati filtrati
            var objectRows = _mlContext.Data.LoadFromEnumerable(testSamples);

            // Valuta le metriche per X
            var xMetrics = _mlContext.Regression.Evaluate(
                objectRows,
                labelColumnName: nameof(ObjectImageData.X),
                scoreColumnName: "ScoreX");

            Console.WriteLine($"   X - R² Score: {xMetrics.RSquared:F4}");
            Console.WriteLine($"   X - MSE: {xMetrics.MeanSquaredError:F4}");
            Console.WriteLine($"   X - RMSE: {xMetrics.RootMeanSquaredError:F4}");

            return xMetrics;
        }

        /// <summary>
        /// Valuta il modello di localizzazione Y
        /// </summary>
        private RegressionMetrics EvaluateLocalizationYModel(ITransformer model, IDataView testData)
        {
            var predictions = model.Transform(testData);

            // Uso alternativo: Converto in enumerable e filtro manualmente anziché FilterRowsByColumn
            var testSamples = _mlContext.Data.CreateEnumerable<ObjectLocalizationEvalYData>(
                predictions, reuseRowObject: false)
                .Where(x => x.Label == _objectType)
                .ToList();

            // Creo un nuovo DataView dai dati filtrati
            var objectRows = _mlContext.Data.LoadFromEnumerable(testSamples);

            // Valuta le metriche per Y
            var yMetrics = _mlContext.Regression.Evaluate(
                objectRows,
                labelColumnName: nameof(ObjectImageData.Y),
                scoreColumnName: "ScoreY");

            Console.WriteLine($"   Y - R² Score: {yMetrics.RSquared:F4}");
            Console.WriteLine($"   Y - MSE: {yMetrics.MeanSquaredError:F4}");
            Console.WriteLine($"   Y - RMSE: {yMetrics.RootMeanSquaredError:F4}");

            return yMetrics;
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
        /// Costruisce la pipeline per il modello di localizzazione
        /// </summary>
        private IEstimator<ITransformer> BuildLocalizationPipeline()
        {
            Console.WriteLine("📊 Preparazione pipeline di localizzazione...");

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

            // Pipeline per X
            var xPipeline = featureExtractionPipeline
                .Append(_mlContext.Regression.Trainers.FastTree(
                    labelColumnName: nameof(ObjectImageData.X),
                    featureColumnName: "NormalizedFeatures",
                    numberOfLeaves: 20,
                    numberOfTrees: 100,
                    minimumExampleCountPerLeaf: 10))
                .Append(_mlContext.Transforms.CopyColumns(
                    outputColumnName: "ScoreX",
                    inputColumnName: "Score"));

            // Addestra e salva il modello X
            return xPipeline;
        }

        /// <summary>
        /// Costruisce la pipeline per il modello di localizzazione Y
        /// </summary>
        private IEstimator<ITransformer> BuildLocalizationYPipeline()
        {
            // Pipeline di base per l'estrazione di feature (stessa di X)
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

            // Pipeline per Y
            var yPipeline = featureExtractionPipeline
                .Append(_mlContext.Regression.Trainers.FastTree(
                    labelColumnName: nameof(ObjectImageData.Y),
                    featureColumnName: "NormalizedFeatures",
                    numberOfLeaves: 20,
                    numberOfTrees: 100,
                    minimumExampleCountPerLeaf: 10))
                .Append(_mlContext.Transforms.CopyColumns(
                    outputColumnName: "ScoreY",
                    inputColumnName: "Score"));

            return yPipeline;
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
        /// Valuta il modello di localizzazione
        /// </summary>
        private (RegressionMetrics X, RegressionMetrics Y) EvaluateLocalizationModel(ITransformer model, IDataView testData)
        {
            var predictions = model.Transform(testData);

            // Uso alternativo: Converto in enumerable e filtro manualmente anziché FilterRowsByColumn
            var testSamples = _mlContext.Data.CreateEnumerable<ObjectLocalizationEvalData>(
                predictions, reuseRowObject: false)
                .Where(x => x.Label == _objectType)
                .ToList();

            // Creo un nuovo DataView dai dati filtrati
            var objectRows = _mlContext.Data.LoadFromEnumerable(testSamples);

            // Valuta le metriche per X
            var xMetrics = _mlContext.Regression.Evaluate(
                objectRows,
                labelColumnName: nameof(ObjectImageData.X),
                scoreColumnName: "ScoreX");

            Console.WriteLine($"   X - R² Score: {xMetrics.RSquared:F4}");
            Console.WriteLine($"   X - MSE: {xMetrics.MeanSquaredError:F4}");
            Console.WriteLine($"   X - RMSE: {xMetrics.RootMeanSquaredError:F4}");

            // Valuta le metriche per Y
            var yMetrics = _mlContext.Regression.Evaluate(
                objectRows,
                labelColumnName: nameof(ObjectImageData.Y),
                scoreColumnName: "ScoreY");

            Console.WriteLine($"   Y - R² Score: {yMetrics.RSquared:F4}");
            Console.WriteLine($"   Y - MSE: {yMetrics.MeanSquaredError:F4}");
            Console.WriteLine($"   Y - RMSE: {yMetrics.RootMeanSquaredError:F4}");

            return (xMetrics, yMetrics);
        }


        /// <summary>
        /// Test del modello su alcuni esempi del dataset
        /// </summary>
        private void TestModelOnSamples(CombinedObjectDetectionModel model, IDataView testData)
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
            CombinedObjectDetectionModel combinedModel;
            try
            {
                combinedModel = CombinedObjectDetectionModel.Load(_modelPath, _mlContext,"Mulo");
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
        /// Preelabora i dati per l'addestramento
        /// </summary>
        private List<ObjectImageData> PreprocessData(List<ObjectImageData> imageData)
        {
            Console.WriteLine("📊 Preprocessamento dati...");

            // Optionally filter out bad data
            var filteredData = imageData.Where(img =>
                img.Label == _objectType && (img.X != 0 || img.Y != 0) ||
                img.Label != _objectType).ToList();

            Console.WriteLine($"   Dati filtrati: {filteredData.Count} immagini valide");

            // Bilanciamento delle classi se necessario
            var positiveCount = filteredData.Count(img => img.Label == _objectType);
            var negativeCount = filteredData.Count(img => img.Label != _objectType);

            if (positiveCount > 0 && negativeCount > 0)
            {
                Console.WriteLine($"   Bilanciamento classi: Positivi {positiveCount}, Negativi {negativeCount}");

                // Se c'è uno sbilanciamento significativo, possiamo applicare sotto-campionamento
                // o sovra-campionamento se necessario
                if (positiveCount < negativeCount / 3)
                {
                    Console.WriteLine("   Applicazione di sovra-campionamento per la classe positiva...");
                    filteredData = ApplyOversampling(filteredData, _objectType);
                }
                else if (negativeCount < positiveCount / 3)
                {
                    Console.WriteLine("   Applicazione di sotto-campionamento per la classe negativa...");
                    filteredData = ApplyUndersampling(filteredData, _objectType);
                }
            }

            Console.WriteLine($"   Dataset finale: {filteredData.Count} immagini");
            Console.WriteLine($"   - {filteredData.Count(img => img.Label == _objectType)} immagini con {_objectType}");
            Console.WriteLine($"   - {filteredData.Count(img => img.Label != _objectType)} immagini senza {_objectType}");

            return filteredData;
        }

        /// <summary>
        /// Applica sovra-campionamento alla classe minoritaria
        /// </summary>
        private List<ObjectImageData> ApplyOversampling(List<ObjectImageData> data, string minorityClass)
        {
            var result = new List<ObjectImageData>(data);
            var minorityData = data.Where(img => img.Label == minorityClass).ToList();
            var majorityData = data.Where(img => img.Label != minorityClass).ToList();

            // Calcola quanti esempi aggiungere
            int targetCount = majorityData.Count / 3;
            if (minorityData.Count >= targetCount) return result;

            int toAdd = targetCount - minorityData.Count;
            var random = new Random(42);

            for (int i = 0; i < toAdd; i++)
            {
                var source = minorityData[random.Next(minorityData.Count)];
                result.Add(new ObjectImageData
                {
                    ImagePath = source.ImagePath,
                    Label = source.Label,
                    X = source.X,
                    Y = source.Y
                });
            }

            return result;
        }

        /// <summary>
        /// Applica sotto-campionamento alla classe maggioritaria
        /// </summary>
        private List<ObjectImageData> ApplyUndersampling(List<ObjectImageData> data, string minorityClass)
        {
            var minorityData = data.Where(img => img.Label == minorityClass).ToList();
            var majorityData = data.Where(img => img.Label != minorityClass).ToList();

            // Calcola quanti esempi mantenere
            int targetCount = minorityData.Count * 3;
            if (majorityData.Count <= targetCount) return data;

            var random = new Random(42);
            var selectedMajorityData = majorityData.OrderBy(x => random.Next()).Take(targetCount).ToList();

            var result = new List<ObjectImageData>();
            result.AddRange(minorityData);
            result.AddRange(selectedMajorityData);

            return result;
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

            return images;
        }
    }

    public class CombinedObjectDetectionModel
    {
        private readonly MLContext _mlContext;
        private readonly ITransformer _classificationModel;
        private readonly ITransformer _localizationXModel;
        private readonly ITransformer _localizationYModel;
        private readonly int _imageWidth;
        private readonly int _imageHeight;
        private string _objectType;
        private readonly PredictionEngine<ObjectImageData, ObjectClassificationPrediction> _classificationEngine;
        private readonly PredictionEngine<ObjectImageData, ObjectLocalizationXPrediction> _localizationXEngine;
        private readonly PredictionEngine<ObjectImageData, ObjectLocalizationYPrediction> _localizationYEngine;

        public CombinedObjectDetectionModel(
            MLContext mlContext,
            ITransformer classificationModel,
            ITransformer localizationXModel,
            ITransformer localizationYModel,
            int imageWidth,
            int imageHeight,
            string objectType)
        {
            _mlContext = mlContext;
            _classificationModel = classificationModel;
            _localizationXModel = localizationXModel;
            _localizationYModel = localizationYModel;
            _imageWidth = imageWidth;
            _imageHeight = imageHeight;

            _classificationEngine = _mlContext.Model.CreatePredictionEngine<ObjectImageData, ObjectClassificationPrediction>(_classificationModel);
            _localizationXEngine = _mlContext.Model.CreatePredictionEngine<ObjectImageData, ObjectLocalizationXPrediction>(_localizationXModel);
            _localizationYEngine = _mlContext.Model.CreatePredictionEngine<ObjectImageData, ObjectLocalizationYPrediction>(_localizationYModel);
            _objectType = objectType;
        }

        public ObjectPrediction Predict(ObjectImageData input)
        {
            // Prima esegui la classificazione
            var classificationPrediction = _classificationEngine.Predict(input);

            // Poi esegui la localizzazione solo se necessario
            float x = 0;
            float y = 0;

            // Trova l'indice della classe positiva (oggetto)
            var positiveClassIndex = Array.FindIndex(classificationPrediction.Score,
                s => s == classificationPrediction.Score.Max());

            var predictedLabel = classificationPrediction.PredictedLabel;

            if (positiveClassIndex == 0 || predictedLabel == _objectType) // Se la classe positiva è la prima o è l'oggetto cercato
            {
                var localizationXPrediction = _localizationXEngine.Predict(input);
                var localizationYPrediction = _localizationYEngine.Predict(input);
                x = localizationXPrediction.X;
                y = localizationYPrediction.Y;

                Console.WriteLine($"Coordinate predette: X={x:F4}, Y={y:F4}");
            }

            return new ObjectPrediction
            {
                PredictedLabel = predictedLabel,
                Score = classificationPrediction.Score,
                X = x,
                Y = y
            };
        }

        public void Save(string directoryPath)
        {
            Directory.CreateDirectory(directoryPath);

            // Salva i modelli
            _mlContext.Model.Save(_classificationModel, null, Path.Combine(directoryPath, "classification.zip"));
            _mlContext.Model.Save(_localizationXModel, null, Path.Combine(directoryPath, "localization_x.zip"));
            _mlContext.Model.Save(_localizationYModel, null, Path.Combine(directoryPath, "localization_y.zip"));

            // Salva i metadati
            using (var writer = new StreamWriter(Path.Combine(directoryPath, "metadata.csv")))
            {
                writer.WriteLine($"{_imageWidth},{_imageHeight}");
            }
        }

        public static CombinedObjectDetectionModel Load(string directoryPath, MLContext mlContext, string objectType)
        {
            var classificationModel = mlContext.Model.Load(Path.Combine(directoryPath, "classification.zip"), out _);
            var localizationXModel = mlContext.Model.Load(Path.Combine(directoryPath, "localization_x.zip"), out _);
            var localizationYModel = mlContext.Model.Load(Path.Combine(directoryPath, "localization_y.zip"), out _);

            // Carica i metadati
            int imageWidth = 224;
            int imageHeight = 224;

            var metadataPath = Path.Combine(directoryPath, "metadata.csv");
            if (File.Exists(metadataPath))
            {
                var line = File.ReadAllLines(metadataPath).FirstOrDefault();
                if (line != null)
                {
                    var parts = line.Split(',');
                    if (parts.Length >= 2)
                    {
                        int.TryParse(parts[0], out imageWidth);
                        int.TryParse(parts[1], out imageHeight);
                    }
                }
            }

            return new CombinedObjectDetectionModel(
                mlContext,
                classificationModel,
                localizationXModel,
                localizationYModel,
                imageWidth,
                imageHeight, objectType);
        }


  
    }

    /// <summary>
    /// Dati di input per l'addestramento e il rilevamento degli oggetti.
    /// </summary>
    public class ObjectImageData
    {
        public string ImagePath { get; set; }
        [ColumnName("Label")]
        public string Label { get; set; }
        public float X { get; set; }
        public float Y { get; set; }
    }

    /// <summary>
    /// Risultato del rilevamento dell'oggetto.
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


    /// <summary>
    /// Previsione di localizzazione delle coordinate dell'oggetto.
    /// </summary>

    /// <summary>
    /// Previsione di classificazione dell'oggetto.
    /// </summary>
    public class ObjectClassificationPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabel { get; set; }

        [ColumnName("Score")]
        public float[] Score { get; set; }
    }

    public class ObjectLocalizationEvalData
    {
        public string Label { get; set; }
        public float X { get; set; }
        public float Y { get; set; }
        public float ScoreX { get; set; }
        public float ScoreY { get; set; }
    }

    /// <summary>
    /// Previsione di localizzazione delle coordinate dell'oggetto.
    /// </summary>
    public class ObjectLocalizationPrediction
    {
        [ColumnName("ScoreX")]
        public float X { get; set; }

        [ColumnName("ScoreY")]
        public float Y { get; set; }
    }

    /// <summary>
    /// Risultato combinato della previsione di classificazione e localizzazione.
    /// </summary>
    public class ObjectPrediction
    {
        public string PredictedLabel { get; set; }
        public float[] Score { get; set; }
        public float X { get; set; }
        public float Y { get; set; }
        public bool IsObjectDetected => PredictedLabel != null && Score != null && Score.Length > 0 && Score.Max() > 0.5f;
        public float Confidence => Score != null && Score.Length > 0 ? Score.Max() : 0;
    }

    // Classi di supporto per la valutazione
    public class ObjectLocalizationEvalXData
    {
        public string Label { get; set; }
        public float X { get; set; }
        public float ScoreX { get; set; }
    }

    public class ObjectLocalizationEvalYData
    {
        public string Label { get; set; }
        public float Y { get; set; }
        public float ScoreY { get; set; }
    }

    // Classi di previsione separate per X e Y
    public class ObjectLocalizationXPrediction
    {
        [ColumnName("ScoreX")]
        public float X { get; set; }
    }

    public class ObjectLocalizationYPrediction
    {
        [ColumnName("ScoreY")]
        public float Y { get; set; }
    }
}