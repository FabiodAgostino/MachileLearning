
/**
 * MuloDetectorObjectDetection
 * 
 * Questo servizio è progettato per rilevare la presenza e la posizione di muli nell'ambiente
 * di gioco di Ultima Online. Il servizio implementa una soluzione di visione artificiale che:
 * 
 * 1. CLASSIFICAZIONE: Identifica se in un'immagine è presente un mulo o meno
 * 2. LOCALIZZAZIONE: Determina le coordinate precise (X, Y) del mulo nell'immagine
 * 
 * Il servizio utilizza ML.NET per creare modelli separati ma coordinati che addestrando
 * un classificatore di immagini (per rilevare la presenza) e due regressori (per le coordinate X e Y).
 * Questi vengono poi combinati in un'unica soluzione per facilitarne l'utilizzo.
 * 
 * Funzionalità:
 * - Caricamento automatico del dataset da una struttura di cartelle organizzata
 * - Lettura di annotazioni di coordinate da file CSV
 * - Addestramento parallelo di modelli per classificazione e regressione
 * - Valutazione delle prestazioni del modello con metriche di accuratezza
 * - Salvataggio dei modelli in un'unica directory per uso futuro
 * - Inferenza su nuove immagini per rilevare e localizzare i muli
 * - Gestione avanzata degli errori durante caricamento e predizione
 * 
 * Approccio tecnico:
 * - Utilizza LoadImages invece di LoadRawImageBytes per generare un tipo Image compatibile
 * - Biforca la pipeline per gestire separatamente classificazione e regressione
 * - Implementa una classe CombinedMuloModel per unificare l'inferenza dei modelli separati
 * - Valuta e riporta metriche di accuratezza per il modello di classificazione
 * 
 * Struttura attesa del dataset:
 * /dataset/
 *    /Mulo/ - contiene immagini di muli (.png)
 *    /NonMulo/ - contiene immagini senza muli (.png)
 *    dataset.csv - file con formato: ImagePath,IsMulo,X,Y
 * 
 * Note particolari:
 * - Per immagini "NonMulo", le coordinate X,Y vengono impostate a (0,0)
 * - Le coordinate X,Y predette vengono restituite solo quando viene rilevato un mulo
 * - Il modello riporta metriche di accuratezza e performance al termine dell'addestramento
 * 
 * Struttura del modello salvato:
 * /model_directory/
 *    /transform.zip - Pipeline di trasformazione base
 *    /classification.zip - Modello di classificazione Mulo/NonMulo
 *    /x_regression.zip - Modello di regressione per coordinata X
 *    /y_regression.zip - Modello di regressione per coordinata Y
 * 
 * Librerie utilizzate:
 * - Microsoft.ML" Version="4.0.2"
 * - Microsoft.ML.DataView" Version="4.0.2"
 * - Microsoft.ML.ImageAnalytics" Version="4.0.2" 
 * - Microsoft.ML.OnnxRuntime" Version="1.21.0" 
 * - Microsoft.ML (2.0.0+)
 * - Microsoft.ML.OnnxTransformer" Version="4.0.2" 
 * - Microsoft.ML.Vision" Version="4.0.2" 
 * - Microsoft.ML.ImageAnalytics (2.0.0+)
 * - SciSharp.TensorFlow.Redist" Version="1.15.0" 
 * - Microsoft.ML.FastTree" Version="4.0.2" 
 * - SixLabors.ImageSharp" Version="3.1.7" 
 */

using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
namespace MachineLearning.ServicesDetection
{

    public class MuloDetectorObjectDetection
    {
        private readonly MLContext _mlContext;
        private readonly string _datasetPath;
        private readonly string _modelPath;
        private readonly int _imageWidth = 224;
        private readonly int _imageHeight = 224;

        // Costruttore che accetta i percorsi necessari
        public MuloDetectorObjectDetection(string datasetPath, string modelPath)
        {
            _mlContext = new MLContext(seed: 42);
            _datasetPath = datasetPath;
            _modelPath = modelPath;
        }

        /// <summary>
        /// Addestra e salva il modello detector per i muli
        /// </summary>
        public void TrainAndSaveModel()
        {
            Console.WriteLine("🔍 Caricamento immagini dal dataset...");
            var imageData = LoadImagesWithAnnotations();

            // Carichiamo i dati come IDataView
            var dataView = _mlContext.Data.LoadFromEnumerable(imageData);

            // Dividiamo in training e test set
            var trainTestData = _mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

            Console.WriteLine("🏗️ Costruzione pipeline di addestramento...");

            // Pipeline di trasformazione e addestramento
            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey(
                    outputColumnName: "LabelKey",
                    inputColumnName: nameof(MuloImageData.Label))
                .Append(_mlContext.Transforms.LoadImages(
                    outputColumnName: "Image",
                    imageFolder: _datasetPath,
                    inputColumnName: nameof(MuloImageData.ImagePath)))
                .Append(_mlContext.Transforms.ResizeImages(
                    outputColumnName: "ResizedImage",
                    imageWidth: _imageWidth,
                    imageHeight: _imageHeight,
                    inputColumnName: "Image"))
                .Append(_mlContext.Transforms.ExtractPixels(
                    outputColumnName: "ImageFeatures",
                    inputColumnName: "ResizedImage",
                    interleavePixelColors: true,
                    offsetImage: 117));

            // Ora biforichiamo la pipeline per fare sia classificazione che regressione
            // Pipeline di classificazione per predire se è un mulo o meno
            var classificationPipeline = pipeline.Append(_mlContext.Transforms.CopyColumns(
                    outputColumnName: "IsMuloFeatures",
                    inputColumnName: "ImageFeatures"))
                .Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(
                    labelColumnName: "LabelKey",
                    featureColumnName: "IsMuloFeatures"))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue(
                    outputColumnName: "PredictedLabel",
                    inputColumnName: "PredictedLabel"));

            // Pipeline di regressione per predire X 
            var xRegressionPipeline = pipeline.Append(_mlContext.Transforms.CopyColumns(
                    outputColumnName: "XFeatures",
                    inputColumnName: "ImageFeatures"))
                .Append(_mlContext.Regression.Trainers.Sdca(
                    labelColumnName: nameof(MuloImageData.X),
                    featureColumnName: "XFeatures"));

            // Pipeline di regressione per predire Y
            var yRegressionPipeline = pipeline.Append(_mlContext.Transforms.CopyColumns(
                    outputColumnName: "YFeatures",
                    inputColumnName: "ImageFeatures"))
                .Append(_mlContext.Regression.Trainers.Sdca(
                    labelColumnName: nameof(MuloImageData.Y),
                    featureColumnName: "YFeatures"));

            Console.WriteLine("🏁 Inizio addestramento modelli...");

            // Addestramento dei modelli singolarmente
            Console.WriteLine("   Addestramento classificatore (Mulo/NonMulo)...");
            var classificationModel = classificationPipeline.Fit(trainTestData.TrainSet);

            Console.WriteLine("   Addestramento regressore per X...");
            var xRegressionModel = xRegressionPipeline.Fit(trainTestData.TrainSet);

            Console.WriteLine("   Addestramento regressore per Y...");
            var yRegressionModel = yRegressionPipeline.Fit(trainTestData.TrainSet);

            // Ora uniamo tutte le trasformazioni e i modelli in un unico TransformerChain
            var transformationsPipeline = pipeline.Fit(trainTestData.TrainSet);

            // Creiamo un nuovo dataview trasformato con la pipeline di base
            IDataView transformedData = transformationsPipeline.Transform(trainTestData.TrainSet);

            // Creiamo un dataset per salvarci tutte le predizioni
            var predictionData = _mlContext.Data.LoadFromEnumerable(new List<MuloImagePrediction>());

            // Combiniamo tutte le predizioni utilizzando il método di trasformazione col dataview
            var transformers = new List<ITransformer>
            {
                classificationModel,
                xRegressionModel,
                yRegressionModel
            };

            var combinedModel = new CombinedMuloModel(
                _mlContext,
                transformationsPipeline,
                classificationModel,
                xRegressionModel,
                yRegressionModel);

            // Salviamo il modello combinato
            Console.WriteLine("💾 Salvataggio modello combinato...");
            combinedModel.Save(_modelPath);
            Console.WriteLine($"✅ Modello salvato con successo in: {_modelPath}");

            // Valuta il modello di classificazione 
            var testPredictions = classificationModel.Transform(transformationsPipeline.Transform(trainTestData.TestSet));
            var metrics = _mlContext.MulticlassClassification.Evaluate(testPredictions, labelColumnName: "LabelKey");

            Console.WriteLine($"✅ Addestramento completato");
            Console.WriteLine($"   Accuratezza classificazione: {metrics.MicroAccuracy:P2}");
            Console.WriteLine($"   Log Loss: {metrics.LogLoss:F4}");
        }

        /// <summary>
        /// Rileva la presenza e la posizione di un mulo in un'immagine
        /// </summary>
        /// <param name="imagePath">Percorso dell'immagine da analizzare</param>
        /// <returns>Risultato della rilevazione</returns>
        public MuloDetectionResult DetectMulo(string imagePath)
        {
            // Carica il modello combinato
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

            // Prepara i dati di input
            var imageData = new MuloImageData
            {
                ImagePath = imagePath,
                Label = string.Empty, // Non serve per la predizione
                X = 0,
                Y = 0
            };

            // Esegui la predizione
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

            // Calcola la confidenza
            float confidence = 0;
            if (prediction.Score != null && prediction.Score.Length > 0)
            {
                confidence = prediction.Score.Max();
            }

            // Determina se è un mulo
            bool isMulo = prediction.PredictedLabel == "Mulo";

            // Restituisci il risultato
            return new MuloDetectionResult
            {
                IsMuloDetected = isMulo,
                X = isMulo ? prediction.X : 0,
                Y = isMulo ? prediction.Y : 0,
                Confidence = confidence
            };
        }

        /// <summary>
        /// Carica le immagini e le annotazioni dal dataset
        /// </summary>
        private List<MuloImageData> LoadImagesWithAnnotations()
        {
            var images = new List<MuloImageData>();
            var categories = new[] { "Mulo", "NonMulo" };

            // Carica le coordinate da CSV
            var labelDict = new Dictionary<string, (float X, float Y)>();
            var csvPath = Path.Combine(_datasetPath, "dataset.csv");

            if (File.Exists(csvPath))
            {
                Console.WriteLine($"📄 Caricamento annotazioni da: {csvPath}");
                var lines = File.ReadAllLines(csvPath).Skip(1); // Salta intestazione
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

            // Carica le immagini da entrambe le categorie
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

                    // Ottieni le coordinate se disponibili
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
    /// Classe che rappresenta un modello combinato per predire la presenza e le coordinate di un mulo
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

            // Crea prediction engines
            _classificationEngine = mlContext.Model.CreatePredictionEngine<MuloImageData, MuloClassificationPrediction>(classificationModel);
            _xRegressionEngine = mlContext.Model.CreatePredictionEngine<MuloImageData, MuloXRegressionPrediction>(xRegressionModel);
            _yRegressionEngine = mlContext.Model.CreatePredictionEngine<MuloImageData, MuloYRegressionPrediction>(yRegressionModel);
        }

        public MuloImagePrediction Predict(MuloImageData input)
        {
            // Esegui le predizioni separate
            var classificationPrediction = _classificationEngine.Predict(input);
            var xPrediction = _xRegressionEngine.Predict(input);
            var yPrediction = _yRegressionEngine.Predict(input);

            // Combina i risultati
            return new MuloImagePrediction
            {
                PredictedLabel = classificationPrediction.PredictedLabel,
                Score = classificationPrediction.Score,
                X = xPrediction.X,
                Y = yPrediction.Y
            };
        }

        // Salva i modelli in un'unica directory
        public void Save(string directoryPath)
        {
            Directory.CreateDirectory(directoryPath);

            // Salva i modelli individuali
            _mlContext.Model.Save(_transformationPipeline, null, Path.Combine(directoryPath, "transform.zip"));
            _mlContext.Model.Save(_classificationModel, null, Path.Combine(directoryPath, "classification.zip"));
            _mlContext.Model.Save(_xRegressionModel, null, Path.Combine(directoryPath, "x_regression.zip"));
            _mlContext.Model.Save(_yRegressionModel, null, Path.Combine(directoryPath, "y_regression.zip"));
        }

        // Carica i modelli da una directory
        public static CombinedMuloModel Load(string directoryPath, MLContext mlContext)
        {
            // Carica i modelli individuali
            ITransformer transformationPipeline = mlContext.Model.Load(Path.Combine(directoryPath, "transform.zip"), out _);
            ITransformer classificationModel = mlContext.Model.Load(Path.Combine(directoryPath, "classification.zip"), out _);
            ITransformer xRegressionModel = mlContext.Model.Load(Path.Combine(directoryPath, "x_regression.zip"), out _);
            ITransformer yRegressionModel = mlContext.Model.Load(Path.Combine(directoryPath, "y_regression.zip"), out _);

            // Crea un nuovo modello combinato
            return new CombinedMuloModel(
                mlContext,
                transformationPipeline,
                classificationModel,
                xRegressionModel,
                yRegressionModel);
        }
    }

    // Classi per i dati di input
    public class MuloImageData
    {
        public string ImagePath { get; set; }
        public string Label { get; set; }
        public float X { get; set; }
        public float Y { get; set; }
    }

    // Classi per le predizioni
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
}
