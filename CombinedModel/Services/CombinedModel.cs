using DetectorModel.Models;
using Microsoft.ML;

namespace DetectorModel.Services
{
    /// <summary>
    /// Classe per rappresentare il modello combinato 
    /// </summary>
    public class CombinedModel
    {
        private readonly MLContext _mlContext;
        private readonly ITransformer _classificationModel;
        private readonly ITransformer _xLocalizationModel;
        private readonly ITransformer _yLocalizationModel;
        private readonly int _imageWidth;
        private readonly int _imageHeight;
        private readonly string _objectType;
        private readonly PredictionEngine<ObjectImageData, ClassificationPrediction> _classificationEngine;
        private readonly PredictionEngine<ObjectImageData, SinglePrediction> _xLocalizationEngine;
        private readonly PredictionEngine<ObjectImageData, SinglePrediction> _yLocalizationEngine;

        public CombinedModel(
            MLContext mlContext,
            ITransformer classificationModel,
            ITransformer xLocalizationModel,
            ITransformer yLocalizationModel,
            int imageWidth,
            int imageHeight,
            string objectType)
        {
            _mlContext = mlContext;
            _classificationModel = classificationModel;
            _xLocalizationModel = xLocalizationModel;
            _yLocalizationModel = yLocalizationModel;
            _imageWidth = imageWidth;
            _imageHeight = imageHeight;
            _objectType = objectType;

            // Crea i motori di predizione
            _classificationEngine = _mlContext.Model.CreatePredictionEngine<ObjectImageData, ClassificationPrediction>(_classificationModel);
            _xLocalizationEngine = _mlContext.Model.CreatePredictionEngine<ObjectImageData, SinglePrediction>(_xLocalizationModel);
            _yLocalizationEngine = _mlContext.Model.CreatePredictionEngine<ObjectImageData, SinglePrediction>(_yLocalizationModel);
        }

        /// <summary>
        /// Carica il modello combinato da disco
        /// </summary>
        public static CombinedModel Load(string modelPath, MLContext mlContext, string objectType)
        {
            // Carica i modelli dal percorso
            var modelDir = modelPath;
            if (!Directory.Exists(modelDir))
            {
                Directory.CreateDirectory(modelDir);
            }

            var classificationModelPath = Path.Combine(modelDir, "classification.zip");
            var xLocalizationModelPath = Path.Combine(modelDir, "x_localization.zip");
            var yLocalizationModelPath = Path.Combine(modelDir, "y_localization.zip");
            var configPath = Path.Combine(modelDir, "config.json");

            // Verifica che i file esistano
            if (!File.Exists(classificationModelPath) ||
                !File.Exists(xLocalizationModelPath) ||
                !File.Exists(yLocalizationModelPath))
            {
                throw new FileNotFoundException("Modelli non trovati. Esegui l'addestramento prima.");
            }

            // Carica configurazione
            var configJson = File.ReadAllText(configPath);
            var config = System.Text.Json.JsonSerializer.Deserialize<ModelConfig>(configJson);

            // Carica i modelli
            var classificationModel = mlContext.Model.Load(classificationModelPath, out _);
            var xLocalizationModel = mlContext.Model.Load(xLocalizationModelPath, out _);
            var yLocalizationModel = mlContext.Model.Load(yLocalizationModelPath, out _);

            return new CombinedModel(
                mlContext,
                classificationModel,
                xLocalizationModel,
                yLocalizationModel,
                config.ImageWidth,
                config.ImageHeight,
                objectType);
        }


        public static async Task<CombinedModel> LoadAsync(string modelPath, MLContext mlContext, string objectType)
        {
            // Carica i modelli dal percorso
            var modelDir = modelPath;
            if (!Directory.Exists(modelDir))
            {
                Directory.CreateDirectory(modelDir);
            }

            var classificationModelPath = Path.Combine(modelDir, "classification.zip");
            var xLocalizationModelPath = Path.Combine(modelDir, "x_localization.zip");
            var yLocalizationModelPath = Path.Combine(modelDir, "y_localization.zip");
            var configPath = Path.Combine(modelDir, "config.json");

            // Verifica che i file esistano
            if (!File.Exists(classificationModelPath) ||
                !File.Exists(xLocalizationModelPath) ||
                !File.Exists(yLocalizationModelPath))
            {
                throw new FileNotFoundException("Modelli non trovati. Esegui l'addestramento prima.");
            }

            // Carica configurazione in modo asincrono
            string configJson;
            using (var reader = new StreamReader(configPath))
            {
                configJson = await reader.ReadToEndAsync();
            }
            var config = System.Text.Json.JsonSerializer.Deserialize<ModelConfig>(configJson);

            // Carica i modelli in modo asincrono
            var models = await Task.Run(() => {
                var classification = mlContext.Model.Load(classificationModelPath, out _);
                var xLocalization = mlContext.Model.Load(xLocalizationModelPath, out _);
                var yLocalization = mlContext.Model.Load(yLocalizationModelPath, out _);

                return (classification, xLocalization, yLocalization);
            });

            return new CombinedModel(
                mlContext,
                models.classification,
                models.xLocalization,
                models.yLocalization,
                config.ImageWidth,
                config.ImageHeight,
                objectType);
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
            var xLocalizationModelPath = Path.Combine(modelDir, "x_localization.zip");
            var yLocalizationModelPath = Path.Combine(modelDir, "y_localization.zip");
            var configPath = Path.Combine(modelDir, "config.json");

            // Salva i modelli
            _mlContext.Model.Save(_classificationModel, null, classificationModelPath);
            _mlContext.Model.Save(_xLocalizationModel, null, xLocalizationModelPath);
            _mlContext.Model.Save(_yLocalizationModel, null, yLocalizationModelPath);

            // Salva configurazione
            var config = new ModelConfig
            {
                ImageWidth = _imageWidth,
                ImageHeight = _imageHeight,
                ObjectType = _objectType
            };

            var configJson = System.Text.Json.JsonSerializer.Serialize(config);
            File.WriteAllText(configPath, configJson);
        }

        /// <summary>
        /// Effettua la predizione combinata
        /// </summary>
        public ObjectPrediction Predict(ObjectImageData data)
        {
            // Predizione di classificazione
            var classificationResult = _classificationEngine.Predict(data);

            // Determina se l'oggetto è rilevato
            bool isObjectDetected = classificationResult.PredictedLabel.Equals(_objectType, StringComparison.OrdinalIgnoreCase);

            float x = 0, y = 0;

            // Predizione delle coordinate solo se l'oggetto è rilevato
            if (isObjectDetected)
            {
                var xResult = _xLocalizationEngine.Predict(data);
                var yResult = _yLocalizationEngine.Predict(data);

                x = xResult.Score;
                y = yResult.Score;

                // Limita le coordinate all'interno dell'immagine
                x = Math.Max(0, Math.Min(_imageWidth, x));
                y = Math.Max(0, Math.Min(_imageHeight, y));
            }

            // Calcola la confidenza dalla classificazione
            float confidence = 0;
            if (classificationResult.Score != null && classificationResult.Score.Length > 0)
            {
                for (int i = 0; i < classificationResult.Score.Length; i++)
                {
                    if (classificationResult.Score[i] > confidence &&
                        classificationResult.PredictedLabel.Equals(_objectType, StringComparison.OrdinalIgnoreCase))
                    {
                        confidence = classificationResult.Score[i];
                    }
                }
            }

            // Crea l'oggetto risultato
            return new ObjectPrediction
            {
                PredictedLabel = classificationResult.PredictedLabel,
                X = x,
                Y = y,
                IsObjectDetected = isObjectDetected,
                Confidence = confidence
            };
        }

        public async Task<ObjectPrediction> PredictAsync(ObjectImageData data)
        {
            // Esegui le predizioni in modo asincrono
            return await Task.Run(() => {
                // Predizione di classificazione
                var classificationResult = _classificationEngine.Predict(data);

                // Determina se l'oggetto è rilevato
                bool isObjectDetected = classificationResult.PredictedLabel.Equals(_objectType, StringComparison.OrdinalIgnoreCase);
                float x = 0, y = 0;

                // Predizione delle coordinate solo se l'oggetto è rilevato
                if (isObjectDetected)
                {
                    var xResult = _xLocalizationEngine.Predict(data);
                    var yResult = _yLocalizationEngine.Predict(data);
                    x = xResult.Score;
                    y = yResult.Score;

                    // Limita le coordinate all'interno dell'immagine
                    x = Math.Max(0, Math.Min(_imageWidth, x));
                    y = Math.Max(0, Math.Min(_imageHeight, y));
                }

                // Calcola la confidenza dalla classificazione
                float confidence = 0;
                if (classificationResult.Score != null && classificationResult.Score.Length > 0)
                {
                    for (int i = 0; i < classificationResult.Score.Length; i++)
                    {
                        if (classificationResult.Score[i] > confidence &&
                            classificationResult.PredictedLabel.Equals(_objectType, StringComparison.OrdinalIgnoreCase))
                        {
                            confidence = classificationResult.Score[i];
                        }
                    }
                }

                // Crea l'oggetto risultato
                return new ObjectPrediction
                {
                    PredictedLabel = classificationResult.PredictedLabel,
                    X = x,
                    Y = y,
                    IsObjectDetected = isObjectDetected,
                    Confidence = confidence
                };
            });
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
    }
}
