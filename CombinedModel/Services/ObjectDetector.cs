using DetectorModel.Models;
using Microsoft.ML;
using System.Drawing;

namespace DetectorModel.Services
{
    public class ObjectDetector
    {
        /// <summary>
        /// Rileva la presenza e le coordinate dell'oggetto in una immagine e salva una nuova immagine con il punto rilevato.
        /// </summary>
        /// 
        private readonly MLContext _mlContext;
        private readonly string _modelPath;
        private readonly string _objectType;

        public ObjectDetector()
        {
            _mlContext = new MLContext(seed: 42);
        }
        public ObjectDetector(string modelPath, string objectType = "Object")
        {
            _mlContext = new MLContext(seed: 42);
            _modelPath = modelPath;
            _objectType = objectType;
        }

        public async Task<ObjectDetectionResult> DetectObjectAsync(string imagePath, string _modelPath, string typeObject = TypeObject.Mulo)
        {
            if (!ModelExist(_modelPath))
                return new ObjectDetectionResult() { ErrorMessage = $"Modello non esistente!" };

            var combinedModel = await CombinedModel.LoadAsync(_modelPath, _mlContext, typeObject);
            var objectPrediction = await combinedModel.PredictAsync(new ObjectImageData(imagePath));
            if (objectPrediction.IsObjectDetected)
            {
                // Carica l'immagine in modo asincrono
                using (var imageStream = new FileStream(imagePath, FileMode.Open, FileAccess.Read, FileShare.Read, 4096, true))
                using (var image = await Task.Run(() => System.Drawing.Image.FromStream(imageStream)))
                {
                    int actualWidth = image.Width;
                    int actualHeight = image.Height;
                    float scaledX = objectPrediction.X * (actualWidth / image.Width);
                    float scaledY = objectPrediction.Y * (actualHeight / image.Height);

                    Console.WriteLine($"Coordinate originali: X={objectPrediction.X:F2}, Y={objectPrediction.Y:F2}");
                    Console.WriteLine($"Coordinate scalate: X={scaledX:F2}, Y={scaledY:F2}");

                    // Usa le coordinate scalate
                    return new ObjectDetectionResult
                    {
                        IsObjectDetected = objectPrediction.IsObjectDetected,
                        X = scaledX,
                        Y = scaledY,
                        Confidence = objectPrediction.Confidence,
                        ObjectType = _objectType
                    };
                }
            }

            return new ObjectDetectionResult() { ErrorMessage = $"L'oggetto {typeObject} non è stato individuato." };
        }

        public ObjectDetectionResult DetectObject(string imagePath, string _modelPath, string typeObject = TypeObject.Mulo)
        {
            if (!ModelExist(_modelPath))
                return new ObjectDetectionResult() { ErrorMessage = $"Modello non esistente!"};

            var combinedModel = CombinedModel.Load(_modelPath, _mlContext, typeObject);
            var objectPrediction = combinedModel.Predict(new ObjectImageData(imagePath));
            if (objectPrediction.IsObjectDetected)
            {
                // Carica l'immagine in modo asincrono
                using (var image = System.Drawing.Image.FromFile(imagePath))
                {
                    int actualWidth = image.Width;
                    int actualHeight = image.Height;

                    // Scala le coordinate dal modello (224x224) alle dimensioni reali
                    float scaledX = objectPrediction.X * (actualWidth / 193.0f);
                    float scaledY = objectPrediction.Y * (actualHeight / 166.0f);


                    Console.WriteLine($"Coordinate originali: X={objectPrediction.X:F2}, Y={objectPrediction.Y:F2}");
                    Console.WriteLine($"Coordinate scalate: X={scaledX:F2}, Y={scaledY:F2}");

                    // Usa le coordinate scalate
                    return new ObjectDetectionResult
                    {
                        IsObjectDetected = objectPrediction.IsObjectDetected,
                        X = scaledX,
                        Y = scaledY,
                        Confidence = objectPrediction.Confidence,
                        ObjectType = _objectType
                    };
                }
            }

            return new ObjectDetectionResult() { ErrorMessage = $"L'oggetto {typeObject} non è stato individuato." };
        }

        private bool ModelExist(string _modelPath)
        {
            string modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Models", _modelPath);
            if (File.Exists(modelPath))
                return true;
            else
                return false;
        }

        public ObjectDetectionResult DetectObjectTest(string imagePath)
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

            string outputImagePath = string.Empty;

            if (isObjectDetected)
            {
                // Ottieni le dimensioni reali dell'immagine
                using (var image = System.Drawing.Image.FromFile(imagePath))
                {
                    int actualWidth = image.Width;
                    int actualHeight = image.Height;

                    // Scala le coordinate dal modello (224x224) alle dimensioni reali
                    float scaledX = prediction.X * (actualWidth / 193.0f);
                    float scaledY = prediction.Y * (actualHeight / 166.0f);


                    Console.WriteLine($"Coordinate originali: X={prediction.X:F2}, Y={prediction.Y:F2}");
                    Console.WriteLine($"Coordinate scalate: X={scaledX:F2}, Y={scaledY:F2}");

                    // Usa le coordinate scalate
                    return new ObjectDetectionResult
                    {
                        IsObjectDetected = isObjectDetected,
                        X = scaledX,
                        Y = scaledY,
                        Confidence = confidence,
                        ObjectType = _objectType
                    };
                }
            }

            return new ObjectDetectionResult
            {
                IsObjectDetected = isObjectDetected,
                X = prediction.X,
                Y = prediction.Y,
                Confidence = confidence,
                ObjectType = _objectType,
                MarkedImagePath = outputImagePath
            };
        }

        /// <summary>
        /// Genera una nuova immagine con un punto rosso sulla posizione rilevata.
        /// </summary>
        private string GenerateImageWithDot(string originalImagePath, float x, float y)
        {
            try
            {
                // Carica l'immagine originale
                using (System.Drawing.Bitmap originalImage = new System.Drawing.Bitmap(originalImagePath))
                {
                    // Crea un oggetto Graphics per disegnare sull'immagine
                    using (System.Drawing.Graphics g = System.Drawing.Graphics.FromImage(originalImage))
                    {
                        // Dimensione del punto
                        int dotSize = 5;

                        // Disegna un cerchio rosso pieno sulla posizione
                        using (System.Drawing.Brush brush = new System.Drawing.SolidBrush(System.Drawing.Color.Red))
                        {
                            g.FillEllipse(brush, x - dotSize / 2, y - dotSize / 2, dotSize, dotSize);
                        }

                        // Disegna un bordo bianco intorno al punto rosso per maggiore visibilità
                        using (System.Drawing.Pen pen = new System.Drawing.Pen(System.Drawing.Color.White, 1))
                        {
                            g.DrawEllipse(pen, x - dotSize / 2 - 1, y - dotSize / 2 - 1, dotSize + 2, dotSize + 2);
                        }
                    }

                    // Genera il percorso per la nuova immagine
                    string outputDirectory = Path.GetDirectoryName(originalImagePath);
                    string fileNameWithoutExtension = Path.GetFileNameWithoutExtension(originalImagePath);
                    string extension = Path.GetExtension(originalImagePath);
                    string outputPath = Path.Combine(outputDirectory, $"{fileNameWithoutExtension}_marked{extension}");

                    // Salva la nuova immagine
                    originalImage.Save(outputPath);

                    Console.WriteLine($"Immagine con punto salvata in: {outputPath}");
                    return outputPath;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Errore durante la generazione dell'immagine con punto: {ex.Message}");
                return string.Empty;
            }
        }
    }
}