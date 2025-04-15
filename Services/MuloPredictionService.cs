using MachineLearning.Models;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearning.Services
{
    public class MuloPredictionService
    {
        private readonly MLContext _mlContext;
        private readonly ITransformer _classifierModel;
        private readonly ITransformer _regressorXModel;
        private readonly ITransformer _regressorYModel;
        private readonly PredictionEngine<MuloData, ClassificationPrediction> _classificationEngine;
        private readonly PredictionEngine<MuloData, RegressionPrediction> _regressionXEngine;
        private readonly PredictionEngine<MuloData, RegressionPrediction> _regressionYEngine;

        public MuloPredictionService(string classifierPath, string regressorXPath, string regressorYPath)
        {
            _mlContext = new MLContext();

            // Carica i modelli
            _classifierModel = _mlContext.Model.Load(classifierPath, out _);
            _regressorXModel = _mlContext.Model.Load(regressorXPath, out _);
            _regressorYModel = _mlContext.Model.Load(regressorYPath, out _);

            _classificationEngine = _mlContext.Model.CreatePredictionEngine<MuloData, ClassificationPrediction>(_classifierModel);
            _regressionXEngine = _mlContext.Model.CreatePredictionEngine<MuloData, RegressionPrediction>(_regressorXModel);
            _regressionYEngine = _mlContext.Model.CreatePredictionEngine<MuloData, RegressionPrediction>(_regressorYModel);
        }

        public void Predict(string imagePath)
        {
            var input = new MuloData
            {
                ImagePath = imagePath
            };

            var classification = _classificationEngine.Predict(input);
            var isMulo = classification.PredictedLabel == true;

            if (isMulo)
            {
                var predX = _regressionXEngine.Predict(input).Score;
                var predY = _regressionYEngine.Predict(input).Score;

                Console.WriteLine($"🟢 È un mulo! Coordinate stimate: X={predX:F0}, Y={predY:F0}");
            }
            else
            {
                Console.WriteLine("🔴 Non è un mulo.");
            }
        }
    }

    // Output delle prediction
    public class ClassificationPrediction
    {
        public bool PredictedLabel { get; set; }
        public float[] Score { get; set; }
    }

    public class RegressionPrediction
    {
        public float Score { get; set; }
    }
}
