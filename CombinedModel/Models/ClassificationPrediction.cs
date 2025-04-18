using Microsoft.ML.Data;

namespace DetectorModel.Models
{
    /// <summary>
    /// Classe per i risultati di classificazione
    /// </summary>
    public class ClassificationPrediction
    {
        public string PredictedLabel { get; set; }

        [VectorType(2)]
        public float[] Score { get; set; }
    }
}
