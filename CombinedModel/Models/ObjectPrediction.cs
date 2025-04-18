namespace DetectorModel.Models
{
    /// <summary>
    /// Classe per il risultato della predizione combinata
    /// </summary>
    public class ObjectPrediction
    {
        public string PredictedLabel { get; set; }
        public float X { get; set; }
        public float Y { get; set; }
        public bool IsObjectDetected { get; set; }
        public float Confidence { get; set; }
    }
}
