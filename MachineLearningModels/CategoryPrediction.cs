namespace MachineLearningModels
{
    public class CategoryPrediction
    {
        public string ImagePath { get; set; }
        public string Label { get; set; }
        public float[] Score { get; set; }
        public string PredictedLabelValue { get; set; }
    }
}
