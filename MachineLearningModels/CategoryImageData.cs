using Microsoft.ML.Data;

namespace MachineLearningModels
{
    public class CategoryImageData
    {
        [LoadColumn(0)]
        public string ImagePath { get; set; }

        [LoadColumn(1)]
        public string Label { get; set; }

        public static CategoryImageData FromFile(string imagePath, string label)
        {
            return new CategoryImageData
            {
                ImagePath = imagePath,
                Label = label
            };
        }
    }
}
