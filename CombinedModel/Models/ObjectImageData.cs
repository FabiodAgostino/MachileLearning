using Microsoft.ML.Data;

namespace DetectorModel.Models
{
    /// <summary>
    /// Classe per l'input del modello
    /// </summary>
    public class ObjectImageData
    {
        [LoadColumn(0)]
        public string ImagePath { get; set; }

        [LoadColumn(1)]
        public string Label { get; set; }

        [LoadColumn(2)]
        public float X { get; set; }

        [LoadColumn(3)]
        public float Y { get; set; }

        public ObjectImageData()
        {
            
        }

        public ObjectImageData(string imagePath)
        {
            ImagePath = imagePath;
            Label = string.Empty;
            X = 0;
            Y = 0;
        }
    }
}
