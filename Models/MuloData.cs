using Microsoft.ML.Data;

namespace MachineLearning.Models
{
    public class MuloData
    {
        [LoadColumn(0)] public string ImagePath;
        [LoadColumn(1)] public bool IsMulo;
        [LoadColumn(2)] public float X;
        [LoadColumn(3)] public float Y;
    }

    
}
