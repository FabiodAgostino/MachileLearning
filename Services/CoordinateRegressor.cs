using MachineLearning.Models;
using Microsoft.ML;
using System;
using System.Linq;

namespace MachineLearning.Services
{
    public class CoordinateRegressor
    {
        public void Train(string csvPath, string modelPathX, string modelPathY)
        {
            var mlContext = new MLContext();

            var data = mlContext.Data.LoadFromTextFile<MuloData>(csvPath, hasHeader: true, separatorChar: ',');

            // Filtra solo i muli
            var muloData = mlContext.Data.FilterRowsByColumn(data, "IsMulo", lowerBound: 1, upperBound: 1);

            var pipelineBase = mlContext.Transforms.LoadImages("ImagePath", "", nameof(MuloData.ImagePath))
                .Append(mlContext.Transforms.ResizeImages("ImagePath", 224, 224))
                .Append(mlContext.Transforms.ExtractPixels("ImagePath"))
                .Append(mlContext.Transforms.CopyColumns("Features", "ImagePath"));

            // Regressore per X
            var xPipeline = pipelineBase.Append(mlContext.Regression.Trainers.FastTree(labelColumnName: "X", featureColumnName: "Features"));
            var modelX = xPipeline.Fit(muloData);
            mlContext.Model.Save(modelX, muloData.Schema, modelPathX);

            // Regressore per Y
            var yPipeline = pipelineBase.Append(mlContext.Regression.Trainers.FastTree(labelColumnName: "Y", featureColumnName: "Features"));
            var modelY = yPipeline.Fit(muloData);
            mlContext.Model.Save(modelY, muloData.Schema, modelPathY);

            Console.WriteLine("✅ Modelli di regressione per X e Y addestrati e salvati.");
        }
    }

}
