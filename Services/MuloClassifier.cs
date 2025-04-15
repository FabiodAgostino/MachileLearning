using MachineLearning.Models;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearning.Services
{
    public class MuloClassifier
    {
        public void Train(string csvPath, string modelPathClassifier, string modelPathRegressorX, string modelPathRegressorY)
        {
            var mlContext = new MLContext();

            // Carica i dati
            var data = mlContext.Data.LoadFromTextFile<MuloData>(
                csvPath, hasHeader: true, separatorChar: ',');

            // === 📌 PIPELINE COMUNE: pre-processamento immagini ===
            var preprocessingPipeline = mlContext.Transforms.LoadImages(
                    outputColumnName: "Image", imageFolder: "", inputColumnName: nameof(MuloData.ImagePath))
                .Append(mlContext.Transforms.ResizeImages(
                    outputColumnName: "Image", imageWidth: 224, imageHeight: 224))
                .Append(mlContext.Transforms.ExtractPixels("Image"))
                .Append(mlContext.Transforms.CopyColumns("Features", "Image"));

            var preprocessedData = preprocessingPipeline.Fit(data).Transform(data);

            // === 🧠 PIPELINE CLASSIFICAZIONE ===
            var classificationPipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", nameof(MuloData.IsMulo))
                .Append(mlContext.MulticlassClassification.Trainers
                    .ImageClassification(featureColumnName: "Image", labelColumnName: "Label"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var classificationModel = classificationPipeline.Fit(preprocessedData);
            mlContext.Model.Save(classificationModel, preprocessedData.Schema, modelPathClassifier);
            Console.WriteLine("✅ Modello classificatore salvato.");

            // === 📍 PIPELINE REGRESSIONE X ===
            var regressionXPipeline = mlContext.Regression.Trainers.FastTree(
                labelColumnName: "X", featureColumnName: "Features");

            var regressionXModel = regressionXPipeline.Fit(preprocessedData);
            mlContext.Model.Save(regressionXModel, preprocessedData.Schema, modelPathRegressorX);
            Console.WriteLine("✅ Modello regressore X salvato.");

            // === 📍 PIPELINE REGRESSIONE Y ===
            var regressionYPipeline = mlContext.Regression.Trainers.FastTree(
                labelColumnName: "Y", featureColumnName: "Features");

            var regressionYModel = regressionYPipeline.Fit(preprocessedData);
            mlContext.Model.Save(regressionYModel, preprocessedData.Schema, modelPathRegressorY);
            Console.WriteLine("✅ Modello regressore Y salvato.");
        }

    }
}
