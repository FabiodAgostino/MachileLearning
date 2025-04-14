using Microsoft.ML.Data;

namespace MachineLearningModels
{
    // Modificare la classe MuloImageData per supportare più categorie
    public class MultiClassImageData
    {
        [LoadColumn(0)]
        public string ImagePath { get; set; }

        [LoadColumn(1)]
        public string Label { get; set; }  // Ora potrà contenere "Mulo", "Zaino", "InterfacciaTM", "AntiMacro", ecc.

        [LoadColumn(2)]
        public string Category { get; set; }  // Categoria dell'oggetto: "Mulo", "Zaino", "InterfacciaTM", "AntiMacro"

        public static MultiClassImageData FromFile(string imagePath, string label, string category)
        {
            return new MultiClassImageData
            {
                ImagePath = imagePath,
                Label = label,
                Category = category
            };
        }
    }
}
