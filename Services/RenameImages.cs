using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearning.Services
{
    public class RenameImages
    {
        public void Rename()
        {
            string baseFolder = "Dataset";
            string muloFolder = Path.Combine(baseFolder, "Mulo");
            string nonMuloFolder = Path.Combine(baseFolder, "NonMulo");
            string outputCsv = Path.Combine(baseFolder, "dataset.csv");

            if (!Directory.Exists(muloFolder) || !Directory.Exists(nonMuloFolder))
            {
                Console.WriteLine("Le cartelle 'Mulo' e 'NonMulo' devono esistere dentro la cartella 'Dataset'");
                return;
            }

            int muloCounter = 1;
            int nonMuloCounter = 1;
            var sb = new StringBuilder();
            sb.AppendLine("ImagePath,IsMulo,X,Y");

            // Elabora immagini di muli
            foreach (var file in Directory.GetFiles(muloFolder))
            {
                string extension = Path.GetExtension(file);
                string newFileName = $"Mulo{muloCounter}{extension}";
                string newPath = Path.Combine(muloFolder, newFileName);
                File.Move(file, newPath, true); // Rinomina nella stessa cartella
                string relativePath = Path.Combine("Dataset", "Mulo", newFileName).Replace("\\", "/");
                sb.AppendLine($"{relativePath},1,0,0");
                muloCounter++;
            }

            // Elabora immagini senza muli
            foreach (var file in Directory.GetFiles(nonMuloFolder))
            {
                string extension = Path.GetExtension(file);
                string newFileName = $"NonMulo{nonMuloCounter}{extension}";
                string newPath = Path.Combine(nonMuloFolder, newFileName);
                File.Move(file, newPath, true); // Rinomina nella stessa cartella
                string relativePath = Path.Combine("Dataset", "NonMulo", newFileName).Replace("\\", "/");
                sb.AppendLine($"{relativePath},0,0,0");
                nonMuloCounter++;
            }

            // Scrivi il file CSV
            File.WriteAllText(outputCsv, sb.ToString());
            Console.WriteLine($"CSV creato con successo: {outputCsv}");
        }
    }
}
