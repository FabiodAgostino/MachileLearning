using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MachineLearning
{
    public class Dataset
    {
        public void RaccogliDataset(int numScreenshots)
        {
            Directory.CreateDirectory("Dataset/Mulo");
            Directory.CreateDirectory("Dataset/NonMulo");

            Console.WriteLine("Premi 'M' quando il mulo è visibile, 'N' quando non è visibile, 'ESC' per terminare");

            for (int i = 0; i < numScreenshots; i++)
            {
                Console.WriteLine($"Screenshot {i + 1}/{numScreenshots}");

                // Attendi input utente
                ConsoleKeyInfo key;
                do
                {
                    key = Console.ReadKey(true);
                } while (key.Key != ConsoleKey.M && key.Key != ConsoleKey.N && key.Key != ConsoleKey.Escape);

                if (key.Key == ConsoleKey.Escape)
                    break;

                // Cattura screenshot
                //var screenshot = AutoClicker.Service.ExtensionMethod.Image.CaptureCenterScreenshot();
                //string folder = key.Key == ConsoleKey.M ? "Mulo" : "NonMulo";
                //string filename = $"Dataset/{folder}/{folder.ToLower()}_{i + 1:D3}.png";

                //screenshot.bitmap.Save(filename);
                //Console.WriteLine($"Salvato {filename}");

                // Breve pausa per cambiare scena nel gioco
                Thread.Sleep(1000);
            }
        }
    }
}
