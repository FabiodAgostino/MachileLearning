using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Windows.Forms;
using System;

namespace MachineLearning.Services
{
    public class CoordinateLabelerService
    {
        private List<string> _csvLines;
        private string _csvPath;
        private int _currentLine;

        public CoordinateLabelerService(string csvPath)
        {
            _csvPath = csvPath;
        }

        public void Run()
        {
            _csvLines = new List<string>(File.ReadAllLines(_csvPath));

            for (int i = 1; i < _csvLines.Count; i++)
            {
                var line = _csvLines[i];
                var parts = line.Split(',');

                if (parts.Length < 4)
                    continue;

                string path = parts[0];
                string isMulo = parts[1];
                string x = parts[2];
                string y = parts[3];

                if (isMulo == "1" && x == "0" && y == "0" && File.Exists(path))
                {
                    _currentLine = i;
                    ShowImageAndCapturePoint(path);
                }
            }

            File.WriteAllLines(_csvPath, _csvLines);
            Console.WriteLine("✅ Tutte le coordinate aggiornate.");
        }

        private void ShowImageAndCapturePoint(string imagePath)
        {
            Form form = new Form
            {
                Text = $"Clicca sul mulo: {Path.GetFileName(imagePath)}",
                StartPosition = FormStartPosition.CenterScreen,
                AutoSize = true
            };

            PictureBox pictureBox = new PictureBox
            {
                Image = Image.FromFile(imagePath),
                SizeMode = PictureBoxSizeMode.Normal,
                Dock = DockStyle.Fill
            };

            pictureBox.MouseClick += (s, e) =>
            {
                int x = e.X;
                int y = e.Y;
                Console.WriteLine($"🖱️ Coordinate catturate: X={x}, Y={y}");

                var parts = _csvLines[_currentLine].Split(',');
                _csvLines[_currentLine] = $"{parts[0]},{parts[1]},{x},{y}";

                form.Close();
            };

            form.Controls.Add(pictureBox);
            Application.Run(form);
        }
    }
}
