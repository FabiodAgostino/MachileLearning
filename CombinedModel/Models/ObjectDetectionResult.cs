namespace DetectorModel.Models
{
    public class ObjectDetectionResult
    {
        public bool IsObjectDetected { get; set; }
        public float X { get; set; }
        public float Y { get; set; }
        public float Confidence { get; set; }
        public string ObjectType { get; set; }
        public string ErrorMessage { get; set; }
        public string MarkedImagePath { get; set; }
    }
}
