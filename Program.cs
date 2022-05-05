using System.IO;
using System;
using DlibDotNet;
using DlibDotNet.Extensions;
using Dlib = DlibDotNet.Dlib;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace CMP304_AI
{
    // The main program class
    class Program
    {
        

        // The main program entry point
        static void Main(string[] args)
        {

            Console.WriteLine("Emotional Facial Recognition Program - By Dylan Baker");
            Console.WriteLine(" ");
            Console.WriteLine("This application reads emotion on a face. You will need to select the image to use (for example: image.jpg).");
            string cdir = Directory.GetCurrentDirectory();
            Console.WriteLine($"Place the image inside {cdir}");
            Console.Write("Enter the name of the image: ");

            // file paths
            string inputFilePath;
            inputFilePath = Console.ReadLine();


            // Set up Dlib Face Detector
            using (var fd = Dlib.GetFrontalFaceDetector())
            using (var sp = ShapePredictor.Deserialize("shape_predictor_68_face_landmarks.dat"))
            {
                // load input image
                var img = Dlib.LoadImage<RgbPixel>(inputFilePath);

                // find all faces in the image
                var faces = fd.Operator(img);
                // for each face draw over the facial landmarks
                foreach (var face in faces)
                {
                    // find the landmark points for this face
                    var shape = sp.Detect(img, face);

                    // draw the landmark points on the image
                    for (var i = 1; i < shape.Parts; i++)
                    {
                        var point = shape.GetPart((uint)i - 1);
                        var rect = new Rectangle(point);
                        Dlib.DrawRectangle(img, rect, color: new RgbPixel(255, 255, 0), thickness: 4);

                        if (i == 22) Dlib.DrawRectangle(img, rect, color: new RgbPixel(0, 255, 255), thickness: 4);
                        if (i == 23) Dlib.DrawRectangle(img, rect, color: new RgbPixel(0, 255, 255), thickness: 4);
                        if (i == 34) Dlib.DrawRectangle(img, rect, color: new RgbPixel(0, 255, 255), thickness: 4);
                        if (i == 40) Dlib.DrawRectangle(img, rect, color: new RgbPixel(0, 255, 255), thickness: 4);
                        if (i == 43) Dlib.DrawRectangle(img, rect, color: new RgbPixel(0, 255, 255), thickness: 4);
                        if (i == 49) Dlib.DrawRectangle(img, rect, color: new RgbPixel(0, 255, 255), thickness: 4);
                        if (i == 52) Dlib.DrawRectangle(img, rect, color: new RgbPixel(0, 255, 255), thickness: 4);
                        if (i == 55) Dlib.DrawRectangle(img, rect, color: new RgbPixel(0, 255, 255), thickness: 4);
                        if (i == 58) Dlib.DrawRectangle(img, rect, color: new RgbPixel(0, 255, 255), thickness: 4);
                    }

                    //Left Eyebrow
                    var LEyePoint = shape.GetPart(39);
                    var LEyeBrowPoint1 = (shape.GetPart(18) - LEyePoint).Length;
                    var LEyeBrowPoint2 = (shape.GetPart(19) - LEyePoint).Length;
                    var LEyeBrowPoint3 = (shape.GetPart(20) - LEyePoint).Length;
                    var LEyeBrowPoint4 = (shape.GetPart(21) - LEyePoint).Length;

                    LEyeBrowPoint1 /= LEyeBrowPoint4;
                    LEyeBrowPoint2 /= LEyeBrowPoint4;
                    LEyeBrowPoint3 /= LEyeBrowPoint4;
                    LEyeBrowPoint4 /= LEyeBrowPoint4;

                    var LEyeBrowSum = LEyeBrowPoint1 + LEyeBrowPoint2 + LEyeBrowPoint3 + LEyeBrowPoint4;

                    //Right Eyebrow
                    var REyePoint = shape.GetPart(42);
                    var REyeBrowPoint1 = (shape.GetPart(25) - REyePoint).Length;
                    var REyeBrowPoint2 = (shape.GetPart(24) - REyePoint).Length;
                    var REyeBrowPoint3 = (shape.GetPart(23) - REyePoint).Length;
                    var REyeBrowPoint4 = (shape.GetPart(22) - REyePoint).Length;

                    REyeBrowPoint1 /= REyeBrowPoint4;
                    REyeBrowPoint2 /= REyeBrowPoint4;
                    REyeBrowPoint3 /= REyeBrowPoint4;
                    REyeBrowPoint4 /= REyeBrowPoint4;

                    var REyeBrowSum = REyeBrowPoint1 + REyeBrowPoint2 + REyeBrowPoint3 + REyeBrowPoint4;

                    //Left Lip
                    var LLip = shape.GetPart(33);
                    var LLip1 = (shape.GetPart(48) - LLip).Length;
                    var LLip2 = (shape.GetPart(49) - LLip).Length;
                    var LLip3 = (shape.GetPart(50) - LLip).Length;
                    var LLip4 = (shape.GetPart(51) - LLip).Length;

                    LLip1 /= LLip4;
                    LLip2 /= LLip4;
                    LLip3 /= LLip4;

                    var LLipSum = LLip1 + LLip2 + LLip3;


                    //Right Lip
                    var RLip = shape.GetPart(33);
                    var RLip1 = (shape.GetPart(54) - RLip).Length;
                    var RLip2 = (shape.GetPart(53) - RLip).Length;
                    var RLip3 = (shape.GetPart(52) - RLip).Length;
                    var RLip4 = (shape.GetPart(51) - RLip).Length;

                    RLip1 /= RLip4;
                    RLip2 /= RLip4;
                    RLip3 /= RLip4;

                    var RLipSum = RLip1 + RLip2 + RLip3;

                    //Lip Height
                    var LipHeight1 = (shape.GetPart(51) - shape.GetPart(57)).Length;
                    var LipHeight2 = (shape.GetPart(33) - shape.GetPart(51)).Length;
                    var LipHeightSum = LipHeight1 /= LipHeight2;

                    //Lip Width
                    var LipWidth1 = (shape.GetPart(48) - shape.GetPart(54)).Length;
                    var LipWidth2 = (shape.GetPart(33) - shape.GetPart(51)).Length;
                    var LipWidthSum = LipWidth1 /= LipWidth2;

                    //Do Prediction
                    string msg = classifier(LEyeBrowSum,REyeBrowSum,LLipSum,RLipSum,LipHeightSum,LipWidthSum);
                    Console.WriteLine($"Emotion: {msg}");


                    using (System.IO.StreamWriter file = new System.IO.StreamWriter(@"data.csv", true))
                    {
                        file.WriteLine(LEyeBrowSum + "," + REyeBrowSum + "," + LLipSum + "," + RLipSum + "," + LipHeightSum + "," + LipWidthSum + "," + msg);
                    }
                }
                // export the modified image
                Dlib.SaveJpeg(img, "output.jpg");
            }
        }

        public static string classifier(double LeftEyebrow1, double RightEyebrow1, double LeftLip1, double RightLip1, double LipHeight1, double LipWidth1)
        {
            string Path = @"data.csv";

            var mlContext = new MLContext();

            IDataView dataView = mlContext.Data.LoadFromTextFile<Input>(Path, hasHeader: true, separatorChar: ',');

            var featureVector = "Features";

            var labelColumn = "Label";

            var pipeline = 
                mlContext.Transforms.Conversion
                .MapValueToKey(inputColumnName: "Emotion", outputColumnName: labelColumn)
                .Append(mlContext.Transforms.Concatenate(featureVector, "LeftEyebrow", "RightEyebrow", "LeftLip", "RightLip", "LipHeight", "LipWidth"))
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumn, featureVector))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var model = pipeline.Fit(dataView);

            var metrics = mlContext.MulticlassClassification.Evaluate(model.Transform(dataView));

            Console.WriteLine(" ");
            Console.WriteLine($"Metrics for Multi-class Classification model - Results Data");
            Console.WriteLine($"MicroAccuracy:    {metrics.MicroAccuracy:0.###}");
            Console.WriteLine($"MacroAccuracy:    {metrics.MacroAccuracy:0.###}");
            Console.WriteLine($"LogLoss:          {metrics.LogLoss:#.###}");
            Console.WriteLine($"LogLossReduction: {metrics.LogLossReduction:#.###}");


            using (var fileStream = new FileStream("model.zip", FileMode.Create, FileAccess.Write))
            {
                mlContext.Model.Save(model, dataView.Schema, fileStream);
            }

            var predictor = mlContext.Model.CreatePredictionEngine<Input, Output>(model);
            var prediction = predictor.Predict(
                new Input()
                {
                    LeftEyebrow = (float)LeftEyebrow1,
                    RightEyebrow = (float)RightEyebrow1,
                    LeftLip = (float)LeftLip1,
                    RightLip = (float)RightLip1,
                    LipHeight = (float)LipHeight1,
                    LipWidth = (float)LipWidth1
                });

            string msg = prediction.Emotion.ToString();
            return msg;
        }

        public class Input
        {
            [LoadColumn(0)]
            public float LeftEyebrow { get; set; }

            [LoadColumn(1)]
            public float RightEyebrow { get; set; }

            [LoadColumn(2)]
            public float LeftLip { get; set; }

            [LoadColumn(3)]
            public float RightLip { get; set; }

            [LoadColumn(4)]
            public float LipHeight { get; set; }

            [LoadColumn(5)]
            public float LipWidth { get; set; }

            [LoadColumn(6)]
            public string Emotion { get; set; }

        }

        public class Output
        {
            [ColumnName("PredictedLabel")]
            public string Emotion { get; set; }

            [ColumnName("Score")]
            public float[] Scores { get; set; }

        }
    }
}
