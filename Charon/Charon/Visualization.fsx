// Grab FsharpChart from here http://code.msdn.microsoft.com/windowsdesktop/FSharpChart-b59073f5

#r "Microsoft.VisualBasic"
#load "Data.fs"
#load @"..\..\..\FSharpChart.fsx"
System.IO.Directory.SetCurrentDirectory(__SOURCE_DIRECTORY__)

open System
open System.Text
open Charon.Data

open Microsoft.VisualBasic.FileIO
open MSDN.FSharp.Charting

let display (dataSet: float[][]) (labels: string []) i j =

    let byLabel =
        dataSet
        |> Array.map (fun e -> e.[i], e.[j])
        |> Array.zip labels

    let uniqueLabels = Seq.distinct labels

    FSharpChart.Combine 
        [ for label in uniqueLabels ->
                let data = 
                    Array.filter (fun e -> label = fst e) byLabel
                    |> Array.map snd
                FSharpChart.Point(data) :> ChartTypes.GenericChart
                |> FSharpChart.WithSeries.Marker(Size=3)
                //|> FSharpChart.WithSeries.DataPoint(Label=label)
        ]
    |> FSharpChart.Create

let trainSampleSet = @"..\..\..\train-sample.csv"
let sample = 
    parseCsv trainSampleSet
    |> Seq.skip 1
    |> Seq.map (fun line ->
        extractPost line,
        line.[14])
    |> Seq.take 1000
    |> Seq.toList

let experience (post: Charon.Post) =
    (post.PostDate - post.UserDate).Days

let numbersFrom (post: Charon.Post) =
    [ (float)post.Reputation; 
      (float)post.Undeleted; 
      (float)post.UserDate.Ticks; 
      (float)(experience post); 
      (float)post.Tags.Length; 
      (float)post.Body.Length ] 

let data = sample |> List.map (fun (post, cat) -> List.toArray(numbersFrom post)) |> List.toArray
let labels = sample |> List.map snd |> List.toArray

display data labels 2 5