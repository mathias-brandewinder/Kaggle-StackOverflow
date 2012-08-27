#r "Microsoft.VisualBasic"
open Microsoft.VisualBasic.FileIO
#load "Data.fs"
#load "NaiveBayes.fs"
open Charon.Data
open MachineLearning.NaiveBayes

#time
//let trainSet =  @"Z:\Data\StackOverflow\train\train.csv"
let trainSampleSet = @"Z:\Data\StackOverflow\train-sample\train-sample.csv"
let benchmarkSet = @"Z:\Data\StackOverflow\public_leaderboard.csv"

// Basic function to reduce questions to open vs. close

//let simplified group =
//    match group with
//    | "open" -> "open"
//    | _      -> "closed"

// retrieve class and title of questions
//let questionTitles =
//    parseCsv trainSampleSet
//    |> Seq.skip 1
//    |> Seq.map (fun line -> line.[14], line.[6])
//    |> Seq.toList

let questionBodies =
    parseCsv trainSampleSet
    |> Seq.skip 1
    |> Seq.map (fun line -> line.[14], line.[7])
    |> Seq.toList

let sampleSize = 10000
let sample = questionBodies |> Seq.take sampleSize
let tokens = topByClass sample 500

let training = train setOfWords questionBodies tokens
let test = classify training

// Visualize classification results

//questionTitles
//|> Seq.skip (sampleSize + 1)
//|> Seq.take 20
//|> Seq.iter (fun (c, t) -> 
//    let result = test t
//    printfn ""
//    printfn "Real: %s" c
//    result |> List.iter (fun (c, p) -> printfn "%s %f" c p))

// Evaluate % correctly classified

//questionTitles
//|> Seq.skip (sampleSize + 1)
//|> Seq.take 1000
//|> Seq.map (fun (c, t) -> 
//    let result = 
//        test t |> Seq.maxBy snd |> fst
//    c, if result = c then 1.0 else 0.0)
//|> Seq.groupBy fst
//|> Seq.map (fun (cl, gr) -> cl, gr |> Seq.averageBy snd)
//|> Seq.iter (fun (cl, prob) -> printfn "%s %f" cl prob)

// Visualize classification results by group
questionBodies
|> Seq.skip (sampleSize + 1)
|> Seq.take 10000
|> Seq.map (fun (c, t) -> 
    let result = 
        test t |> Seq.maxBy snd |> fst
    c, result)
|> Seq.groupBy fst
|> Seq.map (fun (cl, gr) -> 
    let grouped = gr |> Seq.groupBy snd
    cl,
    grouped |> Seq.map (fun (res, cases) -> res, Seq.length cases))
|> Seq.iter (fun (cl, results) -> 
    printfn ""
    printfn "Real: %s" cl
    results |> Seq.iter (fun (g, c) -> printfn "%s, %i" g c))