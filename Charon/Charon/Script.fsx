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
let subsetSize = 20000

// retrieve class and title of questions
let questionTitles =
    parseCsv trainSampleSet
    |> Seq.skip 1
    |> Seq.take subsetSize
    |> Seq.map (fun line -> line.[14], line.[6])
    |> Seq.toList

let sampleSize = 10000
let sample = questionTitles |> Seq.take sampleSize
//let tokens = topWords questionTitles |> Seq.take 500
let tokens = topByClass sample 500

let training = train setOfWords questionTitles tokens
let test = classify training

//questionTitles
//|> Seq.skip (sampleSize + 1)
//|> Seq.take 20
//|> Seq.iter (fun (c, t) -> 
//    let result = test t
//    printfn ""
//    printfn "Real: %s" c
//    result |> List.iter (fun (c, p) -> printfn "%s %f" c p))

questionTitles
|> Seq.skip (sampleSize + 1)
|> Seq.take 1000
|> Seq.map (fun (c, t) -> 
    let result = 
        test t |> Seq.maxBy snd |> fst
    c, if result = c then 1.0 else 0.0)
|> Seq.groupBy fst
|> Seq.map (fun (cl, gr) -> cl, gr |> Seq.averageBy snd)
|> Seq.iter (fun (cl, prob) -> printfn "%s %f" cl prob)
