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
let subsetSize = 10000

// retrieve class and title of questions
let questionTitles =
    parseCsv trainSampleSet
    |> Seq.skip 1
    |> Seq.take subsetSize
    |> Seq.map (fun line -> line.[14], line.[6])
    |> Seq.toList

let sampleSize = 5000
let sample = questionTitles |> Seq.take sampleSize
let tokens = topWords questionTitles |> Seq.take 500

//let estimator = train bagOfWords questionTitles tokens
//let test = classify estimator
let test = classifier setOfWords questionTitles tokens

questionTitles
|> Seq.skip (sampleSize + 1)
|> Seq.take 20
|> Seq.iter (fun (c, t) -> 
    let result = test t
    printfn ""
    printfn "Real: %s" c
    result |> List.iter (fun (c, p) -> printfn "%s %f" c p))