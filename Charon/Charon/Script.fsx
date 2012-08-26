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

// retrieve class and title of questions
let bayesTitle =
    parseCsv trainSampleSet
    |> Seq.map (fun line -> line.[14], line.[6])
    |> Seq.toList

let tokens = 
    extractWords bayesTitle
    |> Seq.take 100
    |> Set.ofSeq

printfn "%i" tokens.Count