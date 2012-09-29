#r "Microsoft.VisualBasic"
#load "Data.fs"
#load "Preprocessing.fs"
#load "Distributions.fs"
#load "Validation.fs"
#load "NaiveBayes.fs"
#load "NumericBayes.fs"
#load "LogisticRegression.fs"
System.IO.Directory.SetCurrentDirectory(__SOURCE_DIRECTORY__)

open System
open System.Text
open Charon.Data
open Charon.Preprocessing
open Charon.Distributions
open Charon.Validation
open MachineLearning.NaiveBayes
open Microsoft.VisualBasic.FileIO
open Charon.NumericBayes
open MachineLearning.LogisticRegression
#time

let trainSampleSet = @"..\..\..\train-sample.csv"
let publicLeaderboard = @"..\..\..\public_leaderboard.csv"
let metaSet = @"..\..\..\meta.csv"

// split the data into train and test sets as 75/25
let trainPct = 0.75
let inline size pct len = int(ceil(pct * float len))

let split dataset percentage =
    let sampleSize = size percentage (Seq.length dataset)
    dataset
    |> Seq.fold (fun (i, (sample, test)) q -> 
        if i <= sampleSize 
        then i + 1, (q::sample, test)
        else i + 1, (sample, q::test)) (1,([],[]))
    |> snd

let sample = 
    parseCsv trainSampleSet
    |> Seq.skip 1
    |> Seq.map (fun line ->
        extractPost line,
        if stringToCategory line.[14] = Category.TooLocal then "TooLocal" else "Rest")
    |> Seq.toList
    
//let trainSet, validateSet = split sample trainPct
//let trainSet = trainSet |> List.map (fun (p, c) -> c, p.Body)
//
//let knowledge =
//    let tokens = topByClass trainSet 500
//    train setOfWords trainSet tokens
//
//let classifier = classify knowledge
//
//let model = fun (text:string) -> (classifier text |> renormalize)

let metaSample = 
    parseCsv metaSet
    |> Seq.map (fun line ->
        (if stringToCategory line.[0] = Category.TooLocal then 1.0 else 0.0),
        [ for i in 1 .. 15 -> Convert.ToDouble(line.[i]) ] )
    |> Seq.toList

let trainSet, validateSet = split metaSample trainPct
let weights = simpleTrain trainSet 100 0.001
let check = 
    validateSet 
    |> Seq.map (fun (c, o) -> c, predict weights o)
    |> Seq.map (fun (c, v) -> if abs (c - v) < 0.5 then 1.0 else 0.0)
    |> Seq.average

let checkTooLoc = 
    validateSet
    |> Seq.filter (fun (c, o) -> c = 1.0)
    |> Seq.map (fun (c, o) -> c, predict weights o)
    |> Seq.map (fun (c, v) -> if abs (c - v) < 0.5 then 1.0 else 0.0)
    |> Seq.average
