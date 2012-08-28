#r "Microsoft.VisualBasic"
#load "Data.fs"
#load "Validation.fs"
#load "NaiveBayes.fs"
System.IO.Directory.SetCurrentDirectory(__SOURCE_DIRECTORY__)

open Charon.Data
open Charon.Validation
open MachineLearning.NaiveBayes
open Microsoft.VisualBasic.FileIO

#time

let trainSampleSet = @"Z:\Data\StackOverflow\train-sample\train-sample.csv"
let publicLeaderboard = @"Z:\Data\StackOverflow\public_leaderboard.csv"

//let trainSampleSet = @"..\..\..\train-sample.csv"
//let publicLeaderboard = @"..\..\..\public_leaderboard.csv"
//let benchmarkSet = @"..\..\..\basic_benchmark.csv"

// split the data into train and test sets as 70/30
let trainPct = 0.05

// Basic function to reduce questions to open vs. close

//let simplified group =
//    match group with
//    | "open" -> "open"
//    | _      -> "closed"

// indices of the title and body columns
let titleCol, bodyCol = 6, 7

// retrieve OpenStatus and given column
let getQuestionsData setFileName col =
    parseCsv setFileName
    |> Seq.skip 1
    |> Seq.map (fun line -> line.[14], line.[col])
    |> Seq.toList

let getPublicData publicLeaderboard= 
    parseCsv publicLeaderboard
    |> Seq.skip 1
    |> Seq.map (fun line -> line.[7])
    |> Seq.toList

let inline size pct len = int(ceil(pct * float len))

// split data into train and test sets
let splitSets fileName trainPct col =
    let questionsData = getQuestionsData fileName col
    let sampleSize = size trainPct questionsData.Length
    questionsData
    |> Seq.fold (fun (i, (sample, test)) q -> 
        if i <= sampleSize then i+1, (q::sample, test)
        else i+1, (sample, q::test)) (1,([],[]))
    |> snd

let trainSample trainSet =
    let tokens = topByClass trainSet 500
    let training = train setOfWords trainSet tokens
    classify training
    //todo: return probabilities here

let inline predict model = Seq.map model 

// Visualize classification results by group
let visualizeByGroup test testSet =
    testSet
    |> Seq.map (fun (c, t) -> 
        let result = test t |> Seq.maxBy snd |> fst
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

let trainSet, testSet = splitSets trainSampleSet trainPct bodyCol
let test = trainSample trainSet
let predictions = predict test (Seq.map snd testSet)

visualizeByGroup test testSet
quality predictions (Seq.map fst testSet)

// metrics example
let ys = [1;1;1;2;2;2]
let preds = [
    [1, 0.5; 2, 0.5]
    [1, 0.1; 2, 0.9]
    [1, 0.01;2, 0.99]
    [1, 0.9; 2, 0.1]
    [1, 0.75; 2, 0.25]
    [1, 0.001; 2, 0.999]
    [1, 1.; 2, 0.]
]

quality preds ys