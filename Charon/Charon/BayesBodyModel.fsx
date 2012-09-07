#r "Microsoft.VisualBasic"
#load "Data.fs"
#load "Preprocessing.fs"
#load "Distributions.fs"
#load "Validation.fs"
#load "NaiveBayes.fs"
System.IO.Directory.SetCurrentDirectory(__SOURCE_DIRECTORY__)

open Charon.Data
open Charon.Preprocessing
open Charon.Distributions
open Charon.Validation
open MachineLearning.NaiveBayes
open Microsoft.VisualBasic.FileIO

let trainSampleSet = @"..\..\..\train-sample.csv"

// split the data into train and test sets as 75/25
let trainPct = 0.75

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

// Model estimation

printfn "Reading data sets"
let trainSet, testSet = splitSets trainSampleSet trainPct bodyCol

printfn "Training model on training set"

let knowledge =
    let tokens = topByClass trainSet 500
    train setOfWords trainSet tokens

let classifier = classify knowledge

let model = fun (text:string) -> (classifier text |> renormalize)

// Model persistence

printfn "Saving"
saveWordsFrequencies @"..\..\..\bayes.csv" knowledge

// Reading back model

printfn "Reading"
let back = readWordsFrequencies @"..\..\..\bayes.csv"
// use priors from training set
let trainingModel = updatePriors back trainingPriors
// use priors from overall dataset
let overallModel = updatePriors back priors