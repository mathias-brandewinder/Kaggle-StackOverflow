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

#time

let trainSampleSet = @"..\..\..\train-sample.csv"
let publicLeaderboard = @"..\..\..\public_leaderboard.csv"

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
        line.[14])
    |> Seq.toList
    
let trainSet, validateSet = split sample trainPct

// build Bayes on Body
printfn "Building Bayes classifier on Post body"
let bodyData = readWordsFrequencies @"..\..\..\bayes-body-filtered.csv"
let bodyTraining = updatePriors bodyData trainingPriors
let bodyClassifier = classify bodyTraining 
let bodyModel = fun (post: Charon.Post) -> (bodyClassifier post.Body |> renormalize)

// build Bayes on Title
printfn "Building Bayes classifier on Post title"
let titleData = readWordsFrequencies @"..\..\..\title-bayes.csv"
let titleTraining = updatePriors titleData trainingPriors
let titleClassifier = classify titleTraining 
let titleModel = fun (post: Charon.Post) -> (titleClassifier post.Title |> renormalize)

let priorModel = fun (post: Charon.Post) -> trainingPriors

   
// search for good weights between models
// typically working on small dataset first to get a sense of right params 
let testSet = validateSet |> Seq.take 100
for bodyW in 0.275 .. 0.025 .. 0.325 do
    let m = max 0.7 (min 0.7 (1.0 - bodyW))
    for titleW in 0.6 .. 0.025 .. m do
        printfn "Combination: Body %f, Title %f" bodyW titleW
        let restW = 1.0 - bodyW - titleW
        let mixModel = fun (post: Charon.Post) ->
            combineMany categories [ (bodyW, bodyModel post); (titleW, titleModel post); (restW, priorModel post) ]
        benchmark mixModel testSet

// current best model: 30% bayes body, 60% bayes title, 10% raw priors
let comboModel = fun (post: Charon.Post) -> combineMany categories [ (0.3, bodyModel post); (0.6, titleModel post); (0.1, priorModel post) ]

//// Create submission file

//// build Bayes on Body
//printfn "Building Bayes classifier on Post body"
//let bodyData = readWordsFrequencies @"..\..\..\bayes-body-filtered.csv"
//let bodyTraining = updatePriors bodyData priors
//let bodyClassifier = classify bodyTraining 
//let bodyModel = fun (post: Charon.Post) -> (bodyClassifier post.Body |> renormalize)
//
//// build Bayes on Title
//printfn "Building Bayes classifier on Post title"
//let titleData = readWordsFrequencies @"..\..\..\title-bayes.csv"
//let titleTraining = updatePriors titleData priors
//let titleClassifier = classify titleTraining 
//let titleModel = fun (post: Charon.Post) -> (titleClassifier post.Title |> renormalize)
//
//let priorModel = fun (post: Charon.Post) -> priors
//
//let comboModel = fun (post: Charon.Post) -> combineMany categories [ (0.3, bodyModel post); (0.6, titleModel post); (0.1, priorModel post) ]
//
//let submissionFile = "TYPE FILE NAME HERE" //@"..\..\..\submission00.csv"
//create publicLeaderboard submissionFile comboModel categories

