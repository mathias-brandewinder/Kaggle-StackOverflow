#r "Microsoft.VisualBasic"
#load "Data.fs"
#load "Preprocessing.fs"
#load "Distributions.fs"
#load "Validation.fs"
#load "NaiveBayes.fs"
System.IO.Directory.SetCurrentDirectory(__SOURCE_DIRECTORY__)

open System
open System.Text
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

// build Bayes on Tags
printfn "Building Bayes classifier on Post tags"
let tagsAsText (post: Charon.Post) = String.Join(" ", post.Tags)
let tagsData = readWordsFrequencies @"..\..\..\bayes-tags.csv"
let tagsTraining = updatePriors tagsData trainingPriors
let tagsClassifier = classify tagsTraining 
let tagsModel = fun (post: Charon.Post) -> (tagsClassifier (tagsAsText post) |> renormalize)

let priorModel = fun (post: Charon.Post) -> trainingPriors

   
// search for good weights between models
// typically working on small dataset first to get a sense of right params 
let testSet = validateSet |> Seq.take 100
for bodyW in 0.1 .. 0.05 .. 0.5 do
//    let m = max 0.7 (min 0.7 (1.0 - bodyW))
    for titleW in 0.4 .. 0.05 .. 1.0 - bodyW do
        for tagsW in 0.0 .. 0.05 .. 1.0 - bodyW - titleW do
            printfn "Combination: Body %f, Title %f, Tags %f" bodyW titleW tagsW
            let restW = 1.0 - bodyW - titleW - tagsW
            let mixModel = fun (post: Charon.Post) ->
                combineMany categories 
                            [ (bodyW, bodyModel post); 
                              (titleW, titleModel post); 
                              (tagsW, tagsModel post);
                              (restW, priorModel post) ]
            benchmark mixModel testSet

// current best model
let comboModel = fun (post: Charon.Post) -> 
    combineMany categories 
                [ (0.20, bodyModel post); 
                  (0.50, titleModel post); 
                  (0.25, tagsModel post); 
                  (0.05, priorModel post) ]

let inline probsToString (m: Map<_, float>) =
    String.Join(",", categories |> Seq.map (fun c -> string m.[c]) |> Seq.toArray)

// save probabilities to a file
let saveProbs model dataset fileName =
    let lines = 
        dataset
        |> Seq.map (fst >> model >> probsToString)
        |> Seq.append [ String.Join(",", Array.ofList categories) ]
    System.IO.File.WriteAllLines(fileName, lines)
