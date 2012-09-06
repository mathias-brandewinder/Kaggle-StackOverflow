#r "Microsoft.VisualBasic"

#load "Data.fs"
#load "Preprocessing.fs"
#load "Distributions.fs"
#load "Validation.fs"
#load "NaiveBayes.fs"
#load "NumericBayes.fs"
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

// build Bayesian update on undeleted and reputation
let leaderboardData =
    parseCsv publicLeaderboard
    |> Seq.skip 1
    |> Seq.map extractPost 

let reputationKnowledge = variableKnowledge (fun post -> post.Reputation) 20 trainSet 
let undeletedKnowledge = variableKnowledge (fun post -> post.Undeleted) 0 trainSet 
let experienceKnowledge = variableKnowledge (fun post -> post.DaysExperience) 40 trainSet 
let bodylengthKnowledge = variableKnowledge (fun post -> post.Body.Length) 450 trainSet

let reputationModel (post: Charon.Post) = 
    numericModel post (fun post -> post.Reputation) 20 reputationKnowledge trainingPriors
let undeletedModel (post: Charon.Post) =
    numericModel post (fun post -> post.Undeleted) 0 undeletedKnowledge trainingPriors
let experienceModel (post: Charon.Post) = 
    numericModel post (fun post -> post.DaysExperience) 40 experienceKnowledge trainingPriors
let bodylengthModel (post: Charon.Post) = 
    numericModel post (fun post -> post.Body.Length) 450 bodylengthKnowledge trainingPriors


let priorModel = fun (post: Charon.Post) -> trainingPriors


let qstat = getQuestionsByUser trainSet
let qstatModel = fun (post: Charon.Post) -> 
    match qstat.TryFind post.OwnerUserId with
    | Some cats -> cats
    | None -> newUserPriors 

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

// Find best params for numeric bayes models
// best rep: 20
// best undeleted: 0
// best exp: 40
// best body length: 360
for r in 1 .. 1 .. 5 do
    let k = variableKnowledge (fun post -> List.length post.Tags) r trainSet 
    let model (post: Charon.Post) = numericModel post (fun post -> List.length post.Tags) r k trainingPriors
    printfn "Threshold: %i" r
    benchmark model validateSet

// current best model
let comboModel = fun (post: Charon.Post) -> 
    combineMany categories 
                [ (0.18, bodyModel post); 
                  (0.51, titleModel post); 
                  (0.25, tagsModel post); 
                  (0.03, reputationModel post);
                  (0.03, reputationModel post) ]

let inline probsToString (m: Map<_, float>) =
    String.Join(",", categories |> Seq.map (fun c -> string m.[c]))

// save probabilities to a file
let saveProbs model dataset fileName =
    let lines = dataset |> Seq.map (fst >> model >> probsToString)
    System.IO.File.WriteAllLines(fileName, lines)

// Create a file with outputs of all models ("meta")
//let meta (dataset: (Charon.Post * string) list)=
//    dataset
//    |> List.map (fun (post, cl) ->
//        cl, 
//        [ 
//          let body = bodyModel post
//          for c in categories -> body.[c]
//          let title = titleModel post
//          for c in categories -> title.[c]
//          let tags = tagsModel post
//          for c in categories -> tags.[c] ])
//
//let save (meta: (string * float list) list) =
//    let lines = 
//        meta 
//        |> List.map (fun (cl, data) -> cl :: (data |> List.map (fun e -> e.ToString())))
//        |> List.map (fun l -> String.Join(",", l))
//    System.IO.File.WriteAllLines(@"..\..\..\meta.csv", lines)


// combo with questions stat
let comboStatModel = fun (post: Charon.Post) ->
    combineMany categories 
                [ 0.2, bodyModel post
                  0.45, titleModel post
                  0.25, tagsModel post 
                  0.05, qstatModel post
                  0.05, priorModel post ]

benchmark comboStatModel validateSet


let createSVMInput model (dataset: #seq<Charon.Post * _>) fileName =
    let lines = dataset |> Seq.map (fun (p, s) -> 
        String.Join(",", [| probsToString (model p); string p.Reputation; string p.Undeleted; s |]))
    System.IO.File.WriteAllLines(fileName, lines)


createSVMInput comboModel trainSet @"..\..\..\comboModelTrain.csv"
createSVMInput comboModel validateSet @"..\..\..\comboModelValidate.csv"