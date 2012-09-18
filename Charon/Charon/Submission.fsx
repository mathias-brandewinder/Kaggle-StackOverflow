// Script to generate a submission file

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

let publicLeaderboard = @"..\..\..\public_leaderboard.csv"
let submissionFile = "REPLACE WITH FILE NAME" // @"..\..\..\submission11.csv" // 

// build Bayes on Body
printfn "Building Bayes classifier on Post body"
let bodyData = readWordsFrequencies @"..\..\..\bayes-body-filtered.csv"
let bodyTraining = updatePriors bodyData priors
let bodyClassifier = classify bodyTraining 
let bodyModel = fun (post: Charon.Post) -> (bodyClassifier post.Body |> renormalize)

// build Bayes on Title
printfn "Building Bayes classifier on Post title"
let titleData = readWordsFrequencies @"..\..\..\title-bayes.csv"
let titleTraining = updatePriors titleData priors
let titleClassifier = classify titleTraining 
let titleModel = fun (post: Charon.Post) -> (titleClassifier post.Title |> renormalize)

// build Bayes on Tags
printfn "Building Bayes classifier on Post tags"
let tagsAsText (post: Charon.Post) = String.Join(" ", post.Tags)
let tagsData = readWordsFrequencies @"..\..\..\bayes-tags.csv"
let tagsTraining = updatePriors tagsData priors
let tagsClassifier = classify tagsTraining 
let tagsModel = fun (post: Charon.Post) -> (tagsClassifier (tagsAsText post) |> renormalize)

// build reputation model - hardcoded
let median = 42 // measured on leaderboard
// P (rep <= median | class), measured on train set
let reputationKnowledge = 
    Map.empty
       .Add("not a real question", 0.7249207796)
       .Add("not constructive",    0.411334241)
       .Add("off topic",           0.537683474)
       .Add("open",                0.4441702418)
       .Add("too localized",       0.6427637315)    

let reputationModel = fun (post: Charon.Post) -> 
    let estimates = 
        if post.Reputation <= median
        then reputationKnowledge |> Map.map (fun k v -> v * priors.[k])
        else reputationKnowledge |> Map.map (fun k v -> (1.0 - v) * priors.[k])
    let total = estimates |> Map.fold (fun acc k v -> acc + v) 0.0
    estimates |> Map.map (fun k v -> v / total)

let priorModel = fun (post: Charon.Post) -> priors

printfn "Building model"
let comboModel = fun (post: Charon.Post) -> 
    combineMany categories 
                [ (0.10, bodyModel post); 
                  (0.25, titleModel post); 
                  (0.125, tagsModel post); 
                  (0.025, priorModel post);
                  (0.50, reputationModel post) ]

printfn "Creating submission file"
create publicLeaderboard submissionFile reputationModel categories

printfn "Done"