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
let submissionFile = "REPLACE WITH FILE NAME" // @"..\..\..\submission06.csv"

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

let priorModel = fun (post: Charon.Post) -> priors

printfn "Building model"
let comboModel = fun (post: Charon.Post) -> 
    combineMany categories 
                [ (0.20, bodyModel post); 
                  (0.50, titleModel post); 
                  (0.25, tagsModel post); 
                  (0.05, priorModel post) ]

printfn "Creating submission file"
create publicLeaderboard submissionFile comboModel categories

printfn "Done"