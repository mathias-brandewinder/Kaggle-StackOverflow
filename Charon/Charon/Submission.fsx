// Script to generate a submission file

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
open Charon.NumericBayes
open MachineLearning.NaiveBayes
open Microsoft.VisualBasic.FileIO

#time

//let publicLeaderboard = @"..\..\..\public_leaderboard.csv"
let publicLeaderboard = @"..\..\..\private_leaderboard.csv"
let submissionFile = @"..\..\..\submission-final.csv" //"REPLACE WITH FILE NAME" //  

// build Bayes on Body
printfn "Building Bayes classifier on Post body"
//let bodyData = readWordsFrequencies @"..\..\..\bayes-body-filtered.csv"
let bodyData = readWordsFrequencies @"..\..\..\body-set.csv"
let bodyTraining = updatePriors bodyData priors
let bodyClassifier = classify bodyTraining 
let bodyModel = fun (post: Charon.Post) -> (bodyClassifier post.Body |> renormalize)

// build Bayes on Title
printfn "Building Bayes classifier on Post title"
//let titleData = readWordsFrequencies @"..\..\..\title-bayes.csv"
let titleData = readWordsFrequencies @"..\..\..\final-title-bayes.csv"
let titleTraining = updatePriors titleData priors
let titleClassifier = classify titleTraining 
let titleModel = fun (post: Charon.Post) -> (titleClassifier post.Title |> renormalize)

// build Bayes on Tags
printfn "Building Bayes classifier on Post tags"
let tagsAsText (post: Charon.Post) = String.Join(" ", post.Tags)
let tagsData = readWordsFrequencies @"..\..\..\final-tags.csv"
let tagsTraining = updatePriors tagsData priors
let tagsClassifier = classify tagsTraining 
let tagsModel = fun (post: Charon.Post) -> (tagsClassifier (tagsAsText post) |> renormalize)

// build reputation model - hardcoded
let reputationLimit = 20 
let reputationKnowledge = 
    Map.empty
       .Add("not a real question", 0.6415765942)
       .Add("not constructive",    0.330241661)
       .Add("off topic",           0.4457373184)
       .Add("open",                0.3551008011)
       .Add("too localized",       0.5505666957)    

let reputationModel (post: Charon.Post) = 
    numericModel post (fun post -> post.Reputation) reputationLimit reputationKnowledge priors

let undeletedLimit = 0
let undeletedKnowledge = 
    Map.empty
       .Add("not a real question", 0.6774319573)
       .Add("not constructive",    0.3567052417)
       .Add("off topic",           0.4787436307)
       .Add("open",                0.3777098599)
       .Add("too localized",       0.5963382738) 
let undeletedModel (post: Charon.Post) =
    numericModel post (fun post -> post.Undeleted) undeletedLimit undeletedKnowledge priors

let experienceLimit = 40
let experienceKnowledge = 
    Map.empty
       .Add("not a real question", 0.544124669)
       .Add("not constructive",    0.3430054459)
       .Add("off topic",           0.370674576)
       .Add("open",                0.3062948479)
       .Add("too localized",       0.4644725371) 
let experienceModel (post: Charon.Post) = 
    numericModel post (fun post -> post.DaysExperience) 40 experienceKnowledge priors

let bodyLimit = 450
let bodylengthKnowledge = 
    Map.empty
       .Add("not a real question", 0.6820332509)
       .Add("not constructive",    0.5576923077)
       .Add("off topic",           0.5742642026)
       .Add("open",                0.3528417935)
       .Add("too localized",       0.3445945946) 
let bodylengthModel (post: Charon.Post) = 
    numericModel post (fun post -> post.Body.Length) 450 bodylengthKnowledge priors

let priorModel = fun (post: Charon.Post) -> priors

printfn "Building model"

// submission 12: 0.18780
//let comboModel = fun (post: Charon.Post) -> 
//    combineMany categories 
//                [ (0.20, bodyModel post); 
//                  (0.50, titleModel post); 
//                  (0.25, tagsModel post); 
//                  (0.01, priorModel post);
//                  (0.02, reputationModel post);
//                  (0.02, reputationModel post) ]

// submission 13: 0.18753
//let comboModel = fun (post: Charon.Post) -> 
//    combineMany categories 
//                [ (0.18, bodyModel post); 
//                  (0.51, titleModel post); 
//                  (0.25, tagsModel post); 
//                  (0.03, reputationModel post);
//                  (0.03, reputationModel post) ]

// Submission 14: 0.18754
//let comboModel = fun (post: Charon.Post) -> 
//    combineMany categories 
//                [ (0.18, bodyModel post); 
//                  (0.51, titleModel post); 
//                  (0.25, tagsModel post); 
//                  (0.02, reputationModel post);
//                  (0.02, reputationModel post);
//                  (0.02, experienceModel post) ]

// Oct 8: 0.18727
//let comboModel = fun (post: Charon.Post) -> 
//    combineMany categories 
//                [ (0.18, bodyModel post); 
//                  (0.51, titleModel post); 
//                  (0.25, tagsModel post); 
//                  (0.03, reputationModel post);
//                  (0.03, reputationModel post) ]

// Oct 8: 
let comboModel = fun (post: Charon.Post) -> 
    combineMany categories 
                [ (0.30, bodyModel post); 
                  (0.45, titleModel post); 
                  (0.21, tagsModel post); 
                  (0.01, reputationModel post);
                  (0.01, undeletedModel post);
                  (0.01, experienceModel post);
                  (0.01, bodylengthModel post) ]


printfn "Creating submission file"
create publicLeaderboard submissionFile comboModel categories

printfn "Done"