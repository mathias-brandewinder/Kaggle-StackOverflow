#r "Microsoft.VisualBasic"
#load "Data.fs"
#load "Preprocessing.fs"
#load "Distributions.fs"
#load "Validation.fs"
#load "LogisticRegression.fs"
System.IO.Directory.SetCurrentDirectory(__SOURCE_DIRECTORY__)

open System
open System.Text
open Charon.Data
open Charon.Preprocessing
open Charon.Distributions
open Charon.Validation
open MachineLearning.LogisticRegression
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

//(float)post.Undeleted;
let experience (post: Charon.Post) =
    (post.PostDate - post.UserDate).Days

let numbersFrom (post: Charon.Post) =
    [ (float)post.Reputation; 
      (float)post.Undeleted; 
      (float)post.UserDate.Ticks; 
      (float)(experience post); 
      (float)post.Tags.Length; 
      (float)post.Body.Length ] 

let numeric (data: (Charon.Post * string) list) =
    data 
    |> List.filter (fun (post, cat) ->
        (stringToCategory cat) = TooLocal || (stringToCategory cat) = NotReal)
    |> List.map (fun (post, cat) ->
        let label = 
            match (stringToCategory cat) with
            | TooLocal -> 1.0
            | _       -> 0.0
        label, 1.0 :: (numbersFrom post))

let nTraining = numeric trainSet
let nValidate = numeric validateSet
let w = train nTraining 100 0.01
let classifier = classify w

let stats (predictions: (float * float) seq) =     
    let correct =
        predictions 
        |> Seq.map (fun (real, pred) -> 
            real,
            if real = 0.0 && pred < 0.5 || real = 1.0 && pred > 0.5 
            then 1.0 
            else 0.0)
    correct
    |> Seq.averageBy snd
    |> printfn "Overall correctly classified: %f"
    correct
    |> Seq.filter (fun (real, pred) -> real = 1.0)
    |> Seq.averageBy snd
    |> printfn "Group 1 correctly classified: %f"
    correct
    |> Seq.filter (fun (real, pred) -> real = 0.0)
    |> Seq.averageBy snd
    |> printfn "Group 0 correctly classified: %f"

let results = 
    nValidate 
    |> List.map (fun (lab, obs) -> lab, classifier (List.tail obs))
    |> stats
