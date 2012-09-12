#r "Microsoft.VisualBasic"
#load "Data.fs"
#load "Preprocessing.fs"
#load "Distributions.fs"
#load "Validation.fs"
#load "LogisticRegression.fs"
System.IO.Directory.SetCurrentDirectory(__SOURCE_DIRECTORY__)

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

let numeric (data: (Charon.Post * string) list) =
    data 
    |> List.map (fun (post, cat) ->
        let label = 
            match (simplified cat) with
            | "open" -> 1.0
            | _      -> 0.0
        label, [ 1.0; (float)post.Reputation ])

let nTraining = numeric trainSet
let nValidate = numeric validateSet
let w = train nTraining 1000 0.01
let classifier = classify w

let results = nValidate |> List.map (fun (lab, obs) -> lab, classifier obs) 