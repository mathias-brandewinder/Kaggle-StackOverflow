#r "Microsoft.VisualBasic"
#load "Data.fs"
#load "Preprocessing.fs"
#load "Distributions.fs"
#load "Validation.fs"
#load "NaiveBayes.fs"
#load "NumericBayes.fs"
#load "DecisionTrees.fs"
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
open MachineLearning.DecisionTrees
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
        let post = extractPost line
        let cat = line.[14]
        [| 
           (if (post.Reputation <= 20) then "Lo" else "Hi");
           (if (post.Undeleted <= 0) then "Hi" else "Lo");
           (if (post.DaysExperience <= 40) then "Lo" else "Hi");
           (if (post.Body.Length <= 450) then "Lo" else "Hi");
           cat
        |] )
    |> Seq.toArray
    
let trainSet, validateSet = split sample trainPct
let headers = [| "Rep"; "Que"; "Exp"; "Bod"; "Cla" |]
