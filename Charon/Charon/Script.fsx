#r "Microsoft.VisualBasic"
#load "Data.fs"
#load "Validation.fs"
#load "NaiveBayes.fs"
System.IO.Directory.SetCurrentDirectory(__SOURCE_DIRECTORY__)

open Charon.Data
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
    
let modelData = readWordsFrequencies @"..\..\..\bayes.csv"
let training = updatePriors modelData trainingPriors
let classifier = classify training 
let model = fun (post: Charon.Post) -> (classifier post.Body |> renormalize)

benchmark model validateSet

// experimenting: benchmarking linear combinations of prior + model
// manual search for best weight
let w = 0.5 
let m = fun (post: Charon.Post) -> combine categories (w, (classifier post.Body |> renormalize)) (1.0-w, trainingPriors)
benchmark m validateSet

//// Create submission file
// on my dinky laptop, takes < 5 minutes
let leader = updatePriors modelData priors
let leaderClassifier = classify leader
// Submission 0: raw model
//let leaderboardModel = fun (post: Charon.Post) -> (leaderClassifier post.Body |> renormalize)
//create publicLeaderboard @"..\..\..\submission00.csv" leaderboardModel categories
// Submission 1: 50% model, 50% priors
let weight = 0.5
let leaderboardModel = fun (post: Charon.Post) -> 
    combine categories (weight, (leaderClassifier post.Body |> renormalize)) (1.0-weight, priors)
create publicLeaderboard @"..\..\..\submission01.csv" leaderboardModel categories


//// need to do some work on this to produce probabilities
//// based on Bayes classifier output. This is raw, based on
//// observed error using classifier
//let bodyClassifier category =
//    match category with
//    | "not a real question" ->
//        [ ("not a real question", 0.397); ("not constructive", 0.146); ("off topic", 0.096); ("open", 0.276); ("too localized", 0.086) ]
//    | "not constructive" ->
//        [ ("not a real question", 0.124); ("not constructive", 0.082); ("off topic", 0.105); ("open", 0.582); ("too localized", 0.108) ]
//    | "off topic" ->
//        [ ("not a real question", 0.103); ("not constructive", 0.237); ("off topic", 0.483); ("open", 0.158); ("too localized", 0.02) ]
//    | "open" ->
//        [ ("not a real question", 0.0091347705760047); ("not constructive", 0.0046458596397953); ("off topic", 0.00520096554605094); ("open", 0.979191390785063); ("too localized", 0.00182701345308509) ]
//    | "too localized" ->
//        [ ("not a real question", 0.162); ("not constructive", 0.042); ("off topic", 0.098); ("open", 0.363); ("too localized", 0.336) ]
//    | _ -> failwith "Unrecognized category"
