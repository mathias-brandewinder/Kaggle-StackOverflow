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

let trainSampleSet = @"Z:\Data\StackOverflow\train-sample\train-sample.csv"
let publicLeaderboard = @"Z:\Data\StackOverflow\public_leaderboard.csv"

//let trainSampleSet = @"..\..\..\train-sample.csv"
//let publicLeaderboard = @"..\..\..\public_leaderboard.csv"
//let benchmarkSet = @"..\..\..\basic_benchmark.csv"

// split the data into train and test sets as 70/30
let trainPct = 0.05

// Basic function to reduce questions to open vs. close

//let simplified group =
//    match group with
//    | "open" -> "open"
//    | _      -> "closed"

// indices of the title and body columns
let titleCol, bodyCol = 6, 7

// retrieve OpenStatus and given column
let getQuestionsData setFileName col =
    parseCsv setFileName
    |> Seq.skip 1
    |> Seq.map (fun line -> line.[14], line.[col])
    |> Seq.toList

let getPublicData publicLeaderboard= 
    parseCsv publicLeaderboard
    |> Seq.skip 1
    |> Seq.map (fun line -> line.[7])
    |> Seq.toList

let inline size pct len = int(ceil(pct * float len))

// split data into train and test sets
let splitSets fileName trainPct col =
    let questionsData = getQuestionsData fileName col
    let sampleSize = size trainPct questionsData.Length
    questionsData
    |> Seq.fold (fun (i, (sample, test)) q -> 
        if i <= sampleSize then i+1, (q::sample, test)
        else i+1, (sample, q::test)) (1,([],[]))
    |> snd

// dumb classifier, matches every category to prior proba of each class
let priorClassifier category =
    match category with
    | "not a real question" ->
        [ ("not a real question", 0.0091347705760047); ("not constructive", 0.0046458596397953); ("off topic", 0.00520096554605094); ("open", 0.979191390785063); ("too localized", 0.00182701345308509) ]
    | "not constructive" ->
        [ ("not a real question", 0.0091347705760047); ("not constructive", 0.0046458596397953); ("off topic", 0.00520096554605094); ("open", 0.979191390785063); ("too localized", 0.00182701345308509) ]
    | "off topic" ->
        [ ("not a real question", 0.0091347705760047); ("not constructive", 0.0046458596397953); ("off topic", 0.00520096554605094); ("open", 0.979191390785063); ("too localized", 0.00182701345308509) ]
    | "open" ->
        [ ("not a real question", 0.0091347705760047); ("not constructive", 0.0046458596397953); ("off topic", 0.00520096554605094); ("open", 0.979191390785063); ("too localized", 0.00182701345308509) ]
    | "too localized" ->
        [ ("not a real question", 0.0091347705760047); ("not constructive", 0.0046458596397953); ("off topic", 0.00520096554605094); ("open", 0.979191390785063); ("too localized", 0.00182701345308509) ]
    | _ -> failwith "Unrecognized category"

// need to do some work on this to produce probabilities
// based on Bayes classifier output. This is raw, based on
// observed error using classifier
let bodyClassifier category =
    match category with
    | "not a real question" ->
        [ ("not a real question", 0.397); ("not constructive", 0.146); ("off topic", 0.096); ("open", 0.276); ("too localized", 0.086) ]
    | "not constructive" ->
        [ ("not a real question", 0.124); ("not constructive", 0.082); ("off topic", 0.105); ("open", 0.582); ("too localized", 0.108) ]
    | "off topic" ->
        [ ("not a real question", 0.103); ("not constructive", 0.237); ("off topic", 0.483); ("open", 0.158); ("too localized", 0.02) ]
    | "open" ->
        [ ("not a real question", 0.0091347705760047); ("not constructive", 0.0046458596397953); ("off topic", 0.00520096554605094); ("open", 0.979191390785063); ("too localized", 0.00182701345308509) ]
    | "too localized" ->
        [ ("not a real question", 0.162); ("not constructive", 0.042); ("off topic", 0.098); ("open", 0.363); ("too localized", 0.336) ]
    | _ -> failwith "Unrecognized category"

let trainSample trainSet =
    let tokens = topByClass trainSet 500
    let training = train setOfWords trainSet tokens
    classify training
    //todo: return probabilities here

let inline predict model = Seq.map model 

// Visualize classification results by group
let visualizeByGroup test testSet =
    testSet
    |> Seq.map (fun (c, t) -> 
        let result = test t |> Seq.maxBy snd |> fst
        c, result)
    |> Seq.groupBy fst
    |> Seq.map (fun (cl, gr) ->
        let grouped = gr |> Seq.groupBy snd
        cl,
        grouped |> Seq.map (fun (res, cases) -> res, Seq.length cases))
    |> Seq.iter (fun (cl, results) -> 
        printfn ""
        printfn "Real: %s" cl
        results |> Seq.iter (fun (g, c) -> printfn "%s, %i" g c))

// Evaluate % correctly classified

//questionTitles
//|> Seq.skip (sampleSize + 1)
//|> Seq.take 1000
//|> Seq.map (fun (c, t) -> 
//    let result = 
//        test t |> Seq.maxBy snd |> fst
//    c, if result = c then 1.0 else 0.0)
//|> Seq.groupBy fst
//|> Seq.map (fun (cl, gr) -> cl, gr |> Seq.averageBy snd)
//|> Seq.iter (fun (cl, prob) -> printfn "%s %f" cl prob)

let trainSet, testSet = splitSets trainSampleSet trainPct bodyCol
let test = trainSample trainSet
let predictions = predict test (Seq.map snd testSet)

visualizeByGroup test testSet
quality predictions (Seq.map fst testSet)