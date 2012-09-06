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

// build Bayesian update on undeleted and reputation
let leaderboardData =
    parseCsv publicLeaderboard
    |> Seq.skip 1
    |> Seq.map extractPost 

let medianReputation =
    leaderboardData 
    |> Seq.map (fun p -> p.Reputation)
    |> fractile 0.5
let medianUndeleted =
    leaderboardData 
    |> Seq.map (fun p -> p.Undeleted)
    |> fractile 0.5

let reputationKnowledge =
    trainSet 
    |> Seq.map (fun (post, label) -> post.Reputation, label)
    |> Seq.groupBy snd
    |> Seq.map (fun (label, data) ->
        let total = Seq.length data
        let below = data |> Seq.filter (fun e -> fst e <= medianReputation) |> Seq.length
        label, (float)below / (float)total)
    |> Map.ofSeq 
let undeletedKnowledge =
    trainSet 
    |> Seq.map (fun (post, label) -> post.Undeleted, label)
    |> Seq.groupBy snd
    |> Seq.map (fun (label, data) ->
        let total = Seq.length data
        let below = data |> Seq.filter (fun e -> fst e <= medianUndeleted) |> Seq.length
        label, (float)below / (float)total)
    |> Map.ofSeq 

let reputationModel = fun (post: Charon.Post) -> 
    let estimates = 
        if post.Reputation <= medianReputation
        then reputationKnowledge |> Map.map (fun k v -> v * trainingPriors.[k])
        else reputationKnowledge |> Map.map (fun k v -> (1.0 - v) * trainingPriors.[k])
    let total = estimates |> Map.fold (fun acc k v -> acc + v) 0.0
    estimates |> Map.map (fun k v -> v / total)

let undeletedModel = fun (post: Charon.Post) -> 
    let estimates = 
        if post.Undeleted <= medianUndeleted
        then undeletedKnowledge |> Map.map (fun k v -> v * trainingPriors.[k])
        else undeletedKnowledge |> Map.map (fun k v -> (1.0 - v) * trainingPriors.[k])
    let total = estimates |> Map.fold (fun acc k v -> acc + v) 0.0
    estimates |> Map.map (fun k v -> v / total)
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

// current best model
let comboModel = fun (post: Charon.Post) -> 
    combineMany categories 
                [ (0.20, bodyModel post); 
                  (0.50, titleModel post); 
                  (0.25, tagsModel post); 
                  (0.05, priorModel post) ]

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

// Experimenting with tag pairs
let tagMatch (tags: string list) (token: string * string) =
    List.exists (fun t -> 
        t = fst token) tags 
        && List.exists (fun t -> t = snd token) tags

let tagPairs (tags: string list) =
    let rec extract (current: (string * string) list) (tokens: string list) =
        match tokens with
        | [] -> current
        | [_] -> current
        | hd :: tl ->
            let combos = tl |> List.map (fun t -> (hd, t))
            let newCurrent = List.append current combos
            extract newCurrent tl
    tags |> List.sort |> extract []

let tagPairsCount data =
    let update (acc: Map<(string * string), int>) (tags: string list) = 
        tags
        |> tagPairs
        |> Seq.fold (fun (a: Map<(string * string), int>) t -> 
            if a.ContainsKey(t) 
            then a.Add(t, a.[t] + 1) 
            else a.Add(t, 1)) acc
    data
    |> Seq.fold (fun acc post -> 
        update acc post) Map.empty 

let topTokenPairs data min =
    data
    |> tagPairsCount
    |> Map.filter (fun k v -> v >= min)

let tokenPairs (dataset: (Charon.Post * string) list) =
    dataset
    |> List.map (fun (post, l) -> tagPairs post.Tags)
    |> List.concat
    |> Set.ofList

let pairsCount (data: (string * Charon.Post) seq) (words: Set<(string * string)>) =
    data
    |> Seq.map (fun (label, sample) -> sample.Tags)
    |> tagPairsCount
    |> Map.map (fun k v -> v + 1)

let tokensTrain (dataset: (Charon.Post * string) list) words =
    let size = Seq.length dataset
    dataset
    |> Seq.map (fun (post, label) -> label, post)
    |> Seq.groupBy fst
    |> Seq.map (fun (label, data) -> 
        label, Seq.length data, pairsCount data words)
    |> Seq.map (fun (label, total, tokenCount) ->
        let totTokens = Map.fold (fun total key value -> total + value) 0 tokenCount
        label, 
        prop(total, size), 
        Map.map (fun token count -> prop(count, totTokens)) tokenCount)
    |> Seq.toList

let tokensClassify (estimator: (string * float * Map<(string * string), float>) seq) (post: Charon.Post) =
    let pairs = post.Tags |> tagPairs     
    estimator
    |> Seq.map (fun (label, proba, tokens) ->
            label,
            pairs
            |> List.fold (fun p token ->
                if tokens.ContainsKey token
                then p + log(tokens.[token])
                else p) (log proba))
        |> Seq.toList

let topPairsByClass (dataset: (Charon.Post * string) seq) min =
        dataset
        |> Seq.map (fun (post, lbl) -> lbl, post.Tags)
        |> Seq.groupBy fst
        |> Seq.map (fun (lbl, group) -> 
            topTokenPairs (group |> Seq.map snd) min
            |> Map.toSeq
            |> Seq.map fst
            |> Set.ofSeq)
        |> Set.unionMany

let smallTrainSet = trainSet |> Seq.take 10000 |> Seq.toList
let tokens = topPairsByClass trainSet 3
let knowledge = tokensTrain smallTrainSet tokens
let tokensClassifier = tokensClassify knowledge
let tokensModel = fun (post: Charon.Post) -> (tokensClassifier post |> renormalize)



for w in 0.2 .. 0.1 .. 0.6 do
    let testModel = fun (post: Charon.Post) -> 
        combineMany categories 
                    [ (w * 0.20, bodyModel post); 
                      (w * 0.50, titleModel post); 
                      (w * 0.25, tagsModel post);
                      (1.0 - w, reputationModel post); 
                      (w * 0.05, priorModel post) ]
    printfn "%f original" w                 
    benchmark testModel smallSet


// combo with questions stat
let comboStatModel = fun (post: Charon.Post) ->
    combineMany categories 
                [ 0.2, bodyModel post
                  0.45, titleModel post
                  0.25, tagsModel post 
                  0.05, qstatModel post
                  0.05, priorModel post ]

benchmark comboStatModel validateSet
