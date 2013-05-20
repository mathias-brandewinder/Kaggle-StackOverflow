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

let tokensFrom (dataset: seq<Charon.Post*string>) folder =
        let words = 
            dataset 
            |> Seq.map (fun (post, label) -> post.Body) 
            |> extractWords
            |> filterStopWords
        let init = 
            words 
            |> Seq.map (fun w -> (w, 0))
            |> Map.ofSeq
        dataset
        |> Seq.map (fun (post, label) -> (label, post.Body))
        |> prepare
        |> Seq.fold (fun state (label, sample) -> folder state sample) init

let tokensFromNoCode (dataset: seq<Charon.Post*string>) folder =
        let words = 
            dataset 
            |> Seq.map (fun (post, label) -> post.Body) 
            |> Seq.map preprocess
            |> extractWords
            |> filterStopWords
        let init = 
            words 
            |> Seq.map (fun w -> (w, 0))
            |> Map.ofSeq
        dataset
        |> Seq.map (fun (post, label) -> (label, post.Body))
        |> prepare
        |> Seq.fold (fun state (label, sample) -> folder state sample) init

let tokensFromTitles (dataset: seq<Charon.Post*string>) folder =
        let words = 
            dataset 
            |> Seq.map (fun (post, label) -> post.Title) 
            |> extractWords
            |> filterStopWords
        let init = 
            words 
            |> Seq.map (fun w -> (w, 0))
            |> Map.ofSeq
        dataset
        |> Seq.map (fun (post, label) -> (label, post.Title))
        |> prepare
        |> Seq.fold (fun state (label, sample) -> folder state sample) init

let tagsAsText (post: Charon.Post) = String.Join(" ", post.Tags)
let tokensFromTags (dataset: seq<Charon.Post*string>) folder =
        let words = 
            dataset 
            |> Seq.map (fun (post, label) -> tagsAsText post) 
            |> extractWords
            |> filterStopWords
        let init = 
            words 
            |> Seq.map (fun w -> (w, 0))
            |> Map.ofSeq
        dataset
        |> Seq.map (fun (post, label) -> (label, tagsAsText post))
        |> prepare
        |> Seq.fold (fun state (label, sample) -> folder state sample) init


let saveTokenCounts (filePath: string) (tokens: string * Map<string, int>) =       
    let refMap = snd tokens |> Map.toSeq
    let convertToText =
        refMap       
        |> Seq.map (fun (word, count) -> word + "," + count.ToString())               
    System.IO.File.AppendAllLines(filePath, convertToText)

// Extract and save tokens by class
// set, no code
for category in categories do
    printfn "%s: Processing %s" (DateTime.Now.ToString()) category
    let posts = 
        trainSet 
        |> List.filter (fun (p, c) -> c = category) 
    let file = @"..\..\..\" + "set-nocode-" + category + ".csv"
    ("", tokensFromNoCode posts setFold)
    |> saveTokenCounts file
    |> ignore
// set, full text
for category in categories do
    printfn "%s: Processing %s" (DateTime.Now.ToString()) category
    let posts = 
        trainSet 
        |> List.filter (fun (p, c) -> c = category) 
    let file = @"..\..\..\" + "set-full-" + category + ".csv"
    ("", tokensFrom posts setFold)
    |> saveTokenCounts file
    |> ignore

// set, title
for category in categories do
    printfn "%s: Processing %s" (DateTime.Now.ToString()) category
    let posts = 
        trainSet 
        |> List.filter (fun (p, c) -> c = category) 
    let file = @"..\..\..\" + "set-title-" + category + ".csv"
    ("", tokensFromTitles posts setFold)
    |> saveTokenCounts file
    |> ignore

// set, tags
for category in categories do
    printfn "%s: Processing %s" (DateTime.Now.ToString()) category
    let posts = 
        trainSet 
        |> List.filter (fun (p, c) -> c = category) 
    let file = @"..\..\..\" + "set-tags-" + category + ".csv"
    ("", tokensFromTags posts setFold)
    |> saveTokenCounts file
    |> ignore


let grabTokens fold target (minByGroup: Map<string,int>) =
    let root = @"..\..\..\"
    let readWordsMoreFrequentThan (filePath: string) min = 
        parseCsv filePath
        |> List.map (fun li -> li.[0], Convert.ToInt32(li.[1]))
        |> List.filter (fun (w, c) -> c > min)
        |> List.map (fun (w, c) -> w)
        |> Set.ofList
    [ for category in categories do
          let file = root + fold + "-" + target + "-" + category + ".csv"
          yield (readWordsMoreFrequentThan file (minByGroup.[category])) ] 
    |> Set.unionMany

// select 10% top
//let minima =
//        Map.empty
//           .Add("not a real question", 12)
//           .Add("not constructive",    23)
//           .Add("off topic",           18)
//           .Add("open",                15)
//           .Add ("too localized",      13)
// top 1000
//let minima =
//        Map.empty
//           .Add("not a real question", 131)
//           .Add("not constructive",    87)
//           .Add("off topic",           98)
//           .Add("open",                465)
//           .Add ("too localized",      43)

for notReal in 100 .. 100 .. 300 do
    for notConstructive in 100 .. 200 .. 500 do
        for offTopic in 100 .. 100 .. 300 do
            for op in 250 .. 250 .. 750 do
                for loc in 50 .. 50 .. 200 do
                    let minima =
                            Map.empty
                               .Add("not a real question", notReal)
                               .Add("not constructive",    notConstructive)
                               .Add("off topic",           offTopic)
                               .Add("open",                op)
                               .Add ("too localized",      loc)
                    printfn "Trying %i %i %i %i %i" notReal notConstructive offTopic op loc
                    let tokensSetNocode10 = grabTokens "set" "nocode" minima
                    let trainNocode = trainSet |> Seq.map (fun (post, label) -> (label, preprocess post.Body))

                    let knowledge = train setOfWords trainNocode tokensSetNocode10
                    let classifier = classify knowledge
                    let model = fun (post: Charon.Post) -> (classifier (preprocess post.Body)) |> renormalize
                    benchmark model validateSet

// 1.25
//                    let minima =
//                            Map.empty
//                               .Add("not a real question", 1000)
//                               .Add("not constructive",    500)
//                               .Add("off topic",           700)
//                               .Add("open",                2500)
//                               .Add ("too localized",      200)

// 1.18
//                    let minima =
//                            Map.empty
//                               .Add("not a real question", 2000)
//                               .Add("not constructive",    1000)
//                               .Add("off topic",           1500)
//                               .Add("open",                5000)
//                               .Add ("too localized",      500)

// 1.22
//                    let minima =
//                            Map.empty
//                               .Add("not a real question", 4000)
//                               .Add("not constructive",    2000)
//                               .Add("off topic",           300)
//                               .Add("open",                10000)
//                               .Add ("too localized",      1000)

// 1.24
//                    let minima =
//                            Map.empty
//                               .Add("not a real question", 4000)
//                               .Add("not constructive",    2000)
//                               .Add("off topic",           3000)
//                               .Add("open",                10000)
//                               .Add ("too localized",      1000)

// 1.21
//                    let minima =
//                            Map.empty
//                               .Add("not a real question", 3000)
//                               .Add("not constructive",    1500)
//                               .Add("off topic",           2200)
//                               .Add("open",                7500)
//                               .Add ("too localized",      750)

// 1.20
//                    let minima =
//                            Map.empty
//                               .Add("not a real question", 1500)
//                               .Add("not constructive",    750)
//                               .Add("off topic",           1100)
//                               .Add("open",                3500)
//                               .Add ("too localized",      400)

                    let minima =
                            Map.empty
                               .Add("not a real question", 2000)
                               .Add("not constructive",    1000)
                               .Add("off topic",           1500)
                               .Add("open",                5000)
                               .Add ("too localized",      500)
//                    printfn "Trying %i %i %i %i %i" notReal notConstructive offTopic op loc
                    let tokensSetNocode10 = grabTokens "set" "nocode" minima
                    let trainNocode = trainSet |> Seq.map (fun (post, label) -> (label, preprocess post.Body))

                    let knowledge = train setOfWords trainNocode tokensSetNocode10
                    let classifier = classify knowledge
                    let model = fun (post: Charon.Post) -> (classifier (preprocess post.Body)) |> renormalize
                    benchmark model validateSet



// optimize titles
// 1.18
//                    let minima =
//                            Map.empty
//                               .Add("not a real question", 200)
//                               .Add("not constructive",    100)
//                               .Add("off topic",           150)
//                               .Add("open",                500)
//                               .Add ("too localized",      50)

// 1.15
//                    let minima =
//                            Map.empty
//                               .Add("not a real question", 100)
//                               .Add("not constructive",    50)
//                               .Add("off topic",           75)
//                               .Add("open",                250)
//                               .Add ("too localized",      25)

// 1.15
//                    let minima =
//                            Map.empty
//                               .Add("not a real question", 50)
//                               .Add("not constructive",    25)
//                               .Add("off topic",           35)
//                               .Add("open",                50)
//                               .Add ("too localized",      10)

// 1.152786
//                    let minima =
//                            Map.empty
//                               .Add("not a real question", 10)
//                               .Add("not constructive",    5)
//                               .Add("off topic",           5)
//                               .Add("open",                5)
//                               .Add ("too localized",      5)

// 
//                    let minima =
//                            Map.empty
//                               .Add("not a real question", 20)
//                               .Add("not constructive",    10)
//                               .Add("off topic",           10)
//                               .Add("open",                10)
//                               .Add ("too localized",      5)

// 1.158
//                    let minima =
//                            Map.empty
//                               .Add("not a real question", 100)
//                               .Add("not constructive",    50)
//                               .Add("off topic",           75)
//                               .Add("open",                250)
//                               .Add ("too localized",      25)
// 1.147
//                    let minima =
//                            Map.empty
//                               .Add("not a real question", 50)
//                               .Add("not constructive",    25)
//                               .Add("off topic",           35)
//                               .Add("open",                100)
//                               .Add ("too localized",      10)

// 1.149
//                    let minima =
//                            Map.empty
//                               .Add("not a real question", 25)
//                               .Add("not constructive",    10)
//                               .Add("off topic",           15)
//                               .Add("open",                50)
//                               .Add ("too localized",      5)

                    let minima =
                            Map.empty
                               .Add("not a real question", 50)
                               .Add("not constructive",    25)
                               .Add("off topic",           35)
                               .Add("open",                100)
                               .Add ("too localized",      10)
                    //printfn "Trying %i %i %i %i %i" notReal notConstructive offTopic op loc
                    let tokensSetNocode10 = grabTokens "set" "title" minima
                    let trainNocode = trainSet |> Seq.map (fun (post, label) -> (label, post.Title))

                    let knowledge = train setOfWords trainNocode tokensSetNocode10
                    let classifier = classify knowledge
                    let model = fun (post: Charon.Post) -> (classifier post.Title) |> renormalize
                    benchmark model validateSet


// OPTIMIZE TAGS

// 1.3198                    
//                            Map.empty
//                               .Add("not a real question", 2000)
//                               .Add("not constructive",    1000)
//                               .Add("off topic",           1500)
//                               .Add("open",                5000)
//                               .Add ("too localized",      500)

// 1.300
//                    let minima =
//                            Map.empty
//                               .Add("not a real question", 1000)
//                               .Add("not constructive",    500)
//                               .Add("off topic",           750)
//                               .Add("open",                2500)
//                               .Add ("too localized",      250)
//

// 1.2701
//                    let minima =
//                            Map.empty
//                               .Add("not a real question", 500)
//                               .Add("not constructive",    250)
//                               .Add("off topic",           350)
//                               .Add("open",                1000)
//                               .Add ("too localized",      100)

// 1.2231
//                    let minima =
//                            Map.empty
//                               .Add("not a real question", 200)
//                               .Add("not constructive",    100)
//                               .Add("off topic",           150)
//                               .Add("open",                50)
//                               .Add ("too localized",      50)

// 1.213
//                    let minima =
//                            Map.empty
//                               .Add("not a real question", 50)
//                               .Add("not constructive",    50)
//                               .Add("off topic",           50)
//                               .Add("open",                50)
//                               .Add ("too localized",      20)

// 1.2075
//                    let minima =
//                            Map.empty
//                               .Add("not a real question", 20)
//                               .Add("not constructive",    10)
//                               .Add("off topic",           15)
//                               .Add("open",                50)
//                               .Add ("too localized",      5)
//

// 1.2056
//                    let minima =
//                            Map.empty
//                               .Add("not a real question", 5)
//                               .Add("not constructive",    5)
//                               .Add("off topic",           5)
//                               .Add("open",                10)
//                               .Add ("too localized",      5)

// 1.215
//                    let minima =
//                            Map.empty
//                               .Add("not a real question", 1)
//                               .Add("not constructive",    1)
//                               .Add("off topic",           1)
//                               .Add("open",                1)
//                               .Add ("too localized",      1)

                    let minima =
                            Map.empty
                               .Add("not a real question", 5)
                               .Add("not constructive",    5)
                               .Add("off topic",           5)
                               .Add("open",                10)
                               .Add ("too localized",      5)
                    //printfn "Trying %i %i %i %i %i" notReal notConstructive offTopic op loc
                    let tokensSetNocode10 = grabTokens "set" "tags" minima
                    let trainNocode = trainSet |> Seq.map (fun (post, label) -> (label, tagsAsText post))

                    let knowledge = train setOfWords trainNocode tokensSetNocode10
                    let classifier = classify knowledge
                    let model = fun (post: Charon.Post) -> (classifier (tagsAsText post)) |> renormalize
                    benchmark model validateSet





printfn "Extracting tokens based on set count"
let setTokens = tokensFrom trainSet setFold

let trainBase = trainSet |> Seq.map (fun (post, label) -> (label, post.Body))

for count in 1 .. 5 do
    printfn "SetOfWords training on set tokens present more than %i times" count
    let knowledge =
        let tokens = setTokens |> Map.filter (fun k c -> c >= count) |> Map.toSeq |> Seq.map (fun (k, c) -> k)
        train setOfWords trainBase tokens
    let classifier = classify knowledge
    let model = fun (post: Charon.Post) -> (classifier (post.Body) |> renormalize)
    printfn "benchmarking"
    benchmark model validateSet

let smallSet = validateSet |> Seq.take 5000
for count in 1 .. 5 do
    printfn "BagOfWords training on set tokens present more than %i times" count
    let knowledge =
        let tokens = setTokens |> Map.filter (fun k c -> c >= count) |> Map.toSeq |> Seq.map (fun (k, c) -> k)
        train bagOfWords trainBase tokens
    let classifier = classify knowledge
    let model = fun (post: Charon.Post) -> (classifier (post.Body) |> renormalize)
    printfn "benchmarking"
    benchmark model smallSet

for count in 1 .. 5 do
    printfn "BagOfWords training on set tokens present more than %i times, classify removing code" count
    let knowledge =
        let tokens = setTokens |> Map.filter (fun k c -> c >= count) |> Map.toSeq |> Seq.map (fun (k, c) -> k)
        train bagOfWords trainBase tokens
    let classifier = classify knowledge
    let model = fun (post: Charon.Post) -> (classifier (preprocess post.Body) |> renormalize)
    printfn "benchmarking"
    benchmark model smallSet

let bagTokens = tokensFrom trainSet bagFold

for count in 1 .. 5 do
    printfn "SetOfWords training on set tokens present more than %i times" count
    let knowledge =
        let tokens = bagTokens |> Map.filter (fun k c -> c >= count) |> Map.toSeq |> Seq.map (fun (k, c) -> k)
        train setOfWords trainBase tokens
    let classifier = classify knowledge
    let model = fun (post: Charon.Post) -> (classifier (post.Body) |> renormalize)
    printfn "benchmarking"
    benchmark model smallSet

for count in 1 .. 5 do
    printfn "BagOfWords training on set tokens present more than %i times, classify removing code" count
    let knowledge =
        let tokens = bagTokens |> Map.filter (fun k c -> c >= count) |> Map.toSeq |> Seq.map (fun (k, c) -> k)
        train bagOfWords trainBase tokens
    let classifier = classify knowledge
    let model = fun (post: Charon.Post) -> (classifier (preprocess post.Body) |> renormalize)
    printfn "benchmarking"
    benchmark model smallSet



// trying extracting only tokens appearing frequently by class
let minByClass (dataset: seq<Charon.Post*string>) min =
    dataset 
    |> Seq.groupBy (fun (p,c) -> c)
    |> Seq.map (fun (c, g) -> tokensFrom g setFold |> Map.filter (fun k v -> v >= min) |> Map.toSeq)
    |> Seq.map (fun t -> t |> Seq.map (fun (k, c) -> k)  |> Set.ofSeq)
    |> Set.unionMany

for count in 2 .. 5 do
    printfn "BagOfWords training on frequent set tokens by class, more than %i times" count
    let knowledge =
        let tokens = minByClass trainSet count
        train setOfWords trainBase tokens
    let classifier = classify knowledge
    let model = fun (post: Charon.Post) -> (classifier (preprocess post.Body) |> renormalize)
    printfn "benchmarking"
    benchmark model smallSet

for count in 2 .. 10 do
    printfn "Model 1 depth = %i" count
    let knowledge =
        let tokens = minByClass trainSet count
        train bagOfWords trainBase tokens
    let classifier = classify knowledge
    let model = fun (post: Charon.Post) -> (classifier post.Body) |> renormalize
    printfn "benchmarking"
    benchmark model smallSet


let filteredTokens25 =
    let readWordsMoreFrequentThan (filePath: string) min = 
        parseCsv filePath
        |> List.map (fun li -> li.[0], Convert.ToInt32(li.[1]))
        |> List.filter (fun (w, c) -> c > min)
        |> List.map (fun (w, c) -> w)
        |> Set.ofList
    let openTokens = readWordsMoreFrequentThan @"..\..\..\open.csv" 6
    let notConstructiveTokens = readWordsMoreFrequentThan @"..\..\..\notconstructive.csv" 5
    let notRealTokens = readWordsMoreFrequentThan @"..\..\..\notreal.csv" 5
    let offTopicTokens = readWordsMoreFrequentThan @"..\..\..\offtopic.csv" 4
    let tooLocalizedTokens = readWordsMoreFrequentThan @"..\..\..\toolocalized.csv" 7
    Set.unionMany [ openTokens;
                    notConstructiveTokens;
                    notRealTokens;
                    offTopicTokens;
                    tooLocalizedTokens]

let knowledge25 = train setOfWords trainBase filteredTokens

let classifier25 = classify knowledge
let model25 = fun (post: Charon.Post) -> (classifier post.Body) |> renormalize

let filteredTokens10 =
    let readWordsMoreFrequentThan (filePath: string) min = 
        parseCsv filePath
        |> List.map (fun li -> li.[0], Convert.ToInt32(li.[1]))
        |> List.filter (fun (w, c) -> c > min)
        |> List.map (fun (w, c) -> w)
        |> Set.ofList
    let notConstructiveTokens = readWordsMoreFrequentThan @"..\..\..\notconstructive.csv" 30
    let notRealTokens = readWordsMoreFrequentThan @"..\..\..\notreal.csv" 19
    let offTopicTokens = readWordsMoreFrequentThan @"..\..\..\offtopic.csv" 28
    let openTokens = readWordsMoreFrequentThan @"..\..\..\open.csv" 26
    let tooLocalizedTokens = readWordsMoreFrequentThan @"..\..\..\toolocalized.csv" 28
    Set.unionMany [ openTokens;
                    notConstructiveTokens;
                    notRealTokens;
                    offTopicTokens;
                    tooLocalizedTokens]

let knowledge10 = train setOfWords trainBase filteredTokens10
let knowledge10 = train bagOfWords trainBase filteredTokens10
let classifier10 = classify knowledge10
let model10 = fun (post: Charon.Post) -> (classifier10 post.Body) |> renormalize

let filteredTokens1 =
    let readWordsMoreFrequentThan (filePath: string) min = 
        parseCsv filePath
        |> List.map (fun li -> li.[0], Convert.ToInt32(li.[1]))
        |> List.filter (fun (w, c) -> c > min)
        |> List.map (fun (w, c) -> w)
        |> Set.ofList
    let notConstructiveTokens = readWordsMoreFrequentThan @"..\..\..\notconstructive.csv" 530
    let notRealTokens = readWordsMoreFrequentThan @"..\..\..\notreal.csv" 457
    let offTopicTokens = readWordsMoreFrequentThan @"..\..\..\offtopic.csv" 478
    let openTokens = readWordsMoreFrequentThan @"..\..\..\open.csv" 790
    let tooLocalizedTokens = readWordsMoreFrequentThan @"..\..\..\toolocalized.csv" 424
    Set.unionMany [ openTokens;
                    notConstructiveTokens;
                    notRealTokens;
                    offTopicTokens;
                    tooLocalizedTokens]

let knowledge1 = train setOfWords trainBase filteredTokens1
let classifier1 = classify knowledge1
let model1 = fun (post: Charon.Post) -> (classifier1 post.Body) |> renormalize
benchmark model1 validateSet

let filteredTokensHybrid =
    let readWordsMoreFrequentThan (filePath: string) min = 
        parseCsv filePath
        |> List.map (fun li -> li.[0], Convert.ToInt32(li.[1]))
        |> List.filter (fun (w, c) -> c > min)
        |> List.map (fun (w, c) -> w)
        |> Set.ofList
    let notConstructiveTokens = readWordsMoreFrequentThan @"..\..\..\notconstructive.csv" 20
    let notRealTokens = readWordsMoreFrequentThan @"..\..\..\notreal.csv" 20
    let offTopicTokens = readWordsMoreFrequentThan @"..\..\..\offtopic.csv" 20
    let openTokens = readWordsMoreFrequentThan @"..\..\..\open.csv" 100
    let tooLocalizedTokens = readWordsMoreFrequentThan @"..\..\..\toolocalized.csv" 20
    Set.unionMany [ openTokens;
                    notConstructiveTokens;
                    notRealTokens;
                    offTopicTokens;
                    tooLocalizedTokens]

let knowledge5 = train setOfWords trainBase filteredTokensHybrid
let classifier5 = classify knowledge5
let model5 = fun (post: Charon.Post) -> (classifier5 post.Body) |> renormalize
benchmark model5 validateSet