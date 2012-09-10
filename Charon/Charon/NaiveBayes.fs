namespace MachineLearning

module NaiveBayes =

    open System
    open System.Text
    open Charon.Preprocessing
    open Charon.Distributions

    // "Tokenize" the dataset: break each text sample
    // into words and how many times they are used.
    let prepare (dataset: (string * string) seq) =
        dataset
        |> Seq.map (fun (label, sample) -> (label, wordsCount sample))

    // Set-of-Words Accumulator function: 
    // state is the current count for each word so far, 
    // sample the tokenized text.
    // setFold increases the count by 1 if the word is 
    // present in the sample.
    let setFold (state: Map<string, int>) sample =
        state
        |> Map.map (fun token count ->
            if sample |> Map.containsKey(token) 
            then count + 1
            else count)

    // Bag-of-Words Accumulator function: 
    // state is the current count for each word so far, 
    // sample the tokenized text.
    // setFold increases the count by the number of occurences
    // of the word in the sample.
    let bagFold (state: Map<string, int>) sample =
        state
        |> Map.map (fun token count -> 
            if sample |> Map.containsKey(token)
            then count + sample.[token]
            else count)

    // Aggregate words frequency across the dataset,
    // using the provided folder.
    // (Supports setFold and bagFold)
    let frequency folder (dataset: (string * Map<string, int>) seq) (words: string seq) =
        let init = 
            words 
            |> Seq.map (fun w -> (w, 1))
            |> Map.ofSeq
        dataset
        |> Seq.fold (fun state (label, sample) -> folder state sample) init

    let topWords dataset =
        let words = dataset |> Seq.map snd |> extractWords
        let init = 
            words 
            |> Seq.map (fun w -> (w, 0))
            |> Map.ofSeq
        dataset
        |> prepare
        |> Seq.fold (fun state (label, sample) -> bagFold state sample) init
        |> Seq.sortBy (fun kv -> - (kv.Value) )
        |> Seq.map (fun kv -> kv.Key)

    let topByClass dataset top =
        dataset
        |> Seq.groupBy fst
        |> Seq.map (fun (cl, group) -> topWords group |> Seq.take top)
        |> Seq.concat
        |> Set.ofSeq

    let topByClassNoCodeNoStopWords dataset top =
        dataset
        |> Seq.groupBy fst
        |> Seq.map (fun (cl, group) -> 
            group 
            |> Seq.map (fun (c, txt) -> c, preprocess txt)
            |> topWords 
            |> Seq.take top 
            |> Set.ofSeq 
            |> filterStopWords)
        |> Set.unionMany
        
    let filteredTopWords dataset =
        let words = 
            dataset 
            |> Seq.map snd 
            |> extractWords
            |> filterStopWords
        let init = 
            words 
            |> Seq.map (fun w -> (w, 0))
            |> Map.ofSeq
        dataset
        |> prepare
        |> Seq.fold (fun state (label, sample) -> bagFold state sample) init
        |> Seq.filter (fun kv -> (kv.Value) > 2 )
        |> Seq.map (fun kv -> kv.Key)
    
    // Convenience functions for training the classifier
    // using set-of-Words and bag-of-Words frequency.
    let bagOfWords dataset words = frequency bagFold dataset words
    let setOfWords dataset words = frequency setFold dataset words

    // Converts 2 integers into a proportion.
    let prop (count, total) = (float)count / (float)total

    // Train based on a set of words and a dataset:
    // the dataset is "tokenized", and broken down into
    // one dataset per classification label.
    // For each group, we compute:
    // the proportion of the group relative to total,
    // the probability of each word within the group.
    let train frequency dataset words =
        let size = Seq.length dataset
        dataset
        |> prepare
        |> Seq.groupBy fst
        |> Seq.map (fun (label, data) -> 
            label, Seq.length data, frequency data words)
        |> Seq.map (fun (label, total, tokenCount) ->
            let totTokens = Map.fold (fun total key value -> total + value) 0 tokenCount
            label, 
            prop(total, size), 
            Map.map (fun token count -> prop(count, totTokens)) tokenCount)
        |> Seq.toList

    // replace estimated priors with custom priors by category
    let updatePriors estimators (priors: Map<string, float>) =
        estimators 
        |> Seq.map (fun (cat, proba, stuff) -> cat, priors.[cat], stuff)

    let classify estimator text =
        let tokenized = vocabulary text
        estimator
        |> Seq.map (fun (label, proba, tokens) ->
            label,
            tokens
            |> Map.fold (fun p token value -> 
                if Set.contains(token) tokenized 
                then p + log(value) 
                else p) (log proba))
        |> Seq.toList

    // Convert Naive Bayes to Distribution over categories
    // based on each category likelihood
    let renormalize (result: ('a * float) seq) =
        let min = result |> Seq.minBy snd |> snd
        result
        |> Seq.map (fun (cat, value) -> (cat, exp (value - min)))
        |> normalize