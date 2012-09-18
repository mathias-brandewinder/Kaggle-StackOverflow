namespace MachineLearning

module LogisticRegression =

    open System

    let sigmoid x = 1.0 / (1.0 + exp -x)

    let dot (vec1: float list) 
            (vec2: float list) =
        List.zip vec1 vec2
        |> List.map (fun e -> fst e * snd e)
        |> List.sum

    let add (vec1: float list) 
            (vec2: float list) =
        List.zip vec1 vec2
        |> List.map (fun e -> fst e + snd e)

    let scalar alpha (vector: float list) =
        List.map (fun e -> alpha * e) vector
    
    // Weights have 1 element more than observations, for constant
    let predict (weights: float list) 
                (obs: float list) =
        1.0 :: obs
        |> dot weights 
        |> sigmoid

    let error (weights: float list)
              (obs: float list)
              label =
        label - predict weights obs

    let norm (vector: float list) = List.sumBy (fun e -> e * e) vector |> sqrt

    let changeRate before after =
        let numerator = 
            List.zip before after
            |> List.map (fun (b, a) -> b - a)
            |> norm
        let denominator = norm before
        numerator / denominator

    let update alpha 
               (weights: float list)
               (observ: float list)
               label =      
        add weights (scalar (alpha * (error weights observ label)) (1.0 :: observ))

    let train (dataset: (float * float list) seq) 
              passes
              alpha =
        let dataset = dataset |> Seq.toArray
        let len = Array.length dataset
        let iterations = passes * len
        let vars = dataset |> Seq.nth 1 |> snd |> List.length
        let weights = [ for i in 0 .. vars -> 1.0 ] // 1 more weight for constant
        let rng = new Random()

        let rec descent iter a curWeights =
            match iter with 
            | 0 -> curWeights
            | _ ->
                let (label, observ) = dataset.[rng.Next(len)] 
                let weights = update alpha curWeights observ label
                let a = a * 0.999 + 0.001
                printfn "%f" (changeRate curWeights weights)
                List.iter (fun e -> printf " %f " e) weights
                descent (iter - 1) a weights

        descent iterations alpha weights