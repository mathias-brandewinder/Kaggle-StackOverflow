namespace MachineLearning

module LogisticRegression =

    open System

    let sigmoid x = 1.0 / (1.0 + exp -x)

    // Vector dot product
    let dot (vec1: float list) 
            (vec2: float list) =
        List.zip vec1 vec2
        |> List.map (fun e -> fst e * snd e)
        |> List.sum

    // Vector addition
    let add (vec1: float list) 
            (vec2: float list) =
        List.zip vec1 vec2
        |> List.map (fun e -> fst e + snd e)

    // Vector scalar product
    let scalar alpha (vector: float list) =
        List.map (fun e -> alpha * e) vector
    
    let norm (vector: float list) = List.sumBy (fun e -> e * e) vector |> sqrt

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

    let simpleTrain (dataset: (float * float list) seq) 
                    passes
                    alpha =

        let rec descent iter curWeights =
            match iter with 
            | 0 -> curWeights
            | _ ->
                dataset
                |> Seq.fold (fun w (label, observ) -> 
                    update alpha w observ label) curWeights
                |> descent (iter - 1)

        let vars = dataset |> Seq.nth 1 |> snd |> List.length
        let weights = [ for i in 0 .. vars -> 1.0 ] // 1 more weight for constant

        descent passes weights

    let train (dataset: (float * float list) seq) epsilon =
        let cooling = 0.9
        let dataset = dataset |> Seq.toArray
        let len = dataset |> Array.length
        let rng = new Random()
        let indices = Seq.initInfinite(fun x -> rng.Next(len))

        let rec descent curWeights alpha =
            let updatedWeights =
                indices
                |> Seq.take len
                |> Seq.fold (fun w i -> 
                    let (label, observ) = dataset.[i]
                    printfn "%i" i
                    update alpha w observ label) curWeights
            if changeRate curWeights updatedWeights <= epsilon
            then updatedWeights
            else 
                let coolerAlpha = cooling * alpha
                descent updatedWeights coolerAlpha

        let vars = dataset |> Seq.nth 1 |> snd |> List.length
        let weights = [ for i in 0 .. vars -> 1.0 ] // 1 more weight for constant

        descent weights 1.0