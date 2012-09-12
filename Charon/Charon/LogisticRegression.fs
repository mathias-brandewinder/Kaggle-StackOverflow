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

    let scalar (vec: float list) alpha =
        List.map (fun e -> alpha * e) vec

    let predict (weights: float list) 
                (obs: float list) =
        dot weights obs |> sigmoid

    let error (weights: float list)
              (obs: float list)
              label =
        label - predict weights obs

    let update alpha 
               (weights: float list)
               (obs: float list)
               label =
        add weights (scalar obs (alpha * (error weights obs label)))

    let train (data: (float list) seq) 
              (labels: float seq)
              passes
              alpha =
        let dataset =
            data
            |> Seq.map (fun e -> 1.0 :: e)
            |> Seq.zip labels
            |> Seq.toArray
        let l = Array.length dataset
        let vars = data |> Seq.nth 1 |> List.length |> (+) 1
        let iterations = passes * l
        let weights = [ for i in 1 .. vars -> 1.0 ]
        let rng = new Random()

        let rec descent iter curWeights =
            match iter with 
            | 0 -> curWeights
            | _ ->
                let (lab, obs) = dataset.[rng.Next(l)] 
                let weights = update alpha curWeights obs lab
                descent (iter - 1) weights

        descent iterations weights

    let classify (weights: float list) 
                 (obs: float list) =
        1.0 :: obs
        |> dot weights 
        |> sigmoid