namespace Charon

module NumericBayes =

    let variableKnowledge (variable: Charon.Post -> 'a) 
                          (threshold: 'a) 
                          (trainingset: (Charon.Post * string) seq) =
        trainingset 
        |> Seq.map (fun (post, label) -> variable(post), label)
        |> Seq.groupBy snd
        |> Seq.map (fun (label, data) ->
            let total = Seq.length data
            let below = data |> Seq.filter (fun e -> fst e <= threshold) |> Seq.length
            label, (float)below / (float)total)
        |> Map.ofSeq 

    // Generic Bayesian model, based on whether variable is <= / > than threshold
    let numericModel (post: Charon.Post)
                     (variable: Charon.Post -> 'a)
                     (threshold: 'a)
                     (probaBelowThreshold: Map<string, float>)
                     (priors: Map<string, float>) =
        let estimates = 
            if variable(post) <= threshold
            then probaBelowThreshold |> Map.map (fun k v -> v * priors.[k])
            else probaBelowThreshold |> Map.map (fun k v -> (1.0 - v) * priors.[k])
        let total = estimates |> Map.fold (fun acc k v -> acc + v) 0.0
        estimates |> Map.map (fun k v -> v / total)

