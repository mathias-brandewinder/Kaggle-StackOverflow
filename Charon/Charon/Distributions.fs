namespace Charon

module Distributions =

    // Normalize a map of outcomes of type 'a
    // with float weights, to a map where weights
    // sum to 1.0 (i.e. a distribution)
    let normalize (result: ('a * float) seq) =
        let total =  result |> Seq.sumBy snd       
        result
        |> Seq.map (fun (key, value) -> (key, value / total))
        |> Map.ofSeq

    // Create a linear combination of 2 maps
    let combine (categories: string list) 
                ((w1: float), (d1: Map<string, float>)) 
                ((w2: float), (d2: Map<string, float>)) =
        categories
        |> List.fold (fun (comb: Map<string, float>) cat ->
            comb.Add(cat, w1 * d1.[cat] + w2 * d2.[cat])) Map.empty