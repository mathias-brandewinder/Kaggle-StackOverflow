namespace Charon

module Validation =

    // Compute metrics with probabilities by class vs real classes
    let quality predictions actuals = 
        Seq.zip predictions actuals
        |> Seq.map (fun (predicted, actual) -> 
            Seq.sumBy (fun (cl, proba) -> 
                if actual = cl then log proba else 0.) predicted)
        |> Seq.average |> (*)-1.


