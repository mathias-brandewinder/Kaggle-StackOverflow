namespace Charon

module Validation =

    open Charon
    open Charon.Data

    // Compute metrics with probabilities by class vs real classes
    let quality predictions actuals = 
        Seq.zip predictions actuals
        |> Seq.map (fun (predicted, actual) -> 
            Seq.sumBy (fun (cl, proba) -> 
                if actual = cl then log proba else 0.) predicted)
        |> Seq.average |> (*)-1.

    let evaluate prediction (category: string) =
        prediction
        |> Map.toSeq
        |> Seq.sumBy (fun (cl, proba) -> 
            if category = cl then - log proba else 0.0)

    let q (dataset: (Post * string) seq) (model: Post -> Map<string, float>) =
        dataset
        |> Seq.map (fun (post, cat) -> (model post), cat)
        |> Seq.map (fun (predict, cat) -> evaluate predict cat)
        |> Seq.average

    let benchmark (model: Post -> Map<string, float>) dataset =
        let priorModel = fun (post:Post) -> priors
        let trainingPriorModel = fun (post:Post) -> trainingPriors
        let uniformModel = fun (post:Post) -> uniformPriors
        let awfulModel = fun (post:Post) -> awfulPriors

        printfn "Model          %f" (q dataset model)
        printfn "Priors         %f" (q dataset priorModel)
        printfn "Train Priors   %f" (q dataset trainingPriorModel)
        printfn "Uniform        %f" (q dataset uniformModel)
        printfn "Awful          %f" (q dataset awfulModel)
        

//// Visualize classification results by group
//let visualizeByGroup test testSet =
//    testSet
//    |> Seq.map (fun (c, t) -> 
//        let result = test t |> Seq.maxBy snd |> fst
//        c, result)
//    |> Seq.groupBy fst
//    |> Seq.map (fun (cl, gr) ->
//        let grouped = gr |> Seq.groupBy snd
//        cl,
//        grouped |> Seq.map (fun (res, cases) -> res, Seq.length cases))
//    |> Seq.iter (fun (cl, results) -> 
//        printfn ""
//        printfn "Real: %s" cl
//        results |> Seq.iter (fun (g, c) -> printfn "%s, %i" g c))

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