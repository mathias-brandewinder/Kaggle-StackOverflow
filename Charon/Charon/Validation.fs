namespace Charon

module Validation =

    open Charon
    open Charon.Data

    let evaluate prediction (category: string) =
        prediction
        |> Map.toSeq
        |> Seq.sumBy (fun (cl, proba) -> 
            if category = cl then - log proba else 0.0)

    // Compute metrics with probabilities by class vs real classes
    let quality predictions actuals = 
        Seq.zip predictions actuals
        |> Seq.map (fun (predicted, actual) -> evaluate (Map.ofSeq predicted) actual)
        |> Seq.average

    let predict (dataset: (Post * string) seq) 
                (model: Post -> Map<string, float>) =
        dataset
        |> Seq.map (fun (post, cat) -> (model post), cat)

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
  
    // Group of origin by predicted group   
    let originByPredictedGroup (predictions: (Map<string, float> * string) seq) =
        predictions
        |> Seq.map (fun (p, c) -> 
            p |> Map.toSeq |> Seq.maxBy (fun (k, v) -> v) |> fst,
            c)
        |> Seq.groupBy fst
        |> Seq.map (fun (cl, gr) ->
            let grouped = gr |> Seq.groupBy snd
            cl,
            grouped |> Seq.map (fun (res, cases) -> res, Seq.length cases))
        |> Seq.iter (fun (cl, results) -> 
            printfn ""
            printfn "Predicted: %s" cl
            results |> Seq.iter (fun (g, c) -> printfn "   %s, %i" g c))

    // Predicted class by class of origin
    let classificationByGroup (predictions: (Map<string, float> * string) seq) =
        predictions
        |> Seq.map (fun (p, c) -> 
            p |> Map.toSeq |> Seq.maxBy (fun (k, v) -> v) |> fst,
            c)
        |> Seq.groupBy snd
        |> Seq.map (fun (cl, gr) ->
            let grouped = gr |> Seq.groupBy fst
            cl,
            grouped |> Seq.map (fun (res, cases) -> res, Seq.length cases))
        |> Seq.iter (fun (cl, results) -> 
            printfn "Real: %s" cl
            results |> Seq.iter (fun (g, c) -> printfn "   %s, %i" g c))

    // Evaluate % correctly classified
    let correctByGroup (predictions: (Map<string, float> * string) seq) =
        predictions
        |> Seq.map (fun (p, c) -> 
            p |> Map.toSeq |> Seq.maxBy (fun (k, v) -> v) |> fst,
            c)
        |> Seq.groupBy snd
        |> Seq.map (fun (cl, gr) ->
            cl,
            gr |> Seq.map (fun (c, p) -> if c = p then 1.0 else 0.0) |> Seq.average)
        |> Seq.iter (fun (cl, proba) -> printfn "Real: %s: %f" cl proba)

    // Evaluate % correctly classified by predicted group
    let correctByPredictedGroup (predictions: (Map<string, float> * string) seq) =
        predictions
        |> Seq.map (fun (p, c) -> 
            p |> Map.toSeq |> Seq.maxBy (fun (k, v) -> v) |> fst,
            c)
        |> Seq.groupBy fst
        |> Seq.map (fun (cl, gr) ->
            cl,
            gr |> Seq.map (fun (c, p) -> if c = p then 1.0 else 0.0) |> Seq.average)
        |> Seq.iter (fun (cl, proba) -> printfn "Real: %s: %f" cl proba)

    let qualityByGroup (predictions: (Map<string, float> * string) seq) =
        predictions
        |> Seq.groupBy snd
        |> Seq.map (fun (cl, data) -> 
            cl, 
            data |> Seq.map (fun (p, c) -> evaluate p c) |> Seq.average)
        |> Seq.iter (fun (cl, qu) -> printfn "%s: %f" cl qu)
