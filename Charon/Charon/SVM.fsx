#r "Microsoft.VisualBasic"
#r "..\..\..\SVM.dll"

#load "Data.fs"
#load "Validation.fs"
System.IO.Directory.SetCurrentDirectory(__SOURCE_DIRECTORY__)

open Charon.Data
open Charon.Validation

// probs by category (see createSVMInput in Script.fsx)
let modelProbsTrain = @"..\..\..\comboModelTrain.csv"
let modelProbsValidate = @"..\..\..\comboModelValidate.csv"

let inline labelByClass c = List.findIndex ((=)c) categories |> float
let inline classByLabel lbl = List.nth categories lbl
//let classes = categories |> Seq.mapi (fun i c -> c, i) |> Map.ofSeq
//let labels  = categories |> Seq.mapi (fun i c -> i, c) |> Map.ofSeq

// "not a real question"; "not constructive"; "off topic"; "open"; "too localized"; reputation; undeleted; status
let loadModel fileName = 
    parseCsv fileName
    |> Seq.map (fun line -> line.[0..6] |> Array.map float, line.[7])
    |> Seq.toArray

open SVM

let chooseParameters problem parameters outFile =
    let c, gamma = ref 0., ref 0.
    ParameterSelection.Grid(problem, parameters, outFile, c, gamma)
    c, gamma

let trainSet = loadModel modelProbsTrain
let validateSet = loadModel modelProbsValidate

let prepareSvmData set = 
    set
    |> Array.map (fun (data, status) -> 
        Array.mapi (fun i v -> Node(i, v)) data, labelByClass status)
    |> Array.unzip

let xs, ys = prepareSvmData (trainSet |> Seq.take 10000 |> Seq.toArray)
let problem = Problem(ys.Length, ys, xs, xs.[0].Length)

let ps = Parameter(Probability = true, C = 512., Gamma = 0.00048828125)
//let c, gamma = chooseParameters problem ps "params.txt"

let svmModel = Training.Train(problem, ps)

let testXs, testYs = prepareSvmData validateSet
let preds = 
    testXs //|> Seq.take 20 
    |> Seq.map (fun x -> Prediction.PredictProbability(svmModel, x)) 
    |> Seq.map (Seq.mapi (fun i p -> classByLabel svmModel.ClassLabels.[i], p) >> Map.ofSeq)
    |> Seq.toArray

let benchmarkSvm preds cats =
    Seq.zip preds cats 
    |> Seq.map (fun (p, c) -> evaluate p c)
    |> Seq.average

benchmarkSvm preds (validateSet |> Seq.map snd)
