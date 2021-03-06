﻿namespace Charon

open System

type Post = 
    { Id: string; 
      PostDate: DateTime;
      UserDate: DateTime;
      Reputation: int;
      Undeleted: int;
      Title: string; 
      Body: string;
      Tag1: string;
      Tag2: string;
      Tag3: string;
      Tag4: string;
      Tag5: string; 
      OwnerUserId: string; }
    member this.Tags = 
        [ this.Tag1; this.Tag2; this.Tag3; this.Tag4; this.Tag5 ]
        |> List.filter (fun t -> not (String.IsNullOrWhiteSpace(t)))
    member this.DaysExperience =
        (this.PostDate - this.UserDate).Days

module Data =

    open System
    open System.IO
    open System.Text.RegularExpressions
    open Microsoft.VisualBasic.FileIO    

    let parseCsv (filePath: string) =

        use reader = new TextFieldParser(filePath)
        reader.TextFieldType <- FieldType.Delimited
        reader.SetDelimiters(",")
        [ while (not reader.EndOfData) do yield reader.ReadFields() ]

    let categories = [
        "not a real question";
        "not constructive";
        "off topic";
        "open";
        "too localized" ]

    type Category = NotReal | NotConstructive | OffTopic | Open | TooLocal

    let categoryToString cat =
        match cat with
        | NotReal         -> "not a real question"
        | NotConstructive -> "not constructive"
        | OffTopic        -> "off topic"
        | Open            -> "open"
        | TooLocal        -> "too localized"

    let stringToCategory text =
        match text with 
        | "not a real question" -> NotReal
        | "not constructive"    -> NotConstructive
        | "off topic"           -> OffTopic
        | "open"                -> Open
        | "too localized"       -> TooLocal
        | _                     -> failwith "ooops?"

    // Prior probability of each category, in overall sample
    let priors =
        Map.empty
           .Add("not a real question", 0.0091347705760047)
           .Add("not constructive",    0.0046458596397953)
           .Add("off topic",           0.00520096554605094)
           .Add("open",                0.979191390785063)
           .Add ("too localized",      0.00182701345308509)

    // Prior probability of each category, in training sample
    let trainingPriors =
        Map.empty
           .Add("not a real question", 0.22)
           .Add("not constructive",    0.11)
           .Add("off topic",           0.13)
           .Add("open",                0.50)
           .Add ("too localized",      0.04)

    // Uniform distribution over categories, for testing/benchmarking
    let uniformPriors =
        Map.empty
           .Add("not a real question", 0.2)
           .Add("not constructive",    0.2)
           .Add("off topic",           0.2)
           .Add("open",                0.2)
           .Add ("too localized",      0.2)

    // Name says it all - for testing/benchmarking
    let awfulPriors =
        Map.empty
           .Add("not a real question", 0.01)
           .Add("not constructive",    0.01)
           .Add("off topic",           0.01)
           .Add("open",                0.01)
           .Add ("too localized",      0.96)

   // Basic function to reduce questions to open vs. close
    let simplified group =
        match group with
        | "open" -> "open"
        | _      -> "closed"

    let readWordsFrequencies (filePath: string) =
        let data = parseCsv filePath |> Seq.toList
        categories 
        |> List.mapi (fun col cat ->
            cat, 
            0.0,
            data 
            |> List.map (fun line ->
                line.[0], Convert.ToDouble(line.[col + 1]))
            |> Map.ofList)

    let saveWordsFrequencies (filePath: string) (model: (string * float * Map<string, float>) seq) =
        let catMap =
            categories
            |> List.map (fun cat -> 
                Seq.find(fun (c, _, _) -> c = cat) model)
            |> List.map (fun (_, _, m) -> m)
        
        let refMap = List.head catMap |> Map.toSeq
        let convertToText =
            refMap       
            |> Seq.map (fun (word, _) ->
                catMap
                |> Seq.fold (fun acc freq -> acc + "," + freq.[word].ToString()) word)

        File.AppendAllLines(filePath, convertToText)
            
    // indices of the title and body columns
    let titleCol, bodyCol = 6, 7
    let tag1Col, tag2Col, tag3Col, tag4Col, tag5Col = 8, 9, 10, 11, 12

    let someDate text =
        let succ, date = DateTime.TryParse(text)
        match succ with
        | true  -> Some(date)
        | false -> None

    let extractPost (line: string []) =
        { Id          = line.[0];
          PostDate    = Convert.ToDateTime(line.[1]);
          UserDate    = Convert.ToDateTime(line.[3]);
          Reputation  = Convert.ToInt32(line.[4]);
          Undeleted   = Convert.ToInt32(line.[5]);
          Title       = line.[titleCol];
          Body        = line.[bodyCol];
          Tag1        = line.[tag1Col];
          Tag2        = line.[tag2Col];
          Tag3        = line.[tag3Col];
          Tag4        = line.[tag4Col];
          Tag5        = line.[tag5Col]; 
          OwnerUserId = line.[2];}

    let prepareResults (data: string [] seq)
                       (model: Post -> Map<string, float>) 
                       categories =
         data
         |> Seq.map (fun d -> extractPost d)
         |> Seq.map (fun d -> d.Id, model(d))
         |> Seq.map (fun (id, dist) -> 
            id :: List.map (fun cat -> dist.[cat].ToString()) categories)
         |> Seq.map (fun line -> String.Join(",", line))
              
    // create submission file
    let create sourceFile 
               resultFile 
               (model: Post -> Map<string, float>) 
               categories =

        let data = parseCsv sourceFile |> List.tail
        let contents = prepareResults data model categories
        File.WriteAllLines(resultFile, contents)

    // stat by user
    let inline freqByCategory s = 
        let total = Seq.length s |> float
        categories 
        |> Seq.map (fun cat -> cat, Seq.sumBy (fun q -> if snd q = cat then 1. / total else 0.) s)
        |> Map.ofSeq

    let getQuestionsByUser (data: #seq<Post * _>) = 
        data
        |> Seq.map (fun p -> (fst p).OwnerUserId, snd p)
        |> Seq.groupBy fst
        |> Seq.map (fun (usr, cats) -> usr, freqByCategory cats)
        |> Map.ofSeq 

    let newUserPriors =
        [ "not a real question", 0.
          "not constructive",    0.
          "off topic",           0.
          "open",                1.
          "too localized",       0. ]
        |> Map.ofList
 