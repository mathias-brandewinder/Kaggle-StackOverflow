namespace Charon

type Post = 
    { Id: string; 
      Title: string; 
      Body: string }

module Data =

    open System
    open System.IO
    open System.Text.RegularExpressions
    open Microsoft.VisualBasic.FileIO    

    let parseCsv (filePath: string) =

        let reader = new TextFieldParser(filePath)
        reader.TextFieldType <- FieldType.Delimited
        reader.SetDelimiters(",")

        Seq.unfold (fun line ->
            if reader.EndOfData 
            then 
                reader.Close()
                None
            else Some(line, reader.ReadFields())) (reader.ReadFields())

    let categories = [
        "not a real question";
        "not constructive";
        "off topic";
        "open";
        "too localized" ]

    // Prior probability of each category
    let priors =
        Map.empty
           .Add("not a real question", 0.0091347705760047)
           .Add("not constructive",    0.0046458596397953)
           .Add("off topic",           0.00520096554605094)
           .Add("open",                0.979191390785063)
           .Add ("too localized",      0.00182701345308509)

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
            
    let opts = RegexOptions.Compiled ||| RegexOptions.IgnoreCase

    let codeRegex = Regex("((^|\\n)\s{4}\s*((?!\\n).)*)+|`[^`]+`", opts)
    let httpAddrRegex = Regex("(http|https)://[^\s]*", opts)

    let inline removeCode str = codeRegex.Replace(str, "")
    let inline removeLinks str = httpAddrRegex.Replace(str, "httpaddress")
    
    let preprocess str = (removeCode >> removeLinks) str

    let extractPost (line: string []) =
        { Id    = line.[0];
          Title = line.[6];
          Body  = line.[7] }

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

        let data = parseCsv sourceFile
        let contents = prepareResults data model categories
        File.WriteAllLines(resultFile, contents)
