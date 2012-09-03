namespace Charon

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
