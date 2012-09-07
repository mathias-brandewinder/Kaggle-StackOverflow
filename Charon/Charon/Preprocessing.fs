namespace Charon

module Preprocessing =

    open System
    open System.Text.RegularExpressions

    let options = RegexOptions.Compiled ||| RegexOptions.IgnoreCase

    // Regular Expression matching full words, case insensitive.
    let matchWords = new Regex(@"\w+", options)

    // Extract and count words from a string.
    // http://stackoverflow.com/a/2159085/114519        
    let wordsCount (text: string) =
        matchWords.Matches((text.ToLower()))
        |> Seq.cast<Match>
        |> Seq.groupBy (fun m -> m.Value)
        |> Seq.map (fun (value, groups) -> 
            value.ToLower(), (groups |> Seq.length))
        |> Map.ofSeq

    // Extracts all words used in a string.
    let vocabulary (text: string) =
        matchWords.Matches((text.ToLower()))
        |> Seq.cast<Match>
        |> Seq.map (fun m -> m.Value)
        |> Set.ofSeq

    // Extracts all words used in a dataset;
    // a Dataset is a sequence of "samples", 
    // each sample has a label (the class), and text.
    let extractWords (dataset: string seq) =
        dataset
        |> Seq.map (fun text -> vocabulary text)
        |> Set.unionMany 

    let codeRegex = Regex("((^|\\n)\s{4}\s*((?!\\n).)*)+|`[^`]+`", options)
    let httpAddrRegex = Regex("(http|https)://[^\s]*", options)

    let inline removeCode str = codeRegex.Replace(str, "")
    let inline removeLinks str = httpAddrRegex.Replace(str, "httpaddress")
    
    let preprocess str = (removeCode >> removeLinks) str

    // http://www.textfixer.com/resources/common-english-words.txt
    let stopWords = 
        let asString = "a,able,about,across,after,all,almost,also,am,among,an,and,any,are,as,at,be,because,been,but,by,can,cannot,could,dear,did,do,does,either,else,ever,every,for,from,get,got,had,has,have,he,her,hers,him,his,how,however,i,if,in,into,is,it,its,just,least,let,like,likely,may,me,might,most,must,my,neither,no,nor,not,of,off,often,on,only,or,other,our,own,rather,said,say,says,she,should,since,so,some,than,that,the,their,them,then,there,these,they,this,tis,to,too,twas,us,wants,was,we,were,what,when,where,which,while,who,whom,why,will,with,would,yet,you,your"
        asString.Split(',') |> Set.ofArray

    let filterStopWords (vocabulary: Set<string>) =        
        vocabulary 
        |> Set.partition (fun w -> Set.contains (w.ToLower()) stopWords)
        |> snd