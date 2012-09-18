let tags1 = [ "A"; "B"; "C" ]
let tags2 = [ "A"; "C"; "Z" ]
let tags3 = [ "C"; "B" ]
let sample  = [ tags1; tags2; tags3 ]

let tagPairs (tags: string list) =
    let rec extract (current: (string * string) list) (tokens: string list) =
        match tokens with
        | [] -> current
        | [_] -> current
        | hd :: tl ->
            let combos = tl |> List.map (fun t -> (hd, t))
            let newCurrent = List.append current combos
            extract newCurrent tl
    tags |> List.sort |> extract []

let tokensCount data =
    let update (acc: Map<(string * string), int>) (tags: string list) = 
        tags
        |> tagPairs
        |> Seq.fold (fun (a: Map<(string * string), int>) t -> 
            if a.ContainsKey(t) 
            then a.Add(t, a.[t] + 1) 
            else a.Add(t, 1)) acc
    data
    |> Seq.fold (fun acc post -> 
        update acc post) Map.empty 

let topTokens data min =
    data
    |> tokensCount
    |> Map.filter (fun k v -> v >= min)