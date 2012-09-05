namespace Charon.Tests

open Charon
open Charon.Data
open NUnit.Framework
open FsUnit

[<TestFixture>] 
type ``Data Tests`` () =

    [<Test>] 
    member test.``Validate Post extraction from data line.`` ()=
        let data = [| "1"; ""; ""; ""; ""; ""; "Title"; "Body" |]
        let post = extractPost data
        post.Id |> should equal "1"
        post.Title |> should equal "Title"
        post.Body |> should equal "Body"

    [<Test>] 
    member test.``Validate preparation of submission data.`` ()=
        let data = 
            [ [| "1"; ""; ""; ""; ""; ""; "Title1"; "Body1" |];
              [| "2"; ""; ""; ""; ""; ""; "Title2"; "Body2" |] ]
            |> List.toSeq
        let fakeModel data = 
            if data.Title = "Title1" 
            then
               Map.empty
                  .Add("A", 0.1)
                  .Add("B", 0.9)
            else
               Map.empty
                  .Add("A", 0.2)
                  .Add("B", 0.8)
        let fakeCategories = [ "A"; "B" ]
        
        let predicted = 
            Data.prepareResults data fakeModel fakeCategories
            |> Seq.toList

        let expected = 
            [ "1,0.1,0.9";
              "2,0.2,0.8" ]
        predicted |> should equal expected
