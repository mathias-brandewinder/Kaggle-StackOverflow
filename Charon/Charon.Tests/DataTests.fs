namespace Charon.Tests

open System
open Charon
open Charon.Data
open NUnit.Framework
open FsUnit

[<TestFixture>] 
type ``Data Tests`` () =

    [<Test>] 
    member test.``Validate Post extraction from data line.`` ()=
        let data = [| "1"; "2/2/2012"; ""; "1/1/2011"; "2"; "3"; "Title"; "Body";"Tag1"; "Tag2"; "Tag3"; "Tag4"; "Tag5"; "3/3/2013" |]
        let post = extractPost data
        post.Id |> should equal "1"
        post.PostDate |> should equal (new DateTime(2012, 2, 2))
        post.UserDate |> should equal (new DateTime(2011, 1, 1))
        post.Reputation |> should equal 2
        post.Undeleted |> should equal 3
        post.Title |> should equal "Title"
        post.Body |> should equal "Body"
        post.Tag1 |> should equal "Tag1"
        post.Tag2 |> should equal "Tag2"
        post.Tag3 |> should equal "Tag3"
        post.Tag4 |> should equal "Tag4"
        post.Tag5 |> should equal "Tag5"
        post.CloseDate |> should equal (Some(new DateTime(2013, 3, 3)))

    [<Test>] 
    member test.``Validate preparation of submission data.`` ()=
        let data = 
            [ [| "1"; "2/2/2012"; ""; "1/1/2011"; "2"; "3"; "Title1"; "Body";"Tag1"; "Tag2"; "Tag3"; "Tag4"; "Tag5"; "3/3/2013" |];
              [| "2"; "2/2/2012"; ""; "1/1/2011"; "2"; "3"; "Title2"; "Body";"Tag1"; "Tag2"; "Tag3"; "Tag4"; "Tag5"; "3/3/2013" |] ]
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
