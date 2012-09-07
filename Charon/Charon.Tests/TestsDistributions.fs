namespace Charon.Tests

open Charon
open Charon.Distributions
open NUnit.Framework
open FsUnit

[<TestFixture>]
type ``Distributions Tests`` ()=

    [<Test>]              
    member test.``Normalize should re-map on [0,1]`` ()=
        let original = Map.empty
                          .Add("A", 5.0)
                          .Add("B", 15.0)
                          |> Map.toSeq
        let expected = Map.empty
                          .Add("A", 0.25)
                          .Add("B", 0.75)
        let actual = normalize original
        actual |> should equal expected

    [<Test>]              
    member test.``Combine should produce linear combination`` ()=
        let alpha = Map.empty
                       .Add("A", 10.0)
                       .Add("B", 20.0)
                          
        let bravo = Map.empty
                       .Add("A", 30.0)
                       .Add("B", 40.0)

        let categories = [ "A"; "B" ]

        let expected = Map.empty
                          .Add("A", 25.0)
                          .Add("B", 35.0)

        let actual = combine categories (0.25, alpha) (0.75, bravo)
        actual |> should equal expected