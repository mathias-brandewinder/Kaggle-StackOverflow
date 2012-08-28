namespace Charon.Tests

open Charon.Validation
open NUnit.Framework
open FsUnit

[<TestFixture>] 
type ``Validation Tests`` () =

    [<Test>] 
    member test.``Single case quality should be log of proba of predicted.`` ()=
        let real = [1]
        let predicted = [ [1, 0.2; 2, 0.8] ]
        let expected = - log 0.2
        (quality predicted real) |> should equal expected

    [<Test>] 
    member test.``Multiple case quality should be average quality.`` ()=
        let real = [1; 1]
        let predicted = [ 
            [1, 0.5; 2, 0.5]
            [1, 0.1; 2, 0.9] ]
        let expected = List.average [- log 0.5; - log 0.1]
        (quality predicted real) |> should equal expected

    [<Test>] 
    member test.``Quality should match correct prediction.`` ()=
        let real = [1; 2]
        let predicted = [ 
            [1, 0.5; 2, 0.5]
            [1, 0.1; 2, 0.9] ]
        let expected = List.average [- log 0.5; - log 0.9]
        (quality predicted real) |> should equal expected
