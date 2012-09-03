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

    [<Test>]
    member test.``Quality of zero proba predictions for a correct class should be infinity`` ()=
        let real = [1; 2]
        let predicted = [
            [1, 0.0; 2, 1.0]
            [1, 0.2; 2, 0.8] ]
        (quality predicted real) |> should equal infinity

open Charon.Data

[<TestFixture>]
type ``Preprocessing Tests`` ()=

    [<Test>]              
    member test.``Line starting with at least four spaces should be removed as code`` ()=
        let text = "Indent several lines of code by at least four spaces \n\n    let x = 1\n\n      x\n." 
        let expected = "Indent several lines of code by at least four spaces \n."
        removeCode text |> should equal expected

    [<Test>]
    member test.``Code inlined with backticks should be removed`` ()=
        let text = "Inline code with backticks `let x = 1\n  `."
        let expected = "Inline code with backticks ."
        removeCode text |> should equal expected

    [<Test>]
    member test.``Code at the beginning of question should be removed`` ()=
        let text = "    let sum a b = a + b\nBegins with code"
        let expected = "\nBegins with code"
        removeCode text |> should equal expected
