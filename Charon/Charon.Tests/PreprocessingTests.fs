namespace Charon.Tests

open Charon
open Charon.Preprocessing
open NUnit.Framework
open FsUnit

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
