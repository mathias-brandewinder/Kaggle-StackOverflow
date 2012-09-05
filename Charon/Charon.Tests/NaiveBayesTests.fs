namespace Charon.Tests

open MachineLearning.NaiveBayes
open NUnit.Framework
open FsUnit

[<TestFixture>] 
type ``Naive Bayes Tests`` () =

    [<Test>] 
    member test.``Validate renormalization.`` ()=
        let bayes = Map.empty
                       .Add("Alpha", log 0.1)
                       .Add("Bravo", log 0.3)
        let expected = Map.empty
                          .Add("Alpha", 0.25)
                          .Add("Bravo", 0.75)
        let actual = renormalize (bayes |> Map.toSeq)

        actual.["Alpha"] |> should (equalWithin 0.01) expected.["Alpha"]
        actual.["Bravo"] |> should (equalWithin 0.01) expected.["Bravo"]

    [<Test>] 
    member test.``Validate renormalization with small values.`` ()=
        let bayes = Map.empty
                       .Add("Alpha", log 0.0000000001)
                       .Add("Bravo", log 0.0000000999)
        let expected = Map.empty
                          .Add("Alpha", 0.001)
                          .Add("Bravo", 0.999)
        let actual = renormalize (bayes |> Map.toSeq)

        actual.["Alpha"] |> should (equalWithin 0.0001) expected.["Alpha"]
        actual.["Bravo"] |> should (equalWithin 0.0001) expected.["Bravo"]